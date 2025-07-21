import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
from pypdf import PdfReader

from docrag.core.doclayout_extraction import extract_page_elements_with_yolo
from docrag.core.document_image_store import (
    Document,
    DocumentImageStore,
    ExtractedPageElement,
    Page,
    PageElement,
)
from docrag.core.utils import (
    bytes_to_pil_image,
    extract_pages_as_images,
    get_file_hash,
    pil_image_to_bytes,
)

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")


@dataclass
class DocLayoutExtractionConfig:
    model_weights: str = "doclayout_yolo_docstructbench_imgsz1024.pt"
    image_size: int = 1024
    confidence_threshold: float = 0.2
    device: str = "cpu"  

    def to_dict(self):
        tmp_dict = {}
        for field in fields(self):
            tmp_dict[field.name] = getattr(self, field.name)
        return tmp_dict

def get_existing_hashes(store: DocumentImageStore) -> set[str]:
    try:
        # Assuming your DocumentStore can read its data into a pandas DataFrame
        existing_docs_table = store.document_db.read(columns=["hash"])
        existing_hashes = set(existing_docs_table['hash'])
        print(f"Found {len(existing_hashes)} existing documents in the store.")
    except FileNotFoundError:
        existing_hashes = set()
        print("No existing document store found. Starting fresh.")
    return existing_hashes

def ingest_documents(
    pdf_paths: str | Path | Iterable[Path], 
    store_path: str | Path = "DocumentImageStore",
    doclayout_extraction_config: DocLayoutExtractionConfig = DocLayoutExtractionConfig(),
    dpi: int = 300
) -> DocumentImageStore:
    """
    Processes a list of PDF documents and saves them to a DocumentImageStore.

    This is the main library-level entry point for the ingestion pipeline.
    It handles the entire workflow of reading PDFs, extracting pages and
    layout elements, and saving the results.

    Args:
        pdf_paths: An iterable of Path objects for the PDFs to ingest.
        store_path: The root directory for the DocumentImageStore.

    Returns:
        An instance of the DocumentImageStore connected to the newly ingested data.
    """
    print(f"Initializing document store at: {store_path}")
    if isinstance(pdf_paths, str) or isinstance(pdf_paths, Path):
        pdf_paths = [pdf_paths]
        
    store = DocumentImageStore.from_path(Path(store_path))

    documents = []
    current_hashes = []
    for pdf_path in pdf_paths:
        logger.info(f"Processing: {pdf_path.name}...")

        try:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {e}")
            continue
        
        hash_string = get_file_hash(pdf_path)
        if hash_string in current_hashes:
            user_logger.warning(f"Skipping {pdf_path.name} because it is a duplicate of another document.")
            continue
        
        current_hashes.append(hash_string)

        document = Document(
            page_count=page_count,
            hash=get_file_hash(pdf_path),
            filepath=str(pdf_path.relative_to(os.getcwd())),
            ingestion_timestamp=datetime.now()
        )
        documents.append(document)
        

    document_ids = store.document_db.add_documents(documents) 
    if len(document_ids) == 0:
        user_logger.warning("No new documents to add to the store.")
        return store
    
    user_logger.info(f"Added {len(document_ids)} documents to the store.")

    
    document_df = store.document_db.read(ids=document_ids, columns=["filepath", "id"]).to_pandas()
    
    
    
    pages = []
    for document_id, pdf_path in zip(document_df["id"], document_df["filepath"]):
        images = extract_pages_as_images(pdf_path, dpi=dpi)
        for page_id, image in enumerate(images):
            page = Page(
                local_page_id=page_id,
                document_id=document_id,
                image=pil_image_to_bytes(image),
                height=image.height,
                width=image.width,
                ingestion_timestamp=datetime.now(),
                dpi=dpi
            )
            pages.append(page)
            
    page_ids = store.page_db.add_pages(pages)
    page_df= store.page_db.read(ids=page_ids, columns=["id", "document_id", "image"]).to_pandas()
    
    
    
    images = []
    for image in page_df["image"]:
        image = bytes_to_pil_image(image)
        images.append(image)
        
    page_elements_list = extract_page_elements_with_yolo(images, **doclayout_extraction_config.to_dict())
    page_elements = []
    for page_id, document_id, page_elements in zip(page_df["id"], page_df["document_id"], page_elements_list):
        for page_element in page_elements:
            page_element.page_id = page_id
            page_element.document_id = document_id
        page_elements.append(page_element)
        
    store.page_element_db.add_page_elements(page_elements)
    
    return store


def enrich_elements_with_llm(
    store: DocumentImageStore,
    model_name: str,
    elements_to_process: pa.Table | None = None
):
    """
    Processes element images with an LLM to extract text and summaries.

    Args:
        store: The main DocumentImageStore instance.
        model_name: The identifier for the LLM being used.
        elements_to_process: Optional table of specific elements to process.
                             If None, it processes all unprocessed elements.
    """
    # 1. Find out what work needs to be done
    processed_ids = store.extracted_page_element_db.get_processed_element_ids()
    print(f"Found {len(processed_ids)} elements that are already processed.")

    if elements_to_process is None:
        # Read all elements if a specific subset isn't provided
        elements_to_process = store.page_element_db.read_all()

    # Filter out elements that are already processed
    unprocessed_elements_filter = ~pc.field("element_id").isin(processed_ids)
    elements_to_process = elements_to_process.filter(unprocessed_elements_filter)
    
    if len(elements_to_process) == 0:
        print("No new elements to process.")
        return

    print(f"Found {len(elements_to_process)} new elements to process with model: {model_name}")

    # 2. Process in batches
    results = []
    for batch in elements_to_process.to_batches(max_chunksize=10): # Process 10 at a time
        batch_df = batch.to_pandas()
        for _, element in batch_df.iterrows():
            element_image = element['image']
            element_id = element['element_id']

            # --- LLM API Call ---
            # extracted_text = get_text_from_image(element_image, model=model_name)
            # summary = get_summary(extracted_text, model=model_name)
            # This is where your actual call to OpenAI, Gemini, etc. would go
            extracted_text = f"Text for element {element_id}" # Placeholder
            summary = f"Summary for element {element_id}" # Placeholder
            # --------------------

            results.append(ExtractedPageElement(
                element_id=element_id,
                extracted_text=extracted_text,
                summary=summary,
                llm_model=model_name
            ))

    # 3. Save the results to the new store
    if results:
        print(f"Saving {len(results)} new content extractions to the store.")
        store.content_db.add_content(results)

    print("Enrichment complete.")



if __name__ == "__main__":
    store = DocumentImageStore.from_path("DocumentImageStore")
    
    
    
    
    enrich_elements_with_llm(store, "gpt-4o")