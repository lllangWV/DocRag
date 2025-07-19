import logging
import shutil
from abc import ABC
from pathlib import Path

from PIL import Image

# from docrag.core.page import Page, extract_page_elements_with_yolo
from docrag.core.doclayout_extraction import (
    extract_page_elements_with_yolo,
    get_doclayout_model,
)

logger = logging.getLogger("docrag")
logger.setLevel(logging.DEBUG)

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data"

pdfs_path = DATA_DIR / "pdfs"





pdf_paths = list(pdfs_path.glob("*.pdf"))
store_path = DATA_DIR / "DocumentImageStore"


from docrag.pipeline import ingest_documents

if store_path.exists():
    shutil.rmtree(store_path)
    
# print(pdf_paths)

# pdf_path = CURRENT_DIR / "test.pdf"
ingest_documents(pdf_paths, store_path)


# from docrag.core.document_image_store import DocumentImageStore

# store = DocumentImageStore.from_path(store_path)




# table = store.page_db.read(columns=["id", "document_id"])


# print(table.shape)





