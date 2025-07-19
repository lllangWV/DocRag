import logging
from abc import ABC
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from parquetdb import ParquetDB

CURRENT_DIR = Path(__file__).parent

logger = logging.getLogger(__name__)

class PyArrowInterface(ABC):
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    def to_dict(self):
        data = {}
        for field in fields(self):
            data[field.name] = getattr(self, field.name)
        return data
    
    @classmethod
    def get_fields(cls):
        return [pa.field(field.name, pa.infer_type([getattr(cls, field.name)])) for field in fields(cls)]
    
    def to_table(self):
        return pa.Table.from_pandas(self.to_pandas())
    
    def to_pandas(self):
        return pd.DataFrame([self.to_dict()])
    
    @staticmethod
    def to_table_from_list(data: list):
        table = None
        for item in data:
            if table is None:
                table = item.to_table()
            else:
                table = pa.concat_tables([table, item.to_table()])
        return table
    
@dataclass
class Page(PyArrowInterface):
    document_id: int = 1 
    local_page_id: int = 1
    image: bytes = b"F"
    height: int = 100
    width: int = 100
    ingestion_timestamp: datetime = datetime.now()
    dpi: int = 300
    
@dataclass
class PageElement(PyArrowInterface):
    document_id: int = 1
    page_id: int = 1
    element_type: str = "text"
    image: bytes = b"F"
    confidence: float = 0.5
    bbox: tuple[int, int, int, int] = (0, 0, 100, 100)
    sort_order: int = 1

    
@dataclass
class Document(PyArrowInterface):
    page_count: int = 1
    hash: str = ""
    filepath: str = ""
    ingestion_timestamp: datetime = datetime.now()
    
    @staticmethod
    def remomve_duplicate(documents: list["Document"]):
        return cls(filepath=str(path.relative_to(os.getcwd())))




class PageStore(ParquetDB):
    initial_fields = Page.get_fields()
    def __init__(self, path: str = "pagedb", **kwargs):
        super().__init__(path, initial_fields=self.initial_fields, convert_to_fixed_shape=False, **kwargs)
        
    def add_page(self, page: Page):
        return self.create(page.to_table())
        
    def add_pages(self, pages: list[Page], **kwargs):
        pages = Page.to_table_from_list(pages)
        return self.create(pages, **kwargs)
    
    def read_pages(self, **kwargs):
        return self.read(**kwargs)
    
    def get_pages(self, document_id: list[int] | int, page_id: list[int] | int, **kwargs):
        if isinstance(document_id, int):
            document_id = [document_id]
        if isinstance(page_id, int):
            page_id = [page_id]
        filters = [pc.field("document_id").isin(document_id), pc.field("id").isin(page_id)]
        return self.read(filters= filters, **kwargs)
    
    def update_page(self, data: dict, **kwargs):
        self.update(data, update_keys = ["document_id", "id"], **kwargs)
        
    def delete_page(self, document_id: int, page_id: int, **kwargs):
        self.delete(filters= [pc.field("document_id") == document_id, pc.field("id") == page_id], **kwargs)
        

class PageElementStore(ParquetDB):
    initial_fields = PageElement.get_fields()
    def __init__(self, path: Path, **kwargs):
        super().__init__(path, initial_fields=self.initial_fields, convert_to_fixed_shape=False, **kwargs)

    def add_page_element(self, page_element: PageElement):
        return self.create(page_element.to_table())
    
    def add_page_elements(self, page_elements: list[PageElement], **kwargs):
        page_elements = PageElement.to_table_from_list(page_elements)
        return self.create(page_elements, **kwargs)
    
    def read_page_elements(self, **kwargs):
        return self.read(**kwargs)

class DocumentStore(ParquetDB):
    initial_fields = Document.get_fields()
    def __init__(self, path: Path, **kwargs):
        super().__init__(path, initial_fields=self.initial_fields, convert_to_fixed_shape=False, **kwargs)
        
    def add_document(self, document: Document):
        return self.create(document.to_table())
        
    def add_documents(self, documents: list[Document], **kwargs):
        if len(documents) != 0:
            documents = Document.to_table_from_list(documents)
            documents = self._check_if_document_exists(documents)
        
        if len(documents) == 0:
            logger.warning("No new documents to add to the store.")
            return []
        return self.create(documents, **kwargs)
        
    def read_documents(self, **kwargs):
        return self.read(**kwargs)
    
    def _check_if_document_exists(self, documents: pa.Table):
        incoming_doc_hashes = documents['hash'].combine_chunks()
        existing_doc_hashes = self.read(columns=["hash"])['hash'].combine_chunks()
        is_in = pc.is_in(incoming_doc_hashes, existing_doc_hashes)
        return documents.filter(pc.invert(is_in))

@dataclass
class DocumentImageStore:
    page_db: PageStore
    page_element_db: PageElementStore
    document_db: DocumentStore
        
    @classmethod
    def from_path(cls, page_image_store_path: Path = "page_image_store"):
        page_db_path = page_image_store_path / "pages"
        page_element_db_path = page_image_store_path / "page_elements"
        document_db_path = page_image_store_path / "documents"
        page_db = PageStore(page_db_path)
        page_element_db = PageElementStore(page_element_db_path)
        document_db = DocumentStore(document_db_path)
        return cls(page_db=page_db, page_element_db=page_element_db, document_db=document_db)
    
        
    
