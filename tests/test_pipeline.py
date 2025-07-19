from pathlib import Path

import pytest

from docrag.core.document_image_store import DocumentImageStore
from docrag.pipeline import ingest_documents
from tests.utils import DATA_DIR


@pytest.fixture
def page_dirpath():
    return DATA_DIR / "pages"

@pytest.fixture
def pdfs_path():
    return DATA_DIR / "documents"

@pytest.fixture
def document_image_store_path(tmp_path):
    return Path(tmp_path) / "document_image_store"

@pytest.fixture
def document_image_store(document_image_store_path):
    return DocumentImageStore.from_path(document_image_store_path)


class TestPipeline:
    def test_ingest_documents(self, pdfs_path, document_image_store_path):
        pdf_paths = list(pdfs_path.glob("*.pdf"))
        
        assert len(pdf_paths) == 3
        
        document_image_store = DocumentImageStore.from_path(document_image_store_path)
        
        ingest_documents(pdf_paths, document_image_store_path)
        
        assert document_image_store.document_db.read().shape[0] == len(pdf_paths) - 1 # -1 because the duplicate is skipped
        
        ingest_documents(pdf_paths, document_image_store_path)
        
        assert document_image_store.document_db.read().shape[0] == len(pdf_paths) - 1 # -1 because the duplicate is skipped
        
        