import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from PIL import Image, ImageDraw

from docrag.core.image_store import ImageStore


class TestImageStore:
    
    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def image_store(self, temp_store_path):
        """Create an ImageStore instance for testing."""
        return ImageStore(store_path=str(temp_store_path))
    
    @pytest.fixture
    def sample_pil_image(self):
        """Create a sample PIL image for testing."""
        image = Image.new('RGB', (100, 100), color='red')
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), "Test", fill='white')
        return image
    
    @pytest.fixture
    def sample_numpy_image(self):
        """Create a sample numpy array image for testing."""
        # Create a simple gradient
        gradient = np.linspace(0, 255, 50*50, dtype=np.uint8).reshape(50, 50)
        return np.stack([gradient, gradient//2, gradient//3], axis=-1)
    
    @pytest.fixture
    def sample_image_file(self, temp_store_path, sample_pil_image):
        """Create a sample image file for testing."""
        image_path = temp_store_path / "test_image.png"
        sample_pil_image.save(image_path)
        return image_path
    
    def test_init_image_store(self, temp_store_path):
        """Test ImageStore initialization."""
        store = ImageStore(store_path=str(temp_store_path))
        
        assert store.store_path == temp_store_path
        assert store.image_store_path == temp_store_path / "images"
        assert store.db_path == temp_store_path / "metadata"
        assert store.image_store_path.exists()
        assert store.db_path.exists()
    
    def test_init_with_custom_metadata_fields(self, temp_store_path):
        """Test ImageStore initialization with custom metadata fields."""
        custom_fields = [
            pa.field("category", pa.string()),
            pa.field("priority", pa.int32())
        ]
        
        store = ImageStore(
            store_path=str(temp_store_path),
            additional_fields=custom_fields
        )
        
        # Check that custom fields are in the schema
        field_names = [field.name for field in store.db.schema]
        assert "category" in field_names
        assert "priority" in field_names
    
    def test_create_image_hash(self, image_store):
        """Test SHA256 hash creation."""
        test_data = b"test image data"
        hash1 = image_store._create_image_hash(test_data)
        hash2 = image_store._create_image_hash(test_data)
        
        assert len(hash1) == 64  # SHA256 produces 64 character hex string
        assert hash1 == hash2  # Same data should produce same hash
        
        # Different data should produce different hash
        different_data = b"different test data"
        hash3 = image_store._create_image_hash(different_data)
        assert hash1 != hash3
    
    def test_get_image_path_from_hash(self, image_store):
        """Test hash-based directory structure creation."""
        test_hash = "abcd1234567890" + "0" * 50  # 64 char hash
        path = image_store._get_image_path_from_hash(test_hash)
        
        expected_path = image_store.image_store_path / "ab" / "cd" / f"{test_hash}.png"
        assert path == expected_path
        
        # Directory should be created
        assert path.parent.exists()
        assert path.parent.name == "cd"
        assert path.parent.parent.name == "ab"
    
    def test_add_pil_image(self, image_store, sample_pil_image):
        """Test adding a PIL Image."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        assert len(hash_val) == 64
        assert image_store.image_exists(hash_val)
        
        # Check file was created in correct location
        image_path = image_store._get_image_path_from_hash(hash_val)
        assert image_path.exists()
        
        # Check metadata was stored
        metadata = image_store.get_metadata(hash_val)
        assert metadata is not None
        assert metadata['original_filename'] == "test.png"
        assert metadata['width'] == 100
        assert metadata['height'] == 100
        assert metadata['format'] == 'PNG'
    
    def test_add_numpy_image(self, image_store, sample_numpy_image):
        """Test adding a numpy array image."""
        hash_val = image_store.add_image(sample_numpy_image, "numpy_test.png")
        
        assert len(hash_val) == 64
        assert image_store.image_exists(hash_val)
        
        # Verify we can retrieve the image
        retrieved = image_store.get_image(hash_val)
        assert retrieved is not None
        assert retrieved.size == (50, 50)
    
    def test_add_image_from_file_path(self, image_store, sample_image_file):
        """Test adding an image from file path."""
        hash_val = image_store.add_image(sample_image_file)
        
        assert len(hash_val) == 64
        assert image_store.image_exists(hash_val)
        
        # Should automatically use the filename
        metadata = image_store.get_metadata(hash_val)
        assert metadata['original_filename'] == "test_image.png"
    
    def test_add_image_from_string_path(self, image_store, sample_image_file):
        """Test adding an image from string path."""
        hash_val = image_store.add_image(str(sample_image_file), "custom_name.png")
        
        assert len(hash_val) == 64
        assert image_store.image_exists(hash_val)
        
        metadata = image_store.get_metadata(hash_val)
        assert metadata['original_filename'] == "custom_name.png"
    
    def test_add_image_with_additional_metadata(self, image_store, sample_pil_image):
        """Test adding image with custom metadata."""
        additional_metadata = {"category": "test", "priority": 1}
        
        # First add custom fields to store
        store_with_custom = ImageStore(
            store_path=str(image_store.store_path),
            additional_fields=[
                pa.field("category", pa.string()),
                pa.field("priority", pa.int32())
            ]
        )
        
        hash_val = store_with_custom.add_image(
            sample_pil_image, 
            "test.png", 
            additional_metadata
        )
        
        metadata = store_with_custom.get_metadata(hash_val)
        assert metadata['category'] == "test"
        assert metadata['priority'] == 1
    
    def test_add_multiple_images(self, image_store):
        """Test adding multiple images at once."""
        images = [
            Image.new('RGB', (50, 50), 'red'),
            Image.new('RGB', (60, 60), 'blue'),
            Image.new('RGB', (70, 70), 'green')
        ]
        filenames = ["red.png", "blue.png", "green.png"]
        
        hashes = image_store.add_images(images, filenames)
        
        assert len(hashes) == 3
        for hash_val in hashes:
            assert len(hash_val) == 64
            assert image_store.image_exists(hash_val)
    
    def test_duplicate_detection(self, image_store, sample_pil_image):
        """Test that duplicate images are detected and not re-stored."""
        hash1 = image_store.add_image(sample_pil_image, "first.png")
        hash2 = image_store.add_image(sample_pil_image, "duplicate.png")
        
        assert hash1 == hash2
        
        # Should only have one entry in metadata
        stats = image_store.get_stats()
        assert stats['total_images'] == 1
    
    def test_get_image(self, image_store, sample_pil_image):
        """Test retrieving an image by hash."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        retrieved = image_store.get_image(hash_val)
        assert retrieved is not None
        assert retrieved.size == sample_pil_image.size
        
        # Test non-existent hash
        fake_hash = "a" * 64
        assert image_store.get_image(fake_hash) is None
    
    def test_get_image_path(self, image_store, sample_pil_image):
        """Test getting image file path."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        path = image_store.get_image_path(hash_val)
        assert path is not None
        assert path.exists()
        assert path.name == f"{hash_val}.png"
        
        # Test non-existent hash
        fake_hash = "a" * 64
        assert image_store.get_image_path(fake_hash) is None
    
    def test_image_exists(self, image_store, sample_pil_image):
        """Test checking if image exists."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        assert image_store.image_exists(hash_val) is True
        
        fake_hash = "a" * 64
        assert image_store.image_exists(fake_hash) is False
    
    def test_get_metadata(self, image_store, sample_pil_image):
        """Test retrieving image metadata."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        metadata = image_store.get_metadata(hash_val)
        assert metadata is not None
        assert metadata['hash'] == hash_val
        assert metadata['original_filename'] == "test.png"
        assert metadata['width'] == 100
        assert metadata['height'] == 100
        assert 'ingestion_timestamp' in metadata
        
        # Test non-existent hash
        fake_hash = "a" * 64
        assert image_store.get_metadata(fake_hash) is None
    
    def test_search_images(self, image_store):
        """Test searching images by metadata."""
        # Add custom fields for search testing
        store_with_search = ImageStore(
            store_path=str(image_store.store_path),
            additional_fields=[
                pa.field("category", pa.string()),
                pa.field("priority", pa.int32())
            ]
        )
        
        # Add images with different metadata
        img1 = Image.new('RGB', (50, 50), 'red')
        img2 = Image.new('RGB', (50, 50), 'blue')
        img3 = Image.new('RGB', (50, 50), 'green')
        
        store_with_search.add_image(img1, "red.png", {"category": "test", "priority": 1})
        store_with_search.add_image(img2, "blue.png", {"category": "demo", "priority": 2})
        store_with_search.add_image(img3, "green.png", {"category": "test", "priority": 3})
        
        # Search by category
        results = store_with_search.search_images(category="test")
        assert len(results) == 2
        
        # Search by priority
        results = store_with_search.search_images(priority=1)
        assert len(results) == 1
        
        # Search with multiple conditions
        results = store_with_search.search_images(category="test", priority=3)
        assert len(results) == 1
    
    def test_delete_image(self, image_store, sample_pil_image):
        """Test deleting an image."""
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        # Verify image exists
        assert image_store.image_exists(hash_val)
        image_path = image_store._get_image_path_from_hash(hash_val)
        assert image_path.exists()
        
        # Delete image
        success = image_store.delete_image(hash_val)
        assert success is True
        
        # Verify image is gone
        assert image_store.image_exists(hash_val) is False
        assert not image_path.exists()
        assert image_store.get_metadata(hash_val) is None
    
    def test_get_stats(self, image_store):
        """Test getting store statistics."""
        # Empty store
        stats = image_store.get_stats()
        assert stats['total_images'] == 0
        assert stats['total_size_bytes'] == 0
        
        # Add some images
        images = [
            Image.new('RGB', (50, 50), 'red'),
            Image.new('RGB', (100, 100), 'blue')
        ]
        image_store.add_images(images, ["red.png", "blue.png"])
        
        stats = image_store.get_stats()
        assert stats['total_images'] == 2
        assert stats['total_size_bytes'] > 0
        assert stats['average_size_bytes'] > 0
        assert stats['unique_formats'] == 1  # All PNG
        assert 'store_path' in stats
    
    def test_cleanup_orphaned_files(self, image_store, sample_pil_image):
        """Test cleaning up orphaned image files."""
        # Add an image normally
        hash_val = image_store.add_image(sample_pil_image, "test.png")
        
        # Create an orphaned file (file without metadata)
        orphaned_hash = "b" * 64
        orphaned_path = image_store._get_image_path_from_hash(orphaned_hash)
        orphaned_path.parent.mkdir(parents=True, exist_ok=True)
        sample_pil_image.save(orphaned_path)
        
        # Run cleanup
        cleanup_stats = image_store.cleanup_orphaned_files()
        
        assert cleanup_stats['total_files_scanned'] == 2
        assert cleanup_stats['orphaned_files_found'] == 1
        assert cleanup_stats['orphaned_files_deleted'] == 1
        assert cleanup_stats['known_hashes_count'] == 1
        
        # Orphaned file should be gone
        assert not orphaned_path.exists()
        # Original file should still exist
        original_path = image_store._get_image_path_from_hash(hash_val)
        assert original_path.exists()
    
    def test_error_handling_invalid_image_type(self, image_store):
        """Test error handling for invalid image types."""
        with pytest.raises(ValueError, match="Unsupported image type"):
            image_store.add_image("not an image type")
    
    def test_error_handling_nonexistent_file(self, image_store):
        """Test error handling for non-existent file paths."""
        with pytest.raises(FileNotFoundError):
            image_store.add_image(Path("nonexistent_file.png"))
    
    def test_bytes_input(self, image_store, sample_pil_image):
        """Test adding image from bytes."""
        # Convert PIL image to bytes
        import io
        img_byte_arr = io.BytesIO()
        sample_pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        hash_val = image_store.add_image(image_bytes, "from_bytes.png")
        
        assert len(hash_val) == 64
        assert image_store.image_exists(hash_val)
        
        # Should be able to retrieve the image
        retrieved = image_store.get_image(hash_val)
        assert retrieved is not None
        assert retrieved.size == sample_pil_image.size
    
    def test_directory_structure_consistency(self, image_store):
        """Test that directory structure is consistent with hash."""
        test_hashes = [
            "ab" + "c" * 62,  # Hash starting with 'ab'
            "12" + "3" * 62,  # Hash starting with '12'
            "ff" + "e" * 62   # Hash starting with 'ff'
        ]
        
        for test_hash in test_hashes:
            path = image_store._get_image_path_from_hash(test_hash)
            
            # Check directory structure
            assert path.parent.name == test_hash[2:4]  # Second level dir
            assert path.parent.parent.name == test_hash[:2]  # First level dir
            assert path.name == f"{test_hash}.png"  # Filename
            
            # Directory should be created
            assert path.parent.exists()
    
    def test_schema_includes_all_default_fields(self, image_store):
        """Test that schema includes all expected default fields."""
        expected_fields = {
            'hash', 'image_path', 'original_filename', 'file_size',
            'width', 'height', 'format', 'ingestion_timestamp'
        }
        
        actual_fields = {field.name for field in image_store.db.schema}
        
        assert expected_fields.issubset(actual_fields)
    