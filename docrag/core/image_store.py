
import hashlib
import io
import logging
import shutil
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
from parquetdb import ParquetDB
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Image metadata structure for ParquetDB."""
    hash: str = ""
    image_path: str = ""
    original_filename: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    format: str = "PNG"
    ingestion_timestamp: datetime = None
    metadata: dict = {}
    
    def __post_init__(self):
        if self.ingestion_timestamp is None:
            self.ingestion_timestamp = datetime.now()
    
    @classmethod
    def get_fields(cls):
        """Get PyArrow fields for this dataclass."""
        return [
            pa.field("hash", pa.string()),
            pa.field("image_path", pa.string()),
            pa.field("original_filename", pa.string()),
            pa.field("file_size", pa.int64()),
            pa.field("width", pa.int32()),
            pa.field("height", pa.int32()),
            pa.field("format", pa.string()),
            pa.field("ingestion_timestamp", pa.timestamp('ms'))
        ]
    
    def to_dict(self):
        """Convert to dictionary."""
        data = {}
        for field in fields(self):
            data[field.name] = getattr(self, field.name)
        return data
    
    def to_table(self):
        """Convert to PyArrow table."""
        import pandas as pd
        return pa.Table.from_pandas(pd.DataFrame([self.to_dict()]))


class ImageMetadataStore(ParquetDB):
    """ParquetDB store for image metadata."""
    
    def __init__(self, path: Path, additional_fields: List[pa.Field] = None, **kwargs):
        # Get base fields from ImageMetadata
        initial_fields = ImageMetadata.get_fields()
        
        # Add any additional custom fields
        if additional_fields:
            initial_fields.extend(additional_fields)
            
        super().__init__(path, initial_fields=initial_fields, **kwargs)
    
    def add_image_metadata(self, metadata_dict: dict) -> List[int]:
        """Add image metadata and return the generated ID(s)."""
        # Convert scalars to lists for PyArrow table creation
        array_dict = {key: [value] for key, value in metadata_dict.items()}
        table = pa.table(array_dict)
        return self.create(table)


class ImageStore:
    """
    An image store that uses ParquetDB for metadata and organizes images 
    in a hash-based directory structure.
    
    Directory structure: /path/ab/cd/abcd1234...5678 
    - First 2 chars of hash: first level directory
    - Next 2 chars: second level directory  
    - Full hash: filename
    """
    
    def __init__(self, 
                 store_path: str = "image_store", 
                 additional_fields: List[pa.Field] = None,
                 **kwargs):
        """
        Initialize the ImageStore.
        
        Args:
            store_path: Root directory for the image store
            additional_fields: Additional fields for the ParquetDB schema
            **kwargs: Additional arguments passed to ParquetDB
        """
        self.store_path = Path(store_path)
        self.image_store_path = self.store_path / "images"
        self.db_path = self.store_path / "metadata"
        
        # Create directories
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.image_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ParquetDB with proper schema
        self.db = ImageMetadataStore(self.db_path, additional_fields, **kwargs)
        
    def _create_image_hash(self, image_data: bytes) -> str:
        """Create SHA256 hash from image bytes."""
        return hashlib.sha256(image_data).hexdigest()
    
    def _get_image_path_from_hash(self, image_hash: str) -> Path:
        """
        Generate the file path from hash using the directory structure:
        /path/ab/cd/abcd1234...5678
        """
        # First 2 characters for first level directory
        level1 = image_hash[:2]
        # Next 2 characters for second level directory
        level2 = image_hash[2:4]
        
        # Create the directory structure
        dir_path = self.image_store_path / level1 / level2
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Full hash as filename (without extension, we'll add .png)
        return dir_path / f"{image_hash}.png"
    
    def _process_image(self, image: Union[Image.Image, bytes, np.ndarray, Path, str], 
                      original_filename: Optional[str] = None) -> tuple[bytes, ImageMetadata]:
        """
        Process an image input and return bytes and metadata.
        
        Args:
            image: Image in various formats
            original_filename: Original filename if available
            
        Returns:
            Tuple of (image_bytes, ImageMetadata)
        """
        # Convert input to PIL Image first
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise ValueError(f"Unsupported image type: {type(image)}")
            pil_image = Image.open(image_path)
            if original_filename is None:
                original_filename = image_path.name
        elif isinstance(image, bytes):
            pil_image = Image.open(io.BytesIO(image))
            if original_filename is None:
                original_filename = "unknown"
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            if original_filename is None:
                original_filename = "array_image"
        elif isinstance(image, Image.Image):
            pil_image = image
            if original_filename is None:
                original_filename = "pil_image"
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to bytes (PNG format)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Create metadata object
        metadata = ImageMetadata(
            original_filename=original_filename,
            file_size=len(image_bytes),
            width=pil_image.width,
            height=pil_image.height,
            format='PNG',
            ingestion_timestamp=datetime.now()
        )
        
        return image_bytes, metadata
        
    def add_image(self, image: Union[Image.Image, bytes, np.ndarray, Path, str], 
                  original_filename: Optional[str] = None,
                  additional_metadata: Optional[dict] = None) -> str:
        """
        Add a single image to the store.
        
        Args:
            image: Image in various formats
            original_filename: Original filename if available
            additional_metadata: Additional metadata fields
            
        Returns:
            The SHA256 hash of the added image
        """
        # Process the image
        image_bytes, metadata = self._process_image(image, original_filename)
        
        # Generate hash
        image_hash = self._create_image_hash(image_bytes)
        
        # Check if image already exists
        if self.image_exists(image_hash):
            logger.info(f"Image with hash {image_hash} already exists. Skipping.")
            return image_hash
        
        # Get the storage path
        image_path = self._get_image_path_from_hash(image_hash)
        
        # Save the image file
        with open(image_path, 'wb') as f:
            f.write(image_bytes)
        
        # Update metadata with hash and path
        metadata.hash = image_hash
        metadata.image_path = str(image_path.relative_to(self.store_path))
        
        # Convert to dict and add any additional metadata
        metadata_dict = metadata.to_dict()
        if additional_metadata:
            metadata_dict.update(additional_metadata)
        
        # Add to database
        self.db.add_image_metadata(metadata_dict)
        
        logger.info(f"Added image with hash {image_hash} to store")
        return image_hash
        
    def add_images(self, images: List[Union[Image.Image, bytes, np.ndarray, Path, str]], 
                   original_filenames: Optional[List[str]] = None,
                   additional_metadata: Optional[List[dict]] = None) -> List[str]:
        """
        Add multiple images to the store.
        
        Args:
            images: List of images in various formats
            original_filenames: List of original filenames
            additional_metadata: List of additional metadata dicts
            
        Returns:
            List of SHA256 hashes of added images
        """
        if original_filenames is None:
            original_filenames = [None] * len(images)
        if additional_metadata is None:
            additional_metadata = [{}] * len(images)
            
        hashes = []
        for i, image in enumerate(images):
            filename = original_filenames[i] if i < len(original_filenames) else None
            metadata = additional_metadata[i] if i < len(additional_metadata) else {}
            hash_val = self.add_image(image, filename, metadata)
            hashes.append(hash_val)
            
        return hashes
    
    def get_image(self, image_hash: str) -> Optional[Image.Image]:
        """
        Retrieve an image by its hash.
        
        Args:
            image_hash: SHA256 hash of the image
            
        Returns:
            PIL Image or None if not found
        """
        image_path = self._get_image_path_from_hash(image_hash)
        
        if not image_path.exists():
            logger.warning(f"Image file not found: {image_path}")
            return None
            
        try:
            return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image {image_hash}: {e}")
            return None
    
    def get_image_path(self, image_hash: str) -> Optional[Path]:
        """
        Get the file path for an image by its hash.
        
        Args:
            image_hash: SHA256 hash of the image
            
        Returns:
            Path to the image file or None if not found
        """
        image_path = self._get_image_path_from_hash(image_hash)
        return image_path if image_path.exists() else None
    
    def image_exists(self, image_hash: str) -> bool:
        """
        Check if an image exists in the store.
        
        Args:
            image_hash: SHA256 hash of the image
            
        Returns:
            True if image exists, False otherwise
        """
        # Check both metadata and file existence
        try:
            metadata_exists = len(self.db.read(filters=[pa.compute.equal(pa.compute.field("hash"), image_hash)])) > 0
            file_path = self._get_image_path_from_hash(image_hash)
            file_exists = file_path.exists()
            return metadata_exists and file_exists
        except Exception:
            return False
    
    def get_metadata(self, image_hash: str) -> Optional[dict]:
        """
        Get metadata for an image by its hash.
        
        Args:
            image_hash: SHA256 hash of the image
            
        Returns:
            Dictionary of metadata or None if not found
        """
        try:
            result = self.db.read(filters=[pa.compute.equal(pa.compute.field("hash"), image_hash)])
            if len(result) > 0:
                return result.to_pandas().iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Error retrieving metadata for {image_hash}: {e}")
            return None
    
    def search_images(self, **filters) -> pa.Table:
        """
        Search for images based on metadata filters.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            PyArrow table with matching results
        """
        # Convert filters to PyArrow expressions
        conditions = []
        for field, value in filters.items():
            if isinstance(value, (list, tuple)):
                # Handle 'in' conditions
                conditions.append(pa.compute.is_in(pa.compute.field(field), value))
            else:
                # Handle equality conditions
                conditions.append(pa.compute.equal(pa.compute.field(field), value))
        
        if conditions:
            # For single condition, use directly; for multiple, combine with AND
            if len(conditions) == 1:
                return self.db.read(filters=conditions)
            else:
                filter_expr = conditions[0]
                for condition in conditions[1:]:
                    filter_expr = pa.compute.and_(filter_expr, condition)
                return self.db.read(filters=[filter_expr])
        else:
            return self.db.read()
    
    def delete_image(self, image_hash: str) -> bool:
        """
        Delete an image from the store.
        
        Args:
            image_hash: SHA256 hash of the image
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete metadata
            self.db.delete(filters=[pa.compute.equal(pa.compute.field("hash"), image_hash)])
            
            # Delete file
            image_path = self._get_image_path_from_hash(image_hash)
            if image_path.exists():
                image_path.unlink()
                
            logger.info(f"Deleted image with hash {image_hash}")
            return True
        except Exception as e:
            logger.error(f"Error deleting image {image_hash}: {e}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the image store.
        
        Returns:
            Dictionary with store statistics
        """
        try:
            metadata_table = self.db.read()
            total_images = len(metadata_table)
            
            if total_images > 0:
                df = metadata_table.to_pandas()
                total_size = df['file_size'].sum()
                avg_size = df['file_size'].mean()
                unique_formats = df['format'].nunique()
            else:
                total_size = avg_size = unique_formats = 0
            
            return {
                'total_images': total_images,
                'total_size_bytes': int(total_size),
                'average_size_bytes': int(avg_size),
                'unique_formats': int(unique_formats),
                'store_path': str(self.store_path)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def cleanup_orphaned_files(self) -> dict:
        """
        Clean up orphaned image files that don't have metadata entries.
        
        Returns:
            Dictionary with cleanup statistics
        """
        # Get all hashes from metadata
        try:
            metadata_table = self.db.read(columns=['hash'])
            known_hashes = set(metadata_table['hash'].to_pylist())
        except Exception:
            known_hashes = set()
        
        # Scan all image files
        orphaned_files = []
        total_files = 0
        
        for level1_dir in self.image_store_path.iterdir():
            if not level1_dir.is_dir() or len(level1_dir.name) != 2:
                continue
                
            for level2_dir in level1_dir.iterdir():
                if not level2_dir.is_dir() or len(level2_dir.name) != 2:
                    continue
                    
                for image_file in level2_dir.glob("*.png"):
                    total_files += 1
                    # Extract hash from filename (remove .png extension)
                    file_hash = image_file.stem
                    
                    if file_hash not in known_hashes:
                        orphaned_files.append(image_file)
        
        # Delete orphaned files
        deleted_count = 0
        for orphaned_file in orphaned_files:
            try:
                orphaned_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting orphaned file {orphaned_file}: {e}")
        
        return {
            'total_files_scanned': total_files,
            'orphaned_files_found': len(orphaned_files),
            'orphaned_files_deleted': deleted_count,
            'known_hashes_count': len(known_hashes)
        }