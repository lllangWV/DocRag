import hashlib
import io
import logging
from pathlib import Path
from typing import List, Union

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)
user_logger = logging.getLogger("user")

def pil_image_to_bytes(pil_image, format="PNG"):
    """Converts a PIL Image to bytes in the specified format."""
    if pil_image is None:
        return None
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def bytes_to_pil_image(img_bytes):
    """Converts image bytes back to a PIL Image."""
    if img_bytes is None:
        return None
    img_byte_arr = io.BytesIO(img_bytes)
    pil_image = Image.open(img_byte_arr)
    return pil_image


def extract_pages_as_images(pdf_path: Union[str, Path], dpi: int = 300) -> List[Image.Image]:
    """
    Extract all pages from PDF as PIL Images.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for PDF to image conversion
        verbose: Whether to print progress information

    Returns:
        List of PIL Image objects, one per page
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        user_logger.error(f"PDF file not found: {pdf_path}")
        return []

    try:
        images = convert_from_path(pdf_path, dpi=dpi)

        return images
    except Exception as e:
        user_logger.error(f"Error extracting pages from PDF: {e}")
        return []
    
    
    

def get_file_hash(file_path: Path, block_size: int = 65536) -> str:
    """
    Calculates the SHA256 hash of a file by reading it in chunks.
    
    Args:
        file_path: The path to the file.

    Returns:
        The hexadecimal SHA256 hash of the file's content.
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:  # Open the file in binary mode
        # Read the file in chunks to handle large files without using too much memory
        chunk = f.read(block_size)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = f.read(block_size)
    return hasher.hexdigest()