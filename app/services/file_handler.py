"""
File handling service for Vision Inspection System.
Separates file I/O logic from UI code.
"""

import hashlib
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from utils.config import UPLOAD_DIR
from utils.logger import setup_logger

logger = setup_logger(__name__, component="FILE_HANDLER")


def validate_image(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded image file.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
    MAX_SIZE_MB = 10
    
    if not uploaded_file:
        return False, "No file uploaded"
    
    # Check file type
    if uploaded_file.type not in ALLOWED_TYPES:
        return False, f"Invalid file type: {uploaded_file.type}. Allowed: {', '.join(ALLOWED_TYPES)}"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_SIZE_MB:
        return False, f"File too large: {file_size_mb:.1f}MB. Maximum: {MAX_SIZE_MB}MB"
    
    return True, ""


def save_uploaded_file(uploaded_file) -> Optional[Path]:
    """
    Save uploaded file to disk with validation.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to saved file, or None if failed
    """
    # Validate first
    is_valid, error = validate_image(uploaded_file)
    if not is_valid:
        logger.error(f"File validation failed: {error}")
        return None
    
    try:
        # Ensure upload directory exists
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(uploaded_file.getvalue()[:1024]).hexdigest()[:8]
        
        # Get original extension
        original_name = Path(uploaded_file.name)
        ext = original_name.suffix.lower() or ".jpg"
        
        # Create filename
        filename = f"{timestamp}_{content_hash}{ext}"
        filepath = UPLOAD_DIR / filename
        
        # Write file
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"Saved uploaded file: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return None


def get_file_info(filepath: Path) -> dict:
    """Get metadata about a file."""
    if not filepath.exists():
        return {"exists": False}
    
    stat = filepath.stat()
    return {
        "exists": True,
        "name": filepath.name,
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "extension": filepath.suffix.lower()
    }
