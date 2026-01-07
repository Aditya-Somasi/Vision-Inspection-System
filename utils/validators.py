"""
Input validators for Vision Inspection System.
Provides validation functions for user inputs and API requests.
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import re
import uuid

from utils.config import config


def validate_criticality(value: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate criticality level input.
    
    Args:
        value: Criticality level string
        
    Returns:
        Tuple of (is_valid, error_message, normalized_value)
    """
    valid_levels = ["low", "medium", "high"]
    normalized = value.lower().strip()
    
    if normalized not in valid_levels:
        return False, f"Invalid criticality. Must be one of: {valid_levels}", value
    
    return True, None, normalized


def validate_domain(value: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate domain input.
    
    Args:
        value: Domain string (optional)
        
    Returns:
        Tuple of (is_valid, error_message, normalized_value)
    """
    if not value:
        return True, None, None
    
    # Normalize: lowercase, replace spaces with underscores
    normalized = value.lower().strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'[^a-z0-9_-]', '', normalized)
    
    if len(normalized) > 100:
        return False, "Domain name too long (max 100 characters)", value
    
    return True, None, normalized


def validate_image_path(path: str) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Validate image file path.
    
    Args:
        path: Path string
        
    Returns:
        Tuple of (is_valid, error_message, Path object)
    """
    try:
        image_path = Path(path)
    except Exception as e:
        return False, f"Invalid path: {e}", None
    
    if not image_path.exists():
        return False, f"File not found: {path}", None
    
    if not image_path.is_file():
        return False, f"Not a file: {path}", None
    
    # Check extension
    ext = image_path.suffix.lower().lstrip(".")
    if ext not in config.allowed_extensions_list:
        return False, f"Invalid file type: {ext}", None
    
    # Check file size
    size_mb = image_path.stat().st_size / (1024 * 1024)
    if size_mb > config.max_file_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {config.max_file_size_mb}MB)", None
    
    if size_mb == 0:
        return False, "File is empty", None
    
    return True, None, image_path


def validate_user_notes(value: Optional[str]) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate user notes input.
    
    Args:
        value: User notes string (optional)
        
    Returns:
        Tuple of (is_valid, error_message, sanitized_value)
    """
    if not value:
        return True, None, None
    
    # Trim whitespace
    sanitized = value.strip()
    
    if len(sanitized) > 1000:
        return False, "Notes too long (max 1000 characters)", value
    
    return True, None, sanitized


def validate_request_id(value: Optional[str]) -> str:
    """
    Validate or generate request ID.
    
    Args:
        value: Request ID string (optional)
        
    Returns:
        Valid request ID
    """
    if value and len(value) >= 8:
        # Sanitize: only alphanumeric and dashes
        sanitized = re.sub(r'[^a-zA-Z0-9-]', '', value)
        if len(sanitized) >= 8:
            return sanitized[:36]  # Max 36 chars (UUID length)
    
    # Generate new ID
    return str(uuid.uuid4())[:8]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators
    filename = Path(filename).name
    
    # Replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    name = Path(sanitized).stem[:50]
    ext = Path(sanitized).suffix[:10]
    
    return f"{name}{ext}"


def validate_inspection_context(
    image_path: str,
    criticality: str,
    domain: Optional[str] = None,
    user_notes: Optional[str] = None
) -> Tuple[bool, list, dict]:
    """
    Validate complete inspection context.
    
    Args:
        image_path: Path to image
        criticality: Criticality level
        domain: Domain (optional)
        user_notes: User notes (optional)
        
    Returns:
        Tuple of (is_valid, errors, validated_context)
    """
    errors = []
    context = {}
    
    # Validate image path
    valid, error, path = validate_image_path(image_path)
    if not valid:
        errors.append(f"Image: {error}")
    else:
        context["image_path"] = str(path)
    
    # Validate criticality
    valid, error, value = validate_criticality(criticality)
    if not valid:
        errors.append(f"Criticality: {error}")
    else:
        context["criticality"] = value
    
    # Validate domain
    valid, error, value = validate_domain(domain)
    if not valid:
        errors.append(f"Domain: {error}")
    else:
        context["domain"] = value
    
    # Validate user notes
    valid, error, value = validate_user_notes(user_notes)
    if not valid:
        errors.append(f"Notes: {error}")
    else:
        context["user_notes"] = value
    
    return len(errors) == 0, errors, context
