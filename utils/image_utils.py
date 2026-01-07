"""
Image utilities for Vision Inspection System.
Handles image loading, preprocessing, resizing, and validation.
"""

import io
from pathlib import Path
from typing import Tuple, Optional, BinaryIO

from PIL import Image
import cv2
import numpy as np

from utils.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__, level=config.log_level, component="IMAGE_UTILS")


def load_image(image_path: Path) -> Image.Image:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image format is unsupported
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        img = Image.open(image_path)
        img.load()  # Force load to catch corrupt images
        logger.debug(f"Loaded image: {image_path.name}, size: {img.size}, mode: {img.mode}")
        return img
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def resize_image(
    img: Image.Image,
    max_dimension: int = None
) -> Image.Image:
    """
    Resize image to fit within max dimension while preserving aspect ratio.
    
    Args:
        img: PIL Image
        max_dimension: Maximum width or height (defaults to config)
        
    Returns:
        Resized PIL Image
    """
    max_dimension = max_dimension or config.max_image_dimension
    
    width, height = img.size
    
    if width <= max_dimension and height <= max_dimension:
        return img
    
    # Calculate new size
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    logger.debug(f"Resized image from {img.size} to {resized.size}")
    
    return resized


def get_image_info(image_path: Path) -> dict:
    """
    Get image metadata.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with image info (size, format, mode, etc.)
    """
    img = load_image(image_path)
    
    return {
        "path": str(image_path),
        "filename": image_path.name,
        "width": img.size[0],
        "height": img.size[1],
        "mode": img.mode,
        "format": img.format,
        "size_bytes": image_path.stat().st_size
    }


def validate_image(
    image_path: Path,
    allowed_extensions: list = None,
    max_size_mb: float = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate image file.
    
    Args:
        image_path: Path to image file
        allowed_extensions: List of allowed extensions
        max_size_mb: Maximum file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    allowed_extensions = allowed_extensions or config.allowed_extensions_list
    max_size_mb = max_size_mb or config.max_file_size_mb
    
    # Check existence
    if not image_path.exists():
        return False, "File does not exist"
    
    # Check extension
    ext = image_path.suffix.lower().lstrip(".")
    if ext not in allowed_extensions:
        return False, f"Invalid extension '{ext}'. Allowed: {allowed_extensions}"
    
    # Check file size
    size_mb = image_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Try to load image
    try:
        img = load_image(image_path)
        if img.size[0] < 10 or img.size[1] < 10:
            return False, "Image too small (minimum 10x10 pixels)"
    except Exception as e:
        return False, f"Invalid image file: {e}"
    
    return True, None


def draw_bounding_boxes(
    image_path: Path,
    boxes: list,
    output_path: Path
) -> Path:
    """
    Draw PRECISE bounding boxes with labels on image.
    Uses corner markers and crosshairs for small defects.
    
    Args:
        image_path: Path to original image
        boxes: List of dicts with x, y, width, height, label, color
        output_path: Path to save annotated image
        
    Returns:
        Path to annotated image
    """
    # Load with OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    # VLM models receive images resized to max 1024px
    # We need to scale coordinates from model space to original image space
    MODEL_MAX_SIZE = 1024
    original_max_dim = max(img_width, img_height)
    if original_max_dim > MODEL_MAX_SIZE:
        scale_factor = original_max_dim / MODEL_MAX_SIZE
    else:
        scale_factor = 1.0
    
    for i, box in enumerate(boxes):
        # Get coordinates from model
        raw_x = box.get("x", 0)
        raw_y = box.get("y", 0)
        raw_w = box.get("width", 100)
        raw_h = box.get("height", 100)
        
        # Scale coordinates
        x = max(0, int(raw_x * scale_factor))
        y = max(0, int(raw_y * scale_factor))
        w = int(raw_w * scale_factor)
        h = int(raw_h * scale_factor)
        
        # Clamp to image bounds
        x = min(x, img_width - 10)
        y = min(y, img_height - 10)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Extract label specific info
        # Extract just the number from label "#1" -> "1"
        label_full = box.get("label", f"#{i+1}")
        try:
            # Try to extract number if format is "#N"
            label_text = label_full.replace("#", "")
        except:
            label_text = str(i+1)
            
        severity = box.get("severity", "MODERATE")
        
        # PROFESSIONAL STYLE:
        # 1. Thin red bounding box (or based on severity, but user asked for red style)
        # 2. White circular marker with red border
        # 3. Black number inside
        
        # Box color - User requested Red thin line. 
        # We'll use Red for critical/moderate to keep it standard, maybe Orange for cosmetic?
        # Actually standardizing on Red/Orange makes it look cleaner like the reference
        box_color = (0, 0, 255) # Red for high visibility standard
        
        if severity == "COSMETIC":
            box_color = (0, 200, 255) # Yellow-ish for cosmetic
            
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness)
        
        # Draw Circular Marker (#1, #2)
        # Position: Top-left corner of the box usually, unless box is tiny then center
        marker_radius = int(max(img_width, img_height) * 0.04) # Dynamic size 4% of image
        marker_radius = max(35, min(marker_radius, 80)) # Clamp size (min 35px, max 80px)
        
        # Determine marker position
        if w < 50 or h < 50:
             # Tiny box -> Center marker
             marker_cx = x + w // 2
             marker_cy = y + h // 2
        else:
             # Normal box -> Top left corner, slightly offset inside
             marker_cx = x + marker_radius + 5
             marker_cy = y + marker_radius + 5
             
        # Draw white filled circle
        cv2.circle(img, (marker_cx, marker_cy), marker_radius, (255, 255, 255), -1)
        
        # Draw red border on circle
        cv2.circle(img, (marker_cx, marker_cy), marker_radius, box_color, 3)
        
        # Draw number
        font_scale = marker_radius / 20.0 * 0.7
        thickness_text = max(2, int(font_scale * 2))
        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)[0]
        text_x = int(marker_cx - text_size[0] / 2)
        text_y = int(marker_cy + text_size[1] / 2)
        
        cv2.putText(img, label_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness_text)
                   
    # Save annotated image
    cv2.imwrite(str(output_path), img)
    return output_path


def create_heatmap_overlay(
    image_path: Path,
    defects: list,
    output_path: Path,
    alpha: float = 0.4
) -> Path:
    """
    Create semi-transparent heatmap overlay on defect regions.
    
    Args:
        image_path: Path to original image
        defects: List of defect dicts with bbox and severity
        output_path: Path to save overlay image
        alpha: Transparency level (0-1)
        
    Returns:
        Path to overlay image
    """
    logger.info(f"Creating heatmap overlay for {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # VLM models receive images resized to max 1024px
    # Scale coordinates from model space to original image space
    MODEL_MAX_SIZE = 1024
    original_max_dim = max(width, height)
    if original_max_dim > MODEL_MAX_SIZE:
        scale_factor = original_max_dim / MODEL_MAX_SIZE
    else:
        scale_factor = 1.0
        
    # Initialize blank mask for heatmap (float 0-1)
    # We will accumulate "heat" on this mask
    heat_mask = np.zeros((height, width), dtype=np.float32)
    
    has_defects = False
    
    for defect in defects:
        has_defects = True
        bbox = defect.get("bbox", {})
        severity = defect.get("safety_impact", "MODERATE")
        
        # Determine heat intensity based on severity
        intensity = 1.0 if severity == "CRITICAL" else 0.7 if severity == "MODERATE" else 0.4
        
        # Check for widespread defects (no bbox)
        widespread_keywords = ["entire surface", "everywhere", "whole component", "complete surface"]
        location_lower = defect.get("location", "").lower()
        
        has_valid_bbox = (bbox and 
                         bbox.get("x") is not None and 
                         bbox.get("y") is not None and
                         bbox.get("width", 0) > 0 and
                         bbox.get("height", 0) > 0)
        
        is_widespread = (not has_valid_bbox and 
                        any(kw in location_lower for kw in widespread_keywords))
        
        if is_widespread:
            # Add general heat to the center area
            center_x, center_y = width // 2, height // 2
            axes = (int(width * 0.4), int(height * 0.4))
            cv2.ellipse(heat_mask, (center_x, center_y), axes, 0, 0, 360, intensity, -1)
            continue
            
        if has_valid_bbox:
            # Extract and scale coordinates
            raw_x = bbox.get("x", 0)
            raw_y = bbox.get("y", 0)
            raw_w = bbox.get("width", 100)
            raw_h = bbox.get("height", 100)
            
            x = int(raw_x * scale_factor)
            y = int(raw_y * scale_factor)
            w = int(raw_w * scale_factor)
            h = int(raw_h * scale_factor)
            
            # Center of the defect
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw an ellipse for the hotspot (more natural than rectangle)
            # Size proportional to the defect area, but enforce a minimum size for visibility
            min_dim = min(width, height)
            min_axis = int(min_dim * 0.08) # Min 8% of image size
            
            axis_x = max(int(w * 0.8), min_axis)
            axis_y = max(int(h * 0.8), min_axis)
            
            axes = (axis_x, axis_y)
            cv2.ellipse(heat_mask, (center_x, center_y), axes, 0, 0, 360, intensity, -1)

    if not has_defects:
        # Just save original if no defects
        cv2.imwrite(str(output_path), img)
        return output_path

    # Apply heavy Gaussian blur to create the smooth thermal gradient look
    # Blur size proportional to image size
    blur_ksize = int(min(width, height) * 0.15) | 1  # Must be odd
    heat_mask = cv2.GaussianBlur(heat_mask, (blur_ksize, blur_ksize), 0)
    
    # Normalize mask to 0-255 range for colormap application
    heat_mask_norm = cv2.normalize(heat_mask, None, 0, 255, cv2.NORM_MINMAX)
    heat_mask_uint8 = heat_mask_norm.astype(np.uint8)
    
    # Apply JET colormap (Classic thermal look: Blue -> Cyan -> Yellow -> Red)
    heatmap_color = cv2.applyColorMap(heat_mask_uint8, cv2.COLORMAP_JET)
    
    # Create an opacity mask where there is heat (so cold/blue areas don't darken the image)
    # We want "no heat" to be transparent, not blue
    # The mask itself (before colormap) is perfect for opacity
    opacity = heat_mask.copy()
    np.clip(opacity, 0, 0.6, out=opacity) # Cap opacity at 0.6
    
    # Blend manually for better control
    # destination = src1 * (1 - alpha) + src2 * alpha
    heatmap_overlay = img.copy()
    
    for c in range(3):
        heatmap_overlay[:, :, c] = (
            img[:, :, c] * (1.0 - opacity) + 
            heatmap_color[:, :, c] * opacity
        )
        
    cv2.imwrite(str(output_path), heatmap_overlay)
    
    return output_path



def create_side_by_side_comparison(
    original_path: Path,
    annotated_path: Path,
    output_path: Path,
    labels: Tuple[str, str] = ("Original Input", "AI Analysis Layer")
) -> Path:
    """
    Create side-by-side comparison image.
    
    Args:
        original_path: Path to original image
        annotated_path: Path to annotated/overlay image
        output_path: Path to save comparison
        labels: Tuple of (left_label, right_label)
        
    Returns:
        Path to comparison image
    """
    logger.info("Creating side-by-side comparison")
    
    # Load images
    original = cv2.imread(str(original_path))
    annotated = cv2.imread(str(annotated_path))
    
    if original is None or annotated is None:
        raise ValueError("Failed to load images for comparison")
    
    # Resize to same height (increased for better clarity)
    target_height = 800
    
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        ratio = target_h / h
        new_w = int(w * ratio)
        return cv2.resize(img, (new_w, target_h))
    
    original = resize_to_height(original, target_height)
    annotated = resize_to_height(annotated, target_height)
    
    # Create header bar
    header_height = 40
    total_width = original.shape[1] + annotated.shape[1] + 10  # 10px divider
    header = np.zeros((header_height, total_width, 3), dtype=np.uint8)
    header.fill(45)  # Dark gray
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    left_label, right_label = labels
    
    # Left label (centered over left image)
    left_center = original.shape[1] // 2
    text_size = cv2.getTextSize(left_label, font, 0.7, 2)[0]
    cv2.putText(header, left_label, 
               (left_center - text_size[0] // 2, 28), 
               font, 0.7, (255, 255, 255), 2)
    
    # Right label (centered over right image)
    right_center = original.shape[1] + 10 + annotated.shape[1] // 2
    text_size = cv2.getTextSize(right_label, font, 0.7, 2)[0]
    cv2.putText(header, right_label, 
               (right_center - text_size[0] // 2, 28), 
               font, 0.7, (255, 255, 255), 2)
    
    # Create divider
    divider = np.zeros((target_height, 10, 3), dtype=np.uint8)
    divider.fill(45)
    
    # Combine horizontally
    combined = np.hstack([original, divider, annotated])
    
    # Add header on top
    result = np.vstack([header, combined])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    logger.info(f"Saved comparison image: {output_path}")
    
    return output_path


def create_status_stamp(
    verdict: str,
    output_path: Path,
    size: Tuple[int, int] = (300, 100)
) -> Path:
    """
    Create a status stamp image (PASSED/REJECTED/REVIEW).
    
    Args:
        verdict: "SAFE", "UNSAFE", or "REQUIRES_HUMAN_REVIEW"
        output_path: Path to save stamp
        size: (width, height) of stamp
        
    Returns:
        Path to stamp image
    """
    width, height = size
    
    # Create transparent background
    stamp = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Determine stamp text and color
    if verdict == "SAFE":
        text = "PASSED"
        color = (0, 200, 0, 255)       # Green
        border_color = (0, 150, 0, 255)
    elif verdict == "UNSAFE":
        text = "REJECTED"
        color = (0, 0, 200, 255)       # Red
        border_color = (0, 0, 150, 255)
    else:
        text = "REVIEW"
        color = (0, 140, 255, 255)     # Orange
        border_color = (0, 100, 200, 255)
    
    # Draw rounded rectangle border
    cv2.rectangle(stamp, (5, 5), (width-5, height-5), border_color, 4)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1.5, 4)[0]
    x = (width - text_size[0]) // 2
    y = (height + text_size[1]) // 2
    cv2.putText(stamp, text, (x, y), font, 1.5, color, 4)
    
    # Save as PNG (to preserve transparency)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), stamp)
    logger.info(f"Saved status stamp: {output_path}")
    
    return output_path
