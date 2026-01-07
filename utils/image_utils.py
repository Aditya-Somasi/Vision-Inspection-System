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
    Draw bounding boxes with labels on image.
    
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
    
    for i, box in enumerate(boxes):
        # Get coordinates
        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("width", 100))
        h = int(box.get("height", 100))
        
        # Get styling
        label = box.get("label", f"Defect {i+1}")
        severity = box.get("severity", "MODERATE")
        
        # Color by severity
        colors = {
            "CRITICAL": (0, 0, 255),    # Red
            "MODERATE": (0, 165, 255),  # Orange
            "COSMETIC": (0, 255, 255)   # Yellow
        }
        color = colors.get(severity, (0, 255, 0))
        thickness = 3 if severity == "CRITICAL" else 2
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            img,
            (x, y - label_size[1] - 10),
            (x + label_size[0] + 10, y),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    # Save annotated image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    logger.info(f"Saved annotated image: {output_path}")
    
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
    
    # Overlay for heatmap effect - reduced opacity for more subtle highlighting
    overlay = img.copy()
    alpha = 0.4  # Lower opacity to make highlights less misleading if coordinates are approximate
    
    for i, defect in enumerate(defects):
        bbox = defect.get("bbox", {})
        severity = defect.get("safety_impact", "MODERATE")
        defect_type = defect.get("type", "DEFECT").upper()
        location = defect.get("location", "").lower()
        
        # Choose color based on severity
        if severity == "CRITICAL":
            color = (0, 0, 255)     # Red BGR
            glow_color = (50, 50, 255)
        elif severity == "MODERATE":
            color = (0, 140, 255)   # Orange BGR
            glow_color = (50, 160, 255)
        else:
            color = (0, 200, 255)   # Yellow BGR
            glow_color = (50, 220, 255)
        
        
        # Check if defect is truly widespread (NO bbox provided AND explicit language)
        # Only trigger this if VLM explicitly can't localize + uses words like "entire" or "everywhere"
        widespread_keywords = ["entire surface", "everywhere", "whole component", "complete surface"]
        location_lower = location.lower()
        
        has_valid_bbox = (bbox and 
                         bbox.get("x") is not None and 
                         bbox.get("y") is not None and
                         bbox.get("width", 0) > 0 and
                         bbox.get("height", 0) > 0)
        
        is_widespread = (not has_valid_bbox and 
                        any(kw in location_lower for kw in widespread_keywords))
        
        if is_widespread:
            # CREATE GENERAL OVERLAY for widespread defects
            # Avoid specific markers (1a, 1b) which can be misleading if placement is arbitrary
            
            # Create a semi-transparent overlay over the entire image (or significant portion)
            # Use a vignette-style gradient to highlight the center but cover mostly everything
            gradient = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw a large filled circle covering most of the image
            center_x, center_y = width // 2, height // 2
            radius = int(min(width, height) * 0.4)
            cv2.circle(gradient, (center_x, center_y), radius, color, -1)
            
            # Apply heavy blur to make it a soft glow
            gradient = cv2.GaussianBlur(gradient, (151, 151), 0)
            
            # Blend lightly
            overlay = cv2.addWeighted(overlay, 1.0, gradient, 0.4, 0)
            
            # Add a single central label
            label = f"#{i+1}"
            cv2.circle(overlay, (center_x, center_y), 30, (255, 255, 255), -1)
            cv2.putText(overlay, label, (center_x - 10, center_y + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Add defect info text at top with "WIDESPREAD" indicator
            info_text = f"#{i+1} {defect_type} - {severity} (WIDESPREAD / ENTIRE SURFACE)"
            
            # Scale text box based on image width
            text_scale = max(width / 1000.0, 1.0)
            box_h = int(35 * text_scale)
            cv2.rectangle(overlay, (10, i * box_h + 5), (width - 10, i * box_h + box_h), (0, 0, 0), -1)
            cv2.putText(overlay, info_text, (15, i * box_h + int(box_h*0.7)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_scale, (255, 255, 255), 1)
            continue
        
        # Extract coordinates
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("width", 100))
        h = int(bbox.get("height", 100))
        
        # Add generous padding (30%) to account for VLM bbox imprecision
        padding_x = int(w * 0.3)
        padding_y = int(h * 0.3)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(width - x, w + 2 * padding_x)
        h = min(height - y, h + 2 * padding_y)
        
        # Draw filled rectangle (the heatmap effect) - more transparent
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        
        # Draw bright border for visibility
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        
        # Draw glow border for critical defects
        if severity == "CRITICAL":
            for offset in range(1, 6):
                alpha_border = 0.8 - (offset * 0.15)
                cv2.rectangle(overlay, 
                            (x - offset, y - offset), 
                            (x + w + offset, y + h + offset), 
                            glow_color, 2)
        
        # Draw defect number marker
        label = f"#{i + 1}"
        marker_size = 20
        cv2.circle(overlay, (x + marker_size, y + marker_size), marker_size, (255, 255, 255), -1)
        cv2.putText(overlay, label, (x + marker_size - 8, y + marker_size + 6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add defect type label INSIDE the box
        type_label = defect_type[:15]  # Truncate if too long
        label_y = y + h - 10
        label_x = x + 5
        
        # Background for label
        text_size = cv2.getTextSize(type_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (label_x - 2, label_y - text_size[1] - 2), 
                     (label_x + text_size[0] + 2, label_y + 2), (0, 0, 0), -1)
        
        # Label text
        cv2.putText(overlay, type_label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Blend overlay with original
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Dynamic scaling for legend based on image width
    # Base reference width is 1000px
    scale_factor = max(width / 1000.0, 1.0)
    
    legend_height = int(40 * scale_factor)
    font_scale = 0.5 * scale_factor
    box_width = int(20 * scale_factor)
    box_height = int(20 * scale_factor)
    padding = int(10 * scale_factor)
    text_offset = int(5 * scale_factor)
    spacing = int(140 * scale_factor)
    
    legend = np.zeros((legend_height, width, 3), dtype=np.uint8)
    legend.fill(30)  # Dark gray background
    
    # Draw legend items
    legend_items = [
        ((0, 0, 255), "CRITICAL"),
        ((0, 140, 255), "MODERATE"),
        ((0, 200, 255), "COSMETIC")
    ]
    
    x_pos = int(20 * scale_factor)
    y_box_top = int((legend_height - box_height) / 2)
    y_text_base = int((legend_height + box_height) / 2) - text_offset
    
    for color, text in legend_items:
        cv2.rectangle(legend, (x_pos, y_box_top), (x_pos + box_width, y_box_top + box_height), color, -1)
        cv2.putText(legend, text, (x_pos + box_width + padding, y_text_base), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        x_pos += spacing
    
    # Combine result with legend
    result_with_legend = np.vstack([result, legend])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_with_legend)
    logger.info(f"Saved heatmap overlay: {output_path}")
    
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
