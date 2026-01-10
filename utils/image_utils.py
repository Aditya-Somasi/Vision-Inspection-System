"""
Image utilities for Vision Inspection System.
Handles image loading, preprocessing, resizing, and validation.
"""

import io
import math
from pathlib import Path
from typing import Tuple, Optional, BinaryIO, List, Dict, Any

from PIL import Image
import cv2
import numpy as np

from utils.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__, level=config.log_level, component="IMAGE_UTILS")


def _convert_bbox_to_pixels(
    bbox: Dict[str, Any],
    img_width: int,
    img_height: int,
) -> Optional[Dict[str, float]]:
    """Convert percentage-or-pixel bbox to pixel coordinates."""
    if not bbox:
        return None

    try:
        raw_x = float(bbox.get("x"))
        raw_y = float(bbox.get("y"))
        raw_w = float(bbox.get("width"))
        raw_h = float(bbox.get("height"))
    except (TypeError, ValueError):
        return None

    if img_width <= 0 or img_height <= 0:
        return None

    is_percentage = (
        0.0 <= raw_x <= 100.0
        and 0.0 <= raw_y <= 100.0
        and 0.0 < raw_w <= 100.0
        and 0.0 < raw_h <= 100.0
        and raw_x + raw_w <= 100.0 + 1e-3
        and raw_y + raw_h <= 100.0 + 1e-3
    )

    if is_percentage:
        x = (raw_x / 100.0) * img_width
        y = (raw_y / 100.0) * img_height
        width = (raw_w / 100.0) * img_width
        height = (raw_h / 100.0) * img_height
    else:
        x, y, width, height = raw_x, raw_y, raw_w, raw_h

    # Clamp to image bounds but retain original center
    x1 = max(0.0, min(float(img_width), x))
    y1 = max(0.0, min(float(img_height), y))
    x2 = max(0.0, min(float(img_width), x + width))
    y2 = max(0.0, min(float(img_height), y + height))

    if x2 <= x1 or y2 <= y1:
        return None

    center_x = min(max(x + width / 2.0, 0.0), float(img_width))
    center_y = min(max(y + height / 2.0, 0.0), float(img_height))

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
        "center_x": center_x,
        "center_y": center_y,
        "is_percentage": is_percentage,
    }


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
    output_path: Path,
    confidence_threshold: str = "low",
    criticality: str = "medium"
) -> Path:
    """
    Draw PRECISE bounding boxes with labels on image.
    Uses corner markers and crosshairs for small defects.
    
    Args:
        image_path: Path to original image
        boxes: List of dicts with x, y, width, height, label, color, confidence (optional)
        output_path: Path to save annotated image
        confidence_threshold: Minimum confidence level to highlight ("low", "medium", "high")
        criticality: Criticality level ("low", "medium", "high") - affects filtering
        
    Returns:
        Path to annotated image
    """
    # Load with OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    img_height, img_width = img.shape[:2]
    
    confidence_levels = {"low": 1, "medium": 2, "high": 3}
    threshold_level = confidence_levels.get(confidence_threshold, 1)
    
    for i, box in enumerate(boxes):
        bbox_pixels = _convert_bbox_to_pixels(
            box.get("bbox", box),
            img_width,
            img_height,
        )
        if not bbox_pixels:
            logger.warning(f"Invalid bbox coordinates (unable to convert): {box}")
            continue
        
        x = int(math.floor(bbox_pixels["x1"]))
        y = int(math.floor(bbox_pixels["y1"]))
        w = int(math.ceil(bbox_pixels["width"]))
        h = int(math.ceil(bbox_pixels["height"]))
        center_x = bbox_pixels["center_x"]
        center_y = bbox_pixels["center_y"]
        
        # Ensure minimum size for marker visibility (not box size)
        min_marker_size = max(20, min(img_width, img_height) // 20)
        
        # Clamp to image bounds
        x = min(max(0, x), img_width - 1)
        y = min(max(0, y), img_height - 1)
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        
        # Skip if box is invalid after clamping
        if w <= 0 or h <= 0:
            logger.warning(f"Bbox invalid after clamping, skipping: {box}")
            continue
        
        # Extract label specific info
        label_full = box.get("label", f"#{i+1}")
        try:
            label_text = label_full.replace("#", "")
        except:
            label_text = str(i+1)
            
        severity = box.get("severity", "MODERATE")
        box_confidence = box.get("confidence", "medium")
        box_level = confidence_levels.get(box_confidence, 2)
        dim_box = box_level < threshold_level and criticality != "high"
        
        # Box color based on severity
        box_color = (0, 0, 255)  # Red for CRITICAL/MODERATE
        if severity == "COSMETIC":
            box_color = (0, 200, 255)  # Yellow-ish for cosmetic
        
        # Line style based on confidence
        thickness = 3 if not dim_box else 1
        line_type = cv2.LINE_AA
        use_dashed = (box_confidence == "low")
        
        # Draw bounding box (dashed for low confidence, solid for medium/high)
        if use_dashed:
            # Draw dashed rectangle using line segments
            dash_length = 10
            gap_length = 5
            # Top edge
            for px in range(x, x + w, dash_length + gap_length):
                end_x = min(px + dash_length, x + w)
                if end_x > px:
                    cv2.line(img, (px, y), (end_x, y), box_color, thickness, line_type)
            # Bottom edge
            for px in range(x, x + w, dash_length + gap_length):
                end_x = min(px + dash_length, x + w)
                if end_x > px:
                    cv2.line(img, (px, y + h), (end_x, y + h), box_color, thickness, line_type)
            # Left edge
            for py in range(y, y + h, dash_length + gap_length):
                end_y = min(py + dash_length, y + h)
                if end_y > py:
                    cv2.line(img, (x, py), (x, end_y), box_color, thickness, line_type)
            # Right edge
            for py in range(y, y + h, dash_length + gap_length):
                end_y = min(py + dash_length, y + h)
                if end_y > py:
                    cv2.line(img, (x + w, py), (x + w, end_y), box_color, thickness, line_type)
        else:
            # Solid rectangle for medium/high confidence
            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, thickness, line_type)
        
        # Draw Circular Marker (#1, #2)
        # Marker size: dynamic based on image size
        marker_radius = int(max(img_width, img_height) * 0.04)
        marker_radius = max(25, min(marker_radius, 60))  # Clamp (min 25px, max 60px)
        
        # Marker centered on bbox for better accuracy
        marker_cx = max(marker_radius + 5, min(int(center_x), img_width - marker_radius - 5))
        marker_cy = max(marker_radius + 5, min(int(center_y), img_height - marker_radius - 5))
             
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
    alpha: float = 0.4,
    actual_model_size: Optional[int] = None,
    confidence_threshold: str = "low",
    criticality: str = "medium"
) -> Path:
    """
    Create semi-transparent heatmap overlay on defect regions.
    
    Args:
        image_path: Path to original image
        defects: List of defect dicts with bbox, severity, and confidence
        output_path: Path to save overlay image
        alpha: Transparency level (0-1)
        actual_model_size: Actual max dimension used by model (from config, default 2048)
        confidence_threshold: Minimum confidence to include ("low", "medium", "high")
        criticality: Criticality level for filtering
        
    Returns:
        Path to overlay image
    """
    logger.info(f"Creating heatmap overlay for {image_path.name}")
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    height, width = img.shape[:2]
    
    # Include ALL defects in heatmap - don't filter by confidence
    # We want to show all detected defects, even if confidence is low
    # The intensity will be adjusted based on confidence/severity
    filtered_defects = defects
    
    # All coordinates are now PERCENTAGES (0-100) - no scaling needed
    # Note: actual_model_size parameter kept for backward compatibility but not used
        
    # Initialize blank mask for heatmap (float 0-1)
    heat_mask = np.zeros((height, width), dtype=np.float32)
    
    has_defects = False
    
    for defect in filtered_defects:
        has_defects = True
        bbox = defect.get("bbox", {})
        severity = defect.get("safety_impact", "MODERATE")
        defect_confidence = defect.get("confidence", "medium")
        
        # Determine heat intensity based on severity and confidence
        # Make ALL defects visible, even small ones - adjust intensity but don't hide them
        severity_weight = {
            'CRITICAL': 1.0,
            'MODERATE': 0.75,  # Increased from 0.6 to make moderate defects more visible
            'COSMETIC': 0.5,   # Increased from 0.3 to make cosmetic defects visible
            'MINOR': 0.5       # Increased from 0.3 to make minor defects visible
        }
        base_intensity = severity_weight.get(severity, 0.6)
        
        # Confidence factor: Still vary by confidence but ensure all defects are visible
        # Reduced the gap between confidence levels so low-confidence defects are still visible
        confidence_factor = {"high": 1.0, "medium": 0.75, "low": 0.55}.get(defect_confidence, 0.65)
        intensity = base_intensity * confidence_factor
        
        # Ensure minimum intensity for visibility - even small/low-confidence defects should be visible
        min_intensity = 0.35  # Minimum intensity to ensure all defects are visible
        intensity = max(intensity, min_intensity)
        
        # Boost intensity for high-confidence critical defects to make them stand out
        if severity == 'CRITICAL' and defect_confidence == "high":
            intensity = min(1.0, intensity * 1.2)  # Boost by 20% but cap at 1.0
        
        # Check for widespread defects (no bbox AND explicit widespread keywords)
        widespread_keywords = ["entire surface", "everywhere", "whole component", "complete surface"]
        location_lower = defect.get("location", "").lower()
        
        has_valid_bbox = (
            bbox
            and bbox.get("x") is not None
            and bbox.get("y") is not None
            and bbox.get("width", 0) > 0
            and bbox.get("height", 0) > 0
        )

        # Only treat as widespread if bbox is explicitly None AND keywords present
        is_widespread = (
            bbox is None
            and any(kw in location_lower for kw in widespread_keywords)
        )
        
        if is_widespread:
            # Add general heat to the entire image with gradient from center
            center_x, center_y = width // 2, height // 2
            radius = max(width, height) // 2
            
            y_coords, x_coords = np.ogrid[:height, :width]
            dist_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            gaussian = intensity * np.exp(-dist_sq / (2 * (radius * 0.7)**2))
            heat_mask = np.maximum(heat_mask, gaussian.astype(np.float32))
            continue
            
        if has_valid_bbox:
            bbox_pixels = _convert_bbox_to_pixels(bbox, width, height)
            if not bbox_pixels:
                logger.warning(f"Invalid bbox in heatmap (conversion failed): {bbox}")
                continue

            x1 = bbox_pixels["x1"]
            y1 = bbox_pixels["y1"]
            x2 = bbox_pixels["x2"]
            y2 = bbox_pixels["y2"]
            box_width = max(bbox_pixels["width"], 1.0)
            box_height = max(bbox_pixels["height"], 1.0)
            center_x = bbox_pixels["center_x"]
            center_y = bbox_pixels["center_y"]

            logger.debug(
                f"Heatmap defect: {defect.get('type')} -> "
                f"pixels ({x1:.1f},{y1:.1f}) to ({x2:.1f},{y2:.1f})"
            )

            # Calculate sigma for Gaussian based on bbox size (per-axis for accuracy)
            sigma_x = max(box_width * 0.6, 12.0)
            sigma_y = max(box_height * 0.6, 12.0)
            max_sigma = min(width, height) * 0.2
            sigma_x = min(sigma_x, max_sigma)
            sigma_y = min(sigma_y, max_sigma)
            sigma = max(sigma_x, sigma_y)

            blur_margin = int(4 * max(sigma_x, sigma_y)) + 15
            mask_x1 = int(max(0, math.floor(center_x - blur_margin)))
            mask_y1 = int(max(0, math.floor(center_y - blur_margin)))
            mask_x2 = int(min(width, math.ceil(center_x + blur_margin + 1)))
            mask_y2 = int(min(height, math.ceil(center_y + blur_margin + 1)))

            if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
                logger.warning(
                    f"Invalid slice region for defect {defect.get('type')}: "
                    f"({mask_x1},{mask_y1}) to ({mask_x2},{mask_y2})"
                )
                continue

            local_y_coords, local_x_coords = np.ogrid[mask_y1:mask_y2, mask_x1:mask_x2]
            dx_raw = local_x_coords - center_x
            dy_raw = local_y_coords - center_y

            gaussian_local = np.exp(
                -((dx_raw ** 2) / (2 * sigma_x ** 2) + (dy_raw ** 2) / (2 * sigma_y ** 2))
            )
            gaussian_local *= intensity
            
            # Apply stronger heat inside the actual bbox region for accuracy
            # Check which pixels are inside the bbox
            in_bbox_x = (local_x_coords >= x1) & (local_x_coords < x2)
            in_bbox_y = (local_y_coords >= y1) & (local_y_coords < y2)
            in_bbox = in_bbox_x & in_bbox_y
            
            # Calculate distance from bbox center for stronger concentration
            # Pixels inside bbox get much stronger intensity, outside fade smoothly
            bbox_radius_x = box_width / 2.0
            bbox_radius_y = box_height / 2.0
            
            # Create elliptical distance metric for bbox shape
            dx_norm = dx_raw / max(bbox_radius_x, 1.0)
            dy_norm = dy_raw / max(bbox_radius_y, 1.0)
            bbox_dist_sq = dx_norm**2 + dy_norm**2
            
            # Strong boost inside bbox (within 1.2x bbox radius)
            in_bbox_strong = bbox_dist_sq < 1.2**2
            
            # Very strong intensity for pixels inside bbox
            boost_factor = np.where(in_bbox_strong, 1.8,  # 80% boost inside bbox
                                  np.where(in_bbox, 1.4,   # 40% boost at bbox edges
                                          1.0))            # No boost outside
            
            gaussian_local = np.minimum(1.0, gaussian_local * boost_factor)
            
            # Apply within 4x sigma radius for smooth falloff and to capture edges
            local_mask = (
                (dx_raw ** 2) / (sigma_x * 4.0) ** 2 + (dy_raw ** 2) / (sigma_y * 4.0) ** 2
            ) <= 1.0
            defect_mask_local = np.where(local_mask, gaussian_local.astype(np.float32), 0)
            
            # Apply additional Gaussian blur for smooth transitions (true Gaussian blur effect)
            # Use adaptive kernel size based on sigma
            blur_kernel_sigma = sigma * 0.4  # Subtle blur for smoothness
            kernel_size = min(int(2 * np.ceil(3 * blur_kernel_sigma) + 1), 51)  # Max 51x51 for speed
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size > 1 and defect_mask_local.size > 0:
                defect_mask_local = cv2.GaussianBlur(defect_mask_local, (kernel_size, kernel_size), blur_kernel_sigma)
            
            # Combine with main heat mask (take maximum to preserve overlapping defects)
            heat_mask[mask_y1:mask_y2, mask_x1:mask_x2] = np.maximum(
                heat_mask[mask_y1:mask_y2, mask_x1:mask_x2],
                defect_mask_local
            )

    if not has_defects:
        # Just save original if no defects
        cv2.imwrite(str(output_path), img)
        return output_path

    # Apply final subtle Gaussian blur to entire heat mask for smoother transitions
    # This creates the true Gaussian blur effect like in the reference image
    # Use a light blur kernel to smooth transitions between hot spots without losing detail
    if heat_mask.max() > 0:
        # Calculate adaptive kernel size - keep it subtle for performance
        blur_sigma = min(width, height) * 0.01  # 1% of image dimension (light blur for speed)
        # Limit kernel size to prevent slow processing (max 31x31 for final blur)
        kernel_size = min(int(2 * np.ceil(3 * blur_sigma) + 1), 31)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size > 1:
            # Apply light blur to smooth transitions
            heat_mask = cv2.GaussianBlur(heat_mask, (kernel_size, kernel_size), blur_sigma)
    
    # Normalize heat mask to 0-255 for colormap
    if heat_mask.max() > 0:
        heat_mask_norm = (heat_mask / heat_mask.max() * 255).astype(np.uint8)
    else:
        heat_mask_norm = (heat_mask * 255).astype(np.uint8)
    
    # Apply JET colormap (Classic thermal look: Blue -> Cyan -> Yellow -> Red)
    heatmap_color = cv2.applyColorMap(heat_mask_norm, cv2.COLORMAP_JET)
    
    # Blend with original (40% heatmap, 60% original - like user's code)
    heatmap_overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
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
