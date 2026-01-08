"""
Image quality assessment for safety-critical inspections.
Assesses sharpness, brightness, resolution to detect poor quality images.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import cv2
import numpy as np
from PIL import Image

from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="IMAGE_QUALITY")


class ImageQualityAssessment:
    """Assess image quality metrics."""
    
    def __init__(self):
        self.logger = logger
        # Quality thresholds (configurable)
        self.min_sharpness = 100.0  # Laplacian variance threshold
        self.min_brightness = 30.0  # Mean pixel value (0-255)
        self.max_brightness = 220.0  # Avoid overexposed
        self.min_resolution = 100  # Minimum width or height in pixels
        self.min_pixels = 10000  # Minimum total pixels
    
    def assess_quality(self, image_path: Path) -> Dict[str, Any]:
        """
        Assess image quality.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with quality metrics and overall score
        """
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                return self._quality_failed(f"Failed to load image: {image_path}")
            
            height, width = img.shape[:2]
            total_pixels = width * height
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness assessment (Laplacian variance)
            sharpness_score = self._assess_sharpness(gray)
            
            # 2. Brightness assessment (mean pixel value)
            brightness_score, mean_brightness = self._assess_brightness(gray)
            
            # 3. Resolution assessment
            resolution_score = self._assess_resolution(width, height, total_pixels)
            
            # Overall quality score (weighted average)
            overall_score = (
                0.4 * sharpness_score +
                0.3 * brightness_score +
                0.3 * resolution_score
            )
            
            quality_passed = overall_score >= 0.6  # Threshold for acceptable quality
            
            result = {
                "quality_score": round(overall_score, 3),
                "quality_passed": quality_passed,
                "sharpness": {
                    "score": round(sharpness_score, 3),
                    "laplacian_variance": self._compute_laplacian_variance(gray),
                    "passed": sharpness_score >= 0.6
                },
                "brightness": {
                    "score": round(brightness_score, 3),
                    "mean_value": round(mean_brightness, 1),
                    "passed": brightness_score >= 0.6
                },
                "resolution": {
                    "score": round(resolution_score, 3),
                    "width": width,
                    "height": height,
                    "total_pixels": total_pixels,
                    "passed": resolution_score >= 0.6
                },
                "image_path": str(image_path)
            }
            
            self.logger.info(
                f"Image quality assessment: score={overall_score:.2f}, "
                f"sharpness={sharpness_score:.2f}, brightness={brightness_score:.2f}, "
                f"resolution={resolution_score:.2f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image quality assessment failed: {e}", exc_info=True)
            return self._quality_failed(f"Assessment error: {str(e)}")
    
    def _assess_sharpness(self, gray_image: np.ndarray) -> float:
        """
        Assess image sharpness using Laplacian variance.
        Higher variance = sharper image.
        """
        laplacian_var = self._compute_laplacian_variance(gray_image)
        
        # Normalize to 0-1 scale
        # Typical values: <100 = blurry, 100-500 = acceptable, >500 = sharp
        if laplacian_var < self.min_sharpness:
            score = laplacian_var / self.min_sharpness * 0.5  # 0-0.5 range for blurry
        else:
            score = min(1.0, 0.5 + (laplacian_var - self.min_sharpness) / 400.0)
        
        return score
    
    def _compute_laplacian_variance(self, gray_image: np.ndarray) -> float:
        """Compute Laplacian variance for sharpness."""
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(laplacian.var())
    
    def _assess_brightness(self, gray_image: np.ndarray) -> Tuple[float, float]:
        """
        Assess image brightness.
        Returns (score, mean_brightness).
        """
        mean_brightness = float(np.mean(gray_image))
        
        # Score based on distance from ideal range (100-150 is ideal)
        if self.min_brightness <= mean_brightness <= self.max_brightness:
            # Within acceptable range
            ideal_center = (self.min_brightness + self.max_brightness) / 2
            distance_from_ideal = abs(mean_brightness - ideal_center)
            max_distance = (self.max_brightness - self.min_brightness) / 2
            score = 1.0 - (distance_from_ideal / max_distance) * 0.3  # Small penalty
        elif mean_brightness < self.min_brightness:
            # Too dark
            score = max(0.0, mean_brightness / self.min_brightness * 0.6)
        else:
            # Too bright (overexposed)
            excess = mean_brightness - self.max_brightness
            max_excess = 255 - self.max_brightness
            score = max(0.0, 1.0 - (excess / max_excess) * 0.8)
        
        return score, mean_brightness
    
    def _assess_resolution(self, width: int, height: int, total_pixels: int) -> float:
        """
        Assess image resolution.
        """
        # Check minimum dimensions
        min_dim = min(width, height)
        if min_dim < self.min_resolution:
            return 0.3  # Very low score for tiny images
        
        # Check total pixels
        if total_pixels < self.min_pixels:
            return 0.5  # Low score
        
        # Score based on resolution (higher is better, capped at 1.0)
        # Typical: 640x480 = 307K pixels (0.7), 1920x1080 = 2M pixels (1.0)
        score = min(1.0, total_pixels / 2000000.0)
        
        return score
    
    def _quality_failed(self, reason: str) -> Dict[str, Any]:
        """Return failed quality assessment result."""
        return {
            "quality_score": 0.0,
            "quality_passed": False,
            "sharpness": {"score": 0.0, "passed": False},
            "brightness": {"score": 0.0, "passed": False},
            "resolution": {"score": 0.0, "passed": False},
            "error": reason
        }


def assess_image_quality(image_path: Path) -> Dict[str, Any]:
    """Assess image quality."""
    assessor = ImageQualityAssessment()
    return assessor.assess_quality(image_path)
