"""
CV-Based Defect Region Validator.

Uses OpenCV to validate that LLM-reported defects have actual visual evidence
in the image. This prevents false positives by checking for:
- Edge density (cracks, fractures)
- Texture anomalies (corrosion, wear)
- Local contrast deviations (surface defects)
- Color deviations (rust, contamination)
- Gradient discontinuities (structural damage)

Each defect gets a visual_evidence_score (0-1):
- < 0.3: Reject as false positive
- 0.3-0.5: Downgrade confidence to "low"
- >= 0.5: Accept as validated defect
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="CV_VALIDATOR")


@dataclass
class ValidationResult:
    """Result of validating a single defect region."""
    is_valid: bool
    visual_evidence_score: float
    edge_score: float
    texture_score: float
    contrast_score: float
    color_score: float
    gradient_score: float
    rejection_reason: Optional[str] = None
    adjusted_confidence: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "visual_evidence_score": round(self.visual_evidence_score, 3),
            "edge_score": round(self.edge_score, 3),
            "texture_score": round(self.texture_score, 3),
            "contrast_score": round(self.contrast_score, 3),
            "color_score": round(self.color_score, 3),
            "gradient_score": round(self.gradient_score, 3),
            "rejection_reason": self.rejection_reason,
            "adjusted_confidence": self.adjusted_confidence
        }


class DefectRegionValidator:
    """
    Validates defect regions using computer vision techniques.
    
    This validator checks if an LLM-reported defect region actually contains
    visual evidence of a defect, helping filter out false positives.
    """
    
    # Validation thresholds
    REJECT_THRESHOLD = 0.3       # Below this = false positive
    DOWNGRADE_THRESHOLD = 0.5   # Below this = downgrade confidence
    
    # Signal weights for combined score
    WEIGHTS = {
        "edge": 0.30,
        "texture": 0.25,
        "contrast": 0.20,
        "color": 0.15,
        "gradient": 0.10
    }
    
    def __init__(self):
        self.logger = logger
    
    def validate_defect_region(
        self,
        image: np.ndarray,
        bbox: Dict[str, float],
        defect_type: str = "unknown",
        context_margin: float = 0.5
    ) -> ValidationResult:
        """
        Validate a defect region for visual evidence.
        
        Args:
            image: OpenCV image (BGR format)
            bbox: Bounding box dict with x, y, width, height (percentages 0-100)
            defect_type: Type of defect (used for type-specific validation)
            context_margin: How much surrounding area to include for comparison (0.5 = 50% extra)
        
        Returns:
            ValidationResult with scores and validity
        """
        height, width = image.shape[:2]
        
        # Convert percentage bbox to pixels
        x = int((bbox.get("x", 0) / 100.0) * width)
        y = int((bbox.get("y", 0) / 100.0) * height)
        w = int((bbox.get("width", 10) / 100.0) * width)
        h = int((bbox.get("height", 10) / 100.0) * height)
        
        # Ensure valid bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        # Extract defect region
        defect_region = image[y:y+h, x:x+w]
        
        if defect_region.size == 0:
            self.logger.warning(f"Empty defect region at bbox: {bbox}")
            return ValidationResult(
                is_valid=False,
                visual_evidence_score=0.0,
                edge_score=0.0,
                texture_score=0.0,
                contrast_score=0.0,
                color_score=0.0,
                gradient_score=0.0,
                rejection_reason="Empty region"
            )
        
        # Extract context region (surrounding area for comparison)
        margin_x = int(w * context_margin)
        margin_y = int(h * context_margin)
        
        ctx_x1 = max(0, x - margin_x)
        ctx_y1 = max(0, y - margin_y)
        ctx_x2 = min(width, x + w + margin_x)
        ctx_y2 = min(height, y + h + margin_y)
        
        context_region = image[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
        
        # Convert to grayscale for analysis
        defect_gray = cv2.cvtColor(defect_region, cv2.COLOR_BGR2GRAY) if len(defect_region.shape) == 3 else defect_region
        context_gray = cv2.cvtColor(context_region, cv2.COLOR_BGR2GRAY) if len(context_region.shape) == 3 else context_region
        
        # Calculate validation signals
        edge_score = self._calculate_edge_score(defect_gray, context_gray, defect_type)
        texture_score = self._calculate_texture_score(defect_gray, context_gray)
        contrast_score = self._calculate_contrast_score(defect_gray, context_gray)
        color_score = self._calculate_color_score(defect_region, context_region)
        gradient_score = self._calculate_gradient_score(defect_gray, context_gray)
        
        # Weighted combined score
        visual_evidence_score = (
            self.WEIGHTS["edge"] * edge_score +
            self.WEIGHTS["texture"] * texture_score +
            self.WEIGHTS["contrast"] * contrast_score +
            self.WEIGHTS["color"] * color_score +
            self.WEIGHTS["gradient"] * gradient_score
        )
        
        # Apply defect-type-specific boosting
        visual_evidence_score = self._apply_type_specific_boost(
            visual_evidence_score, defect_type, edge_score, texture_score, color_score
        )
        
        # Determine validity and confidence adjustment
        is_valid = visual_evidence_score >= self.REJECT_THRESHOLD
        rejection_reason = None
        adjusted_confidence = None
        
        if not is_valid:
            rejection_reason = f"Insufficient visual evidence (score: {visual_evidence_score:.2f} < {self.REJECT_THRESHOLD})"
            self.logger.info(f"Rejecting defect '{defect_type}': {rejection_reason}")
        elif visual_evidence_score < self.DOWNGRADE_THRESHOLD:
            adjusted_confidence = "low"
            self.logger.info(f"Downgrading defect '{defect_type}' confidence to 'low' (score: {visual_evidence_score:.2f})")
        
        return ValidationResult(
            is_valid=is_valid,
            visual_evidence_score=visual_evidence_score,
            edge_score=edge_score,
            texture_score=texture_score,
            contrast_score=contrast_score,
            color_score=color_score,
            gradient_score=gradient_score,
            rejection_reason=rejection_reason,
            adjusted_confidence=adjusted_confidence
        )
    
    def _calculate_edge_score(
        self, 
        defect_gray: np.ndarray, 
        context_gray: np.ndarray,
        defect_type: str
    ) -> float:
        """
        Calculate edge density score using Canny edge detection.
        
        Defects like cracks, fractures, and tears have strong edges.
        Score is based on edge density relative to context.
        """
        # Apply Gaussian blur to reduce noise
        defect_blur = cv2.GaussianBlur(defect_gray, (5, 5), 0)
        context_blur = cv2.GaussianBlur(context_gray, (5, 5), 0)
        
        # Adaptive thresholds based on image statistics
        defect_median = np.median(defect_blur)
        lower = int(max(0, 0.5 * defect_median))
        upper = int(min(255, 1.5 * defect_median))
        
        # Edge detection
        defect_edges = cv2.Canny(defect_blur, lower, upper)
        context_edges = cv2.Canny(context_blur, lower, upper)
        
        # Calculate edge density (percentage of edge pixels)
        defect_edge_density = np.count_nonzero(defect_edges) / max(1, defect_edges.size)
        context_edge_density = np.count_nonzero(context_edges) / max(1, context_edges.size)
        
        # Score based on how much defect region differs from context
        # Higher edge density in defect region = higher score
        if context_edge_density > 0:
            relative_density = defect_edge_density / context_edge_density
        else:
            relative_density = defect_edge_density * 10  # Boost if context has no edges
        
        # Normalize to 0-1 range
        # - ratio of 1.0 = same as context = score ~0.3
        # - ratio of 2.0+ = significantly more edges = score ~0.7+
        # - ratio of 0.5 = fewer edges = score ~0.2
        score = min(1.0, relative_density / 2.0)
        
        # Boost for crack/fracture type defects (these MUST have edges)
        edge_critical_types = ["crack", "fracture", "tear", "split", "break", "fissure"]
        if any(t in defect_type.lower() for t in edge_critical_types):
            # For edge-critical types, require higher edge density
            if defect_edge_density < 0.02:  # Very few edges = likely false positive
                score *= 0.3
            elif relative_density > 1.5:
                score = min(1.0, score * 1.3)  # Boost
        
        return score
    
    def _calculate_texture_score(
        self, 
        defect_gray: np.ndarray, 
        context_gray: np.ndarray
    ) -> float:
        """
        Calculate texture anomaly score using Laplacian variance.
        
        Defects often have different texture patterns than their surroundings.
        """
        # Laplacian variance measures texture complexity
        defect_laplacian = cv2.Laplacian(defect_gray, cv2.CV_64F)
        context_laplacian = cv2.Laplacian(context_gray, cv2.CV_64F)
        
        defect_var = defect_laplacian.var()
        context_var = context_laplacian.var()
        
        # Calculate texture deviation from context
        if context_var > 0:
            texture_ratio = abs(defect_var - context_var) / context_var
        else:
            texture_ratio = defect_var / 100.0
        
        # Normalize: ratio of 0 = identical texture, ratio of 1+ = very different
        score = min(1.0, texture_ratio)
        
        return score
    
    def _calculate_contrast_score(
        self, 
        defect_gray: np.ndarray, 
        context_gray: np.ndarray
    ) -> float:
        """
        Calculate local contrast deviation score.
        
        Defects often have different brightness/contrast than surroundings.
        """
        defect_mean = np.mean(defect_gray)
        defect_std = np.std(defect_gray)
        context_mean = np.mean(context_gray)
        context_std = np.std(context_gray)
        
        # Mean deviation (brightness difference)
        if context_mean > 0:
            mean_deviation = abs(defect_mean - context_mean) / context_mean
        else:
            mean_deviation = defect_mean / 128.0
        
        # Std deviation (contrast difference)
        if context_std > 0:
            std_deviation = abs(defect_std - context_std) / context_std
        else:
            std_deviation = defect_std / 50.0
        
        # Combined score (weighted toward mean deviation)
        score = 0.6 * min(1.0, mean_deviation) + 0.4 * min(1.0, std_deviation)
        
        return score
    
    def _calculate_color_score(
        self, 
        defect_region: np.ndarray, 
        context_region: np.ndarray
    ) -> float:
        """
        Calculate color deviation score.
        
        Detects rust, discoloration, contamination, stains.
        """
        if len(defect_region.shape) != 3 or len(context_region.shape) != 3:
            return 0.5  # Neutral if grayscale
        
        # Convert to HSV for better color analysis
        defect_hsv = cv2.cvtColor(defect_region, cv2.COLOR_BGR2HSV)
        context_hsv = cv2.cvtColor(context_region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean HSV values
        defect_h, defect_s, defect_v = [np.mean(defect_hsv[:,:,i]) for i in range(3)]
        context_h, context_s, context_v = [np.mean(context_hsv[:,:,i]) for i in range(3)]
        
        # Hue difference (circular, max diff is 90 on 0-180 scale)
        hue_diff = min(abs(defect_h - context_h), 180 - abs(defect_h - context_h))
        hue_score = hue_diff / 45.0  # Normalize: 45 degree diff = score 1.0
        
        # Saturation difference
        sat_diff = abs(defect_s - context_s)
        sat_score = sat_diff / 128.0  # Normalize
        
        # Value difference
        val_diff = abs(defect_v - context_v)
        val_score = val_diff / 128.0
        
        # Combined (hue is most important for color anomalies)
        score = 0.5 * min(1.0, hue_score) + 0.3 * min(1.0, sat_score) + 0.2 * min(1.0, val_score)
        
        return score
    
    def _calculate_gradient_score(
        self, 
        defect_gray: np.ndarray, 
        context_gray: np.ndarray
    ) -> float:
        """
        Calculate gradient discontinuity score using Sobel operators.
        
        Structural damage shows up as gradient anomalies.
        """
        # Sobel gradients
        defect_gx = cv2.Sobel(defect_gray, cv2.CV_64F, 1, 0, ksize=3)
        defect_gy = cv2.Sobel(defect_gray, cv2.CV_64F, 0, 1, ksize=3)
        defect_magnitude = np.sqrt(defect_gx**2 + defect_gy**2)
        
        context_gx = cv2.Sobel(context_gray, cv2.CV_64F, 1, 0, ksize=3)
        context_gy = cv2.Sobel(context_gray, cv2.CV_64F, 0, 1, ksize=3)
        context_magnitude = np.sqrt(context_gx**2 + context_gy**2)
        
        # Mean gradient magnitude
        defect_grad_mean = np.mean(defect_magnitude)
        context_grad_mean = np.mean(context_magnitude)
        
        # Gradient deviation from context
        if context_grad_mean > 0:
            grad_ratio = abs(defect_grad_mean - context_grad_mean) / context_grad_mean
        else:
            grad_ratio = defect_grad_mean / 50.0
        
        score = min(1.0, grad_ratio)
        
        return score
    
    def _apply_type_specific_boost(
        self,
        base_score: float,
        defect_type: str,
        edge_score: float,
        texture_score: float,
        color_score: float
    ) -> float:
        """
        Apply defect-type-specific boosting or penalty.
        
        Different defect types have different visual signatures.
        """
        defect_lower = defect_type.lower()
        
        # Edge-critical defects (cracks, fractures)
        edge_types = ["crack", "fracture", "tear", "split", "break", "fissure", "hairline"]
        if any(t in defect_lower for t in edge_types):
            if edge_score < 0.3:
                # Cracks MUST have edges - penalize if missing
                return base_score * 0.5
            elif edge_score > 0.6:
                return min(1.0, base_score * 1.2)
        
        # Texture-critical defects (corrosion, wear)
        texture_types = ["corrosion", "rust", "wear", "erosion", "oxidation", "pitting"]
        if any(t in defect_lower for t in texture_types):
            if texture_score > 0.5:
                return min(1.0, base_score * 1.2)
            elif texture_score < 0.2:
                return base_score * 0.7
        
        # Color-critical defects (rust, stains, discoloration)
        color_types = ["rust", "stain", "discoloration", "contamination", "burn"]
        if any(t in defect_lower for t in color_types):
            if color_score > 0.5:
                return min(1.0, base_score * 1.2)
            elif color_score < 0.2:
                return base_score * 0.7
        
        return base_score
    
    def validate_all_defects(
        self,
        image_path: Path,
        defects: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate all defects in a list and return validated/rejected lists.
        
        Args:
            image_path: Path to the image file
            defects: List of defect dicts with bbox and type fields
        
        Returns:
            Tuple of (validated_defects, rejected_defects)
        """
        self.logger.info(f"Validating {len(defects)} defects for visual evidence")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Failed to load image: {image_path}")
            return defects, []  # Return all defects if image load fails
        
        validated = []
        rejected = []
        
        for defect in defects:
            bbox = defect.get("bbox")
            if not bbox:
                # No bbox - can't validate visually, keep with warning
                defect["cv_validation"] = {"skipped": True, "reason": "No bounding box"}
                validated.append(defect)
                continue
            
            defect_type = defect.get("type", "unknown")
            
            result = self.validate_defect_region(
                image, 
                bbox, 
                defect_type
            )
            
            # Add validation result to defect
            defect["cv_validation"] = result.to_dict()
            
            if result.is_valid:
                # Adjust confidence if needed
                if result.adjusted_confidence:
                    defect["confidence"] = result.adjusted_confidence
                    defect["confidence_reason"] = f"Downgraded due to low visual evidence (score: {result.visual_evidence_score:.2f})"
                validated.append(defect)
            else:
                rejected.append(defect)
        
        self.logger.info(
            f"Validation complete: {len(validated)} validated, {len(rejected)} rejected as false positives"
        )
        
        return validated, rejected


def validate_defects(
    image_path: Path,
    defects: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convenience function to validate defects.
    
    Args:
        image_path: Path to the image
        defects: List of defect dicts
    
    Returns:
        Tuple of (validated_defects, rejected_defects)
    """
    validator = DefectRegionValidator()
    return validator.validate_all_defects(image_path, defects)
