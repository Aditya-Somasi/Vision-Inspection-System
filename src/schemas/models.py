"""
Pydantic schemas for data validation.
"""

import time
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates in PERCENTAGE format (0-100)."""
    x: float = Field(..., description="X coordinate as percentage (0-100) from left edge")
    y: float = Field(..., description="Y coordinate as percentage (0-100) from top edge")
    width: float = Field(..., description="Width as percentage (0-100) of image width")
    height: float = Field(..., description="Height as percentage (0-100) of image height")
    
    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v
    
    @model_validator(mode="after")
    def validate_percentage_range(self):
        """Validate that all coordinates are in valid percentage range (0-100)."""
        if self.x < 0 or self.x > 100:
            raise ValueError(f"X coordinate must be between 0 and 100, got {self.x}")
        if self.y < 0 or self.y > 100:
            raise ValueError(f"Y coordinate must be between 0 and 100, got {self.y}")
        if self.width <= 0 or self.width > 100:
            raise ValueError(f"Width must be between 0 and 100, got {self.width}")
        if self.height <= 0 or self.height > 100:
            raise ValueError(f"Height must be between 0 and 100, got {self.height}")
        if self.x + self.width > 100:
            raise ValueError(f"Bounding box exceeds image width: x={self.x}, width={self.width} (x+width={self.x+self.width} > 100)")
        if self.y + self.height > 100:
            raise ValueError(f"Bounding box exceeds image height: y={self.y}, height={self.height} (y+height={self.y+self.height} > 100)")
        return self
    
    def is_reasonable(self, min_area_percent: float = 0.1, max_area_percent: float = 50.0) -> bool:
        """
        Check if bounding box is reasonable (not too small or too large).
        
        Args:
            min_area_percent: Minimum area as percentage of image (default 0.1%)
            max_area_percent: Maximum area as percentage of image (default 50%)
        
        Returns:
            True if bbox is reasonable, False otherwise
        """
        area_percent = (self.width * self.height) / 100.0
        return min_area_percent <= area_percent <= max_area_percent


class DefectInfo(BaseModel):
    """Structured defect information."""
    defect_id: str = Field(
        default_factory=lambda: f"defect_{int(time.time()*1000)}"
    )
    type: str = Field(..., description="Defect type (e.g., crack, rust)")
    location: str = Field(..., description="Human-readable location description")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box if available")
    safety_impact: Literal["CRITICAL", "MODERATE", "COSMETIC"] = Field(
        ..., description="Safety impact level"
    )
    reasoning: str = Field(..., description="Why this defect is concerning")
    confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Detection confidence"
    )
    recommended_action: str = Field(..., description="Suggested action to take")
    
    @field_validator("type")
    @classmethod
    def normalize_defect_type(cls, v: str) -> str:
        """Normalize defect type to lowercase."""
        return v.lower().strip()
    
    def is_critical(self) -> bool:
        """Check if defect is critical."""
        return self.safety_impact == "CRITICAL"


class VLMAnalysisResult(BaseModel):
    """VLM analysis result with structured defects."""
    object_identified: str = Field(
        ..., description="What object/component was identified"
    )
    overall_condition: Literal["damaged", "good", "uncertain"] = Field(
        ..., description="Overall condition assessment"
    )
    defects: List[DefectInfo] = Field(
        default_factory=list, description="List of detected defects"
    )
    overall_confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Overall analysis confidence"
    )
    analysis_reasoning: Optional[str] = Field(
        None, description="General reasoning about the image"
    )
    # Agent-inferred criticality
    inferred_criticality: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Agent-inferred criticality level based on object type and defects"
    )
    inferred_criticality_reasoning: Optional[str] = Field(
        None, description="Reasoning for the inferred criticality level"
    )
    # Error handling
    analysis_failed: bool = Field(
        default=False, description="Whether the analysis failed due to an error"
    )
    failure_reason: Optional[str] = Field(
        None, description="Reason for analysis failure if analysis_failed is True"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def has_defects(self) -> bool:
        """Check if any defects were found."""
        return len(self.defects) > 0
    
    @property
    def critical_defect_count(self) -> int:
        """Count critical defects."""
        return sum(1 for d in self.defects if d.is_critical())
    
    @property
    def defect_types(self) -> List[str]:
        """Get list of unique defect types."""
        return list(set(d.type for d in self.defects))


class ConsensusResult(BaseModel):
    """Result of consensus analysis between two VLMs."""
    models_agree: bool = Field(..., description="Whether models agree on findings")
    inspector_result: VLMAnalysisResult
    auditor_result: VLMAnalysisResult
    agreement_score: float = Field(..., ge=0, le=1, description="Agreement score 0-1")
    disagreement_details: Optional[str] = Field(
        None, description="Details of disagreements"
    )
    combined_defects: List[DefectInfo] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def compute_combined_defects(self):
        """Combine defects from both models, preserving location differences."""
        def _defects_semantically_similar(defect1: DefectInfo, defect2: DefectInfo) -> bool:
            """Check if two defects are semantically similar (same type, potentially same meaning)."""
            type1 = defect1.type.lower().strip()
            type2 = defect2.type.lower().strip()
            
            # Exact match
            if type1 == type2:
                return True
            
            # Common semantic variations
            semantic_groups = [
                {"crack", "hairline_crack", "fracture", "fissure"},
                {"rust", "corrosion", "oxidation"},
                {"scratch", "scrape", "abrasion"},
                {"dent", "deformation", "dent"},
                {"discoloration", "stain", "discoloration"}
            ]
            
            for group in semantic_groups:
                if type1 in group and type2 in group:
                    return True
            
            return False
        
        def _bboxes_overlap(bbox1: Optional[BoundingBox], bbox2: Optional[BoundingBox], threshold: float = 0.5) -> bool:
            """Check if two bounding boxes overlap significantly."""
            if bbox1 is None or bbox2 is None:
                return False
            
            # Calculate IoU (Intersection over Union)
            x1_min, y1_min = bbox1.x, bbox1.y
            x1_max, y1_max = bbox1.x + bbox1.width, bbox1.y + bbox1.height
            
            x2_min, y2_min = bbox2.x, bbox2.y
            x2_max, y2_max = bbox2.x + bbox2.width, bbox2.y + bbox2.height
            
            # Intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return False
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = bbox1.width * bbox1.height
            area2 = bbox2.width * bbox2.height
            union_area = area1 + area2 - inter_area
            
            if union_area == 0:
                return False
            
            iou = inter_area / union_area
            return iou >= threshold
        
        # Start with inspector defects
        self.combined_defects = []
        inspector_defects = self.inspector_result.defects.copy()
        auditor_defects = self.auditor_result.defects.copy()
        
        # Track which auditor defects have been matched
        auditor_matched = [False] * len(auditor_defects)
        
        # For each inspector defect, try to match with auditor defects
        for inspector_defect in inspector_defects:
            matched = False
            
            for i, auditor_defect in enumerate(auditor_defects):
                if auditor_matched[i]:
                    continue
                
                # Check if semantically similar
                if _defects_semantically_similar(inspector_defect, auditor_defect):
                    # If same type AND overlapping bbox -> merge (use inspector as primary)
                    if _bboxes_overlap(inspector_defect.bbox, auditor_defect.bbox):
                        # Merge: keep inspector defect, mark auditor as matched
                        self.combined_defects.append(inspector_defect)
                        auditor_matched[i] = True
                        matched = True
                        break
                    # If same type but different location -> keep both
                    # (This will be handled by not matching)
            
            # If no match found, add inspector defect as-is
            if not matched:
                self.combined_defects.append(inspector_defect)
        
        # Add unmatched auditor defects
        for i, auditor_defect in enumerate(auditor_defects):
            if not auditor_matched[i]:
                self.combined_defects.append(auditor_defect)
        
        return self


class SafetyVerdict(BaseModel):
    """Final safety verdict after all checks."""
    verdict: Literal["SAFE", "UNSAFE", "REQUIRES_HUMAN_REVIEW"] = Field(
        ..., description="Final safety verdict"
    )
    reason: str = Field(..., description="Reason for verdict")
    requires_human: bool = Field(..., description="Whether human review is required")
    confidence_level: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in verdict"
    )
    triggered_gates: List[str] = Field(
        default_factory=list, description="Which safety gates were triggered"
    )
    defect_summary: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(
        default_factory=list, description="List of errors encountered during inspection"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InspectionContext(BaseModel):
    """Context information for inspection."""
    image_id: str
    criticality: Literal["low", "medium", "high"] = "medium"
    domain: Optional[str] = None
    reference_standards: Optional[List[str]] = None
    user_notes: Optional[str] = None


__all__ = [
    "BoundingBox",
    "DefectInfo",
    "VLMAnalysisResult",
    "ConsensusResult",
    "SafetyVerdict",
    "InspectionContext",
]
