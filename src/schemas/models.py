"""
Pydantic schemas for data validation.
"""

import time
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: float = Field(..., description="X coordinate (0-1 normalized or pixel)")
    y: float = Field(..., description="Y coordinate (0-1 normalized or pixel)")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")
    
    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v


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
        """Combine defects from both models."""
        # Use inspector as primary, add unique defects from auditor
        inspector_types = set(d.type for d in self.inspector_result.defects)
        
        self.combined_defects = self.inspector_result.defects.copy()
        
        for defect in self.auditor_result.defects:
            if defect.type not in inspector_types:
                self.combined_defects.append(defect)
        
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
