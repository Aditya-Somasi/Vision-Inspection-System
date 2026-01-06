"""
SQLAlchemy ORM models for inspection records.
"""

from datetime import datetime
from typing import Dict, Any

from sqlalchemy import (
    Column, Integer, String, Float, DateTime,
    Text, Boolean, ForeignKey, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class InspectionRecord(Base):
    """Main inspection record."""
    __tablename__ = "inspections"
    
    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(String, unique=True, index=True, nullable=False)
    image_path = Column(String, nullable=False)
    image_filename = Column(String, nullable=False)
    image_size_kb = Column(Float)
    image_format = Column(String)
    
    # Context
    criticality = Column(String, nullable=False)  # low, medium, high
    domain = Column(String)
    user_notes = Column(Text)
    
    # Results
    overall_verdict = Column(String, nullable=False)  # SAFE, UNSAFE, REQUIRES_REVIEW
    defect_count = Column(Integer, default=0)
    critical_defect_count = Column(Integer, default=0)
    
    # Model outputs
    inspector_confidence = Column(String)  # high, medium, low
    auditor_confidence = Column(String)
    models_agree = Column(Boolean, default=False)
    agreement_score = Column(Float)
    
    # Safety gates
    triggered_gates = Column(JSON)  # List of triggered gate names
    requires_human = Column(Boolean, default=False)
    
    # Processing
    processing_time_seconds = Column(Float)
    report_path = Column(String)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    defects = relationship(
        "DefectRecord",
        back_populates="inspection",
        cascade="all, delete-orphan"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "image_filename": self.image_filename,
            "criticality": self.criticality,
            "domain": self.domain,
            "overall_verdict": self.overall_verdict,
            "defect_count": self.defect_count,
            "critical_defect_count": self.critical_defect_count,
            "inspector_confidence": self.inspector_confidence,
            "auditor_confidence": self.auditor_confidence,
            "models_agree": self.models_agree,
            "agreement_score": self.agreement_score,
            "requires_human": self.requires_human,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DefectRecord(Base):
    """Individual defect record."""
    __tablename__ = "defects"
    
    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(
        String,
        ForeignKey("inspections.inspection_id"),
        nullable=False
    )
    
    defect_id = Column(String, nullable=False)
    defect_type = Column(String, nullable=False, index=True)
    location = Column(String)
    
    # Bounding box
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_width = Column(Float)
    bbox_height = Column(Float)
    
    safety_impact = Column(String, nullable=False)  # CRITICAL, MODERATE, COSMETIC
    reasoning = Column(Text)
    confidence = Column(String)  # high, medium, low
    recommended_action = Column(Text)
    
    detected_by = Column(String)  # inspector, auditor, both
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    inspection = relationship("InspectionRecord", back_populates="defects")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type,
            "location": self.location,
            "bbox": {
                "x": self.bbox_x,
                "y": self.bbox_y,
                "width": self.bbox_width,
                "height": self.bbox_height
            } if self.bbox_x is not None else None,
            "safety_impact": self.safety_impact,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
            "detected_by": self.detected_by
        }
