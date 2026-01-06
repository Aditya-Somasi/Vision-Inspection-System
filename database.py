"""
Database models and repository for inspection records.
Uses SQLAlchemy ORM for structured data persistence and analytics.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    Text, Boolean, ForeignKey, JSON, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

from logger import setup_logger
from config import config

logger = setup_logger(__name__, level=config.log_level, component="DATABASE")

# Create database engine
engine = create_engine(
    f"sqlite:///{config.database_path}",
    echo=config.database_echo,
    connect_args={"check_same_thread": False}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============================================================================
# DATABASE MODELS
# ============================================================================

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
    defects = relationship("DefectRecord", back_populates="inspection", cascade="all, delete-orphan")
    
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
    inspection_id = Column(String, ForeignKey("inspections.inspection_id"), nullable=False)
    
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


# ============================================================================
# REPOSITORY
# ============================================================================

class InspectionRepository:
    """Repository for inspection CRUD operations."""
    
    def __init__(self):
        self.logger = logger
    
    def get_session(self) -> Session:
        """Get database session."""
        return SessionLocal()
    
    def create_inspection(
        self,
        inspection_data: Dict[str, Any],
        defects_data: List[Dict[str, Any]]
    ) -> InspectionRecord:
        """
        Create a new inspection record with defects.
        
        Args:
            inspection_data: Inspection metadata
            defects_data: List of defect data
        
        Returns:
            Created inspection record
        """
        session = self.get_session()
        
        try:
            # Create inspection
            inspection = InspectionRecord(**inspection_data)
            session.add(inspection)
            session.flush()  # Get inspection_id
            
            # Create defects
            for defect_data in defects_data:
                defect = DefectRecord(
                    inspection_id=inspection.inspection_id,
                    **defect_data
                )
                session.add(defect)
            
            session.commit()
            session.refresh(inspection)
            
            self.logger.info(f"Created inspection record: {inspection.inspection_id}")
            
            return inspection
        
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to create inspection: {e}")
            raise
        
        finally:
            session.close()
    
    def get_inspection(self, inspection_id: str) -> Optional[InspectionRecord]:
        """Get inspection by ID."""
        session = self.get_session()
        
        try:
            inspection = session.query(InspectionRecord).filter(
                InspectionRecord.inspection_id == inspection_id
            ).first()
            
            return inspection
        
        finally:
            session.close()
    
    def list_inspections(
        self,
        limit: int = 50,
        offset: int = 0,
        verdict: Optional[str] = None,
        criticality: Optional[str] = None
    ) -> List[InspectionRecord]:
        """
        List inspections with optional filters.
        
        Args:
            limit: Maximum number of records
            offset: Offset for pagination
            verdict: Filter by verdict
            criticality: Filter by criticality
        
        Returns:
            List of inspection records
        """
        session = self.get_session()
        
        try:
            query = session.query(InspectionRecord)
            
            if verdict:
                query = query.filter(InspectionRecord.overall_verdict == verdict)
            
            if criticality:
                query = query.filter(InspectionRecord.criticality == criticality)
            
            inspections = query.order_by(
                InspectionRecord.created_at.desc()
            ).limit(limit).offset(offset).all()
            
            return inspections
        
        finally:
            session.close()
    
    def get_inspection_count(self) -> int:
        """Get total inspection count."""
        session = self.get_session()
        
        try:
            count = session.query(InspectionRecord).count()
            return count
        
        finally:
            session.close()
    
    def get_defect_statistics(self) -> Dict[str, Any]:
        """
        Get defect statistics for analytics.
        
        Returns:
            Dictionary with statistics
        """
        session = self.get_session()
        
        try:
            # Total defects by type
            defect_counts = session.query(
                DefectRecord.defect_type,
                func.count(DefectRecord.id)
            ).group_by(DefectRecord.defect_type).all()
            
            # Defects by severity
            severity_counts = session.query(
                DefectRecord.safety_impact,
                func.count(DefectRecord.id)
            ).group_by(DefectRecord.safety_impact).all()
            
            # Verdict distribution
            verdict_counts = session.query(
                InspectionRecord.overall_verdict,
                func.count(InspectionRecord.id)
            ).group_by(InspectionRecord.overall_verdict).all()
            
            # Agreement rate
            total_inspections = session.query(InspectionRecord).count()
            agreed_inspections = session.query(InspectionRecord).filter(
                InspectionRecord.models_agree == True
            ).count()
            
            agreement_rate = (
                agreed_inspections / total_inspections if total_inspections > 0 else 0
            )
            
            # Average processing time
            avg_time = session.query(
                func.avg(InspectionRecord.processing_time_seconds)
            ).scalar() or 0
            
            return {
                "defect_counts": dict(defect_counts),
                "severity_counts": dict(severity_counts),
                "verdict_counts": dict(verdict_counts),
                "agreement_rate": agreement_rate,
                "total_inspections": total_inspections,
                "avg_processing_time": avg_time
            }
        
        finally:
            session.close()
    
    def delete_inspection(self, inspection_id: str):
        """Delete an inspection and its defects."""
        session = self.get_session()
        
        try:
            inspection = session.query(InspectionRecord).filter(
                InspectionRecord.inspection_id == inspection_id
            ).first()
            
            if inspection:
                session.delete(inspection)
                session.commit()
                self.logger.info(f"Deleted inspection: {inspection_id}")
        
        finally:
            session.close()


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_database():
    """Initialize database schema."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def health_check_database() -> bool:
    """Check database health."""
    try:
        session = SessionLocal()
        # Simple query to test connection
        session.query(InspectionRecord).first()
        session.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# Initialize on import
init_database()