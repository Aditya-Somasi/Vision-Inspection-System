"""
Repository for inspection CRUD operations.
"""

from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base, InspectionRecord, DefectRecord
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="DATABASE")

# Create database engine
engine = create_engine(
    f"sqlite:///{config.database_path}",
    echo=config.database_echo,
    connect_args={"check_same_thread": False}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
            Dictionary with statistics including object type distribution
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
            
            # Object type distribution (NEW - for analytics by object type)
            object_counts = session.query(
                InspectionRecord.object_identified,
                func.count(InspectionRecord.id)
            ).filter(
                InspectionRecord.object_identified.isnot(None)
            ).group_by(InspectionRecord.object_identified).all()
            
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
                "object_counts": dict(object_counts),  # NEW: Object type distribution
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


# Note: init_database() should be called explicitly at app startup
# Do NOT auto-initialize on import as it can cause issues with testing and imports
