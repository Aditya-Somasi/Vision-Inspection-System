"""
State definition for inspection workflow.
"""

from typing import TypedDict, Optional, Dict, Any


class InspectionState(TypedDict):
    """State for inspection workflow."""
    
    # Input
    image_path: str
    context: Dict[str, Any]  # InspectionContext as dict
    
    # Request tracking
    request_id: str
    start_time: float
    
    # VLM results
    inspector_result: Optional[Dict[str, Any]]  # VLMAnalysisResult as dict
    auditor_result: Optional[Dict[str, Any]]  # VLMAnalysisResult as dict
    
    # Consensus and safety
    consensus: Optional[Dict[str, Any]]  # ConsensusResult as dict
    safety_verdict: Optional[Dict[str, Any]]  # SafetyVerdict as dict
    
    # Human review
    requires_human_review: bool
    human_decision: Optional[str]  # "approve", "reject", "modify"
    human_notes: Optional[str]
    
    # Explanation and report
    explanation: Optional[str]
    report_path: Optional[str]
    
    # Metadata
    processing_time: Optional[float]
    error: Optional[str]
    current_step: str
