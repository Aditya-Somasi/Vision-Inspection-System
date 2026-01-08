"""
State definition for inspection workflow.
"""

from __future__ import annotations

from typing import TypedDict, Optional, Dict, Any, List, Union, Tuple


def validate_state(state: InspectionState, required_fields: Optional[List[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate inspection state before critical operations.
    
    Args:
        state: Inspection state to validate
        required_fields: Optional list of required field names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(state, dict):
        return False, "State must be a dictionary"
    
    # Default required fields for critical operations
    if required_fields is None:
        required_fields = [
            "image_path",
            "context",
            "request_id",
            "current_step"
        ]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in state or state[field] is None]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    # Validate context structure
    if "context" in state:
        context = state["context"]
        if not isinstance(context, dict):
            return False, "Context must be a dictionary"
        
        required_context_fields = ["criticality"]
        missing_context = [f for f in required_context_fields if f not in context]
        if missing_context:
            return False, f"Missing required context fields: {', '.join(missing_context)}"
        
        # Validate criticality value
        if "criticality" in context:
            criticality = context["criticality"]
            if criticality not in ["low", "medium", "high"]:
                return False, f"Invalid criticality value: {criticality} (must be 'low', 'medium', or 'high')"
    
    # Validate image_path
    if "image_path" in state:
        image_path = state["image_path"]
        if not isinstance(image_path, (str, list)):
            return False, "image_path must be a string or list of strings"
        
        if isinstance(image_path, list):
            if not all(isinstance(p, str) for p in image_path):
                return False, "All image paths in list must be strings"
    
    # Validate inspector_result structure if present
    if "inspector_result" in state and state["inspector_result"] is not None:
        inspector_result = state["inspector_result"]
        if not isinstance(inspector_result, dict):
            return False, "inspector_result must be a dictionary"
        
        # Check for analysis_failed flag
        if "analysis_failed" in inspector_result and inspector_result["analysis_failed"]:
            # If failed, should have failure_reason
            if "failure_reason" not in inspector_result or not inspector_result["failure_reason"]:
                return False, "inspector_result with analysis_failed=True must have failure_reason"
    
    # Validate auditor_result structure if present
    if "auditor_result" in state and state["auditor_result"] is not None:
        auditor_result = state["auditor_result"]
        if not isinstance(auditor_result, dict):
            return False, "auditor_result must be a dictionary"
        
        # Check for analysis_failed flag
        if "analysis_failed" in auditor_result and auditor_result["analysis_failed"]:
            # If failed, should have failure_reason
            if "failure_reason" not in auditor_result or not auditor_result["failure_reason"]:
                return False, "auditor_result with analysis_failed=True must have failure_reason"
    
    return True, None


class InspectionState(TypedDict):
    """State for inspection workflow."""
    
    # Input
    image_path: Union[str, List[str]]  # Single image path or list of paths
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
    clean_verification: Optional[Dict[str, Any]]  # Clean verification result
    
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
    failure_history: Optional[List[str]]  # List of all failures encountered
    has_critical_failure: Optional[bool]  # Flag for critical failures
    inspector_retry_count: Optional[int]  # Retry counter for inspector
    auditor_retry_count: Optional[int]  # Retry counter for auditor
    image_quality: Optional[Dict[str, Any]]  # Image quality assessment result
    current_step: str
