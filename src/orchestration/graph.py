"""
LangGraph workflow construction and execution.
"""

import time
import uuid
from typing import Literal, Dict, Any, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.orchestration.state import InspectionState
from src.orchestration.nodes import (
    initialize_inspection,
    check_image_quality,
    run_inspector,
    run_auditor,
    analyze_consensus_node,
    evaluate_safety_node,
    clean_verification_node,
    human_review_node,
    generate_explanation,
    save_to_database,
    finalize_inspection,
)
from utils.config import config
from utils.logger import setup_logger

logger = setup_logger(__name__, level=config.log_level, component="GRAPH")

# Global checkpointer for state persistence across requests
_checkpointer = InMemorySaver()

# Store active workflow apps by thread_id for resumption
_active_workflows: Dict[str, Any] = {}


def should_run_human_review(
    state: InspectionState
) -> Literal["human_review", "generate_explanation"]:
    """
    Determine if human review is needed based on safety verdict.
    
    Human review is required when:
    - requires_human_review flag is True
    - Verdict is REQUIRES_HUMAN_REVIEW
    - Critical failures detected
    - High criticality items (even when "clean")
    """
    # Check if human review is required
    requires_human = state.get("requires_human_review", False)
    has_critical_failure = state.get("has_critical_failure", False)
    
    # Check verdict
    safety_verdict = state.get("safety_verdict", {})
    verdict = safety_verdict.get("verdict", "UNKNOWN")
    
    # Check context for high criticality
    context = state.get("context", {})
    criticality = context.get("criticality", "medium")
    
    # Require human review if:
    # 1. Explicitly flagged by safety gates
    # 2. Verdict is REQUIRES_HUMAN_REVIEW
    # 3. Critical failure detected
    # 4. High criticality (conservative approach)
    if requires_human or verdict == "REQUIRES_HUMAN_REVIEW" or has_critical_failure:
        logger.info("Human review required - routing to human_review node")
        return "human_review"
    
    # For high criticality, require review even if verdict is SAFE (conservative)
    # This addresses the verification report concern about high-criticality "clean" items
    if criticality == "high":
        consensus = state.get("consensus", {})
        defect_count = len(consensus.get("combined_defects", []))
        
        # High criticality + zero defects still requires human verification
        if defect_count == 0:
            logger.info("High criticality with zero defects - requiring human review")
            return "human_review"
    
    # Otherwise, proceed to explanation
    return "generate_explanation"


def create_inspection_workflow() -> StateGraph:
    """
    Create the inspection workflow graph.
    
    Returns:
        Configured StateGraph
    """
    # Create graph
    workflow = StateGraph(InspectionState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_inspection)
    workflow.add_node("quality_check", check_image_quality)
    workflow.add_node("inspector", run_inspector)
    workflow.add_node("auditor", run_auditor)
    workflow.add_node("consensus", analyze_consensus_node)
    workflow.add_node("safety", evaluate_safety_node)
    workflow.add_node("clean_verification", clean_verification_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("explanation", generate_explanation)
    workflow.add_node("database", save_to_database)
    workflow.add_node("finalize", finalize_inspection)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    workflow.add_edge("initialize", "quality_check")
    workflow.add_edge("quality_check", "inspector")
    workflow.add_edge("inspector", "auditor")
    workflow.add_edge("auditor", "consensus")
    workflow.add_edge("consensus", "safety")
    workflow.add_edge("safety", "clean_verification")
    
    # Conditional edge for human review (after clean verification)
    workflow.add_conditional_edges(
        "clean_verification",
        should_run_human_review,
        {
            "human_review": "human_review",
            "generate_explanation": "explanation"
        }
    )
    
    # Continue from human review
    workflow.add_edge("human_review", "explanation")
    
    # Generate explanation and save
    workflow.add_edge("explanation", "database")
    workflow.add_edge("database", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


def run_single_image_inspection(
    image_path: str,
    criticality: str = "medium",
    domain: Optional[str] = None,
    user_notes: Optional[str] = None,
    image_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run inspection workflow for a single image.
    
    Args:
        image_path: Path to image file
        criticality: Criticality level (low, medium, high)
        domain: Optional domain hint
        user_notes: Optional user notes
        image_id: Optional image ID for multi-image sessions
    
    Returns:
        Final inspection state for this image
    """
    return run_inspection(image_path, criticality, domain, user_notes)


def run_inspection(
    image_path: str,
    criticality: str = "medium",
    domain: Optional[str] = None,
    user_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete inspection workflow for a single image.
    
    Args:
        image_path: Path to image file (can be str or List[str] for backward compatibility)
        criticality: Criticality level (low, medium, high)
        domain: Optional domain hint
        user_notes: Optional user notes
    
    Returns:
        Final inspection state or partial state if interrupted
    """
    # Create workflow
    workflow = create_inspection_workflow()
    
    # Compile with checkpointer for human-in-loop support
    app = workflow.compile(checkpointer=_checkpointer)
    
    # Generate unique thread ID for this inspection
    thread_id = str(uuid.uuid4())[:8]
    thread_config = {"configurable": {"thread_id": thread_id}}
    
    # Initial state
    initial_state: InspectionState = {
        "image_path": image_path,
        "context": {
            "image_id": str(uuid.uuid4())[:8],
            "criticality": criticality,
            "domain": domain,
            "user_notes": user_notes
        },
        "request_id": thread_id,
        "start_time": time.time(),
        "inspector_result": None,
        "auditor_result": None,
        "consensus": None,
        "safety_verdict": None,
        "clean_verification": None,
        "requires_human_review": False,
        "human_decision": None,
        "human_notes": None,
        "explanation": None,
        "report_path": None,
        "processing_time": None,
        "error": None,
        "failure_history": [],
        "has_critical_failure": False,
        "inspector_retry_count": 0,
        "auditor_retry_count": 0,
        "current_step": "pending"
    }
    
    # Run workflow
    try:
        final_state = app.invoke(initial_state, thread_config)
        
        # Check if we hit an interrupt (human review required)
        if final_state.get("current_step") == "awaiting_human_review":
            # Store the app and config for later resumption
            _active_workflows[thread_id] = {
                "app": app,
                "config": thread_config,
                "state": final_state
            }
            final_state["_thread_id"] = thread_id
            final_state["_requires_resume"] = True
            logger.info(f"Workflow paused for human review. Thread ID: {thread_id}")
        
        return final_state
    
    except Exception as e:
        # Check if this is an interrupt exception (expected for human-in-loop)
        if "interrupt" in str(type(e).__name__).lower() or "interrupt" in str(e).lower():
            # Get the current state from checkpointer
            state = app.get_state(thread_config)
            current_state = dict(state.values) if hasattr(state, 'values') else {}
            
            _active_workflows[thread_id] = {
                "app": app,
                "config": thread_config,
                "state": current_state
            }
            current_state["_thread_id"] = thread_id
            current_state["_requires_resume"] = True
            logger.info(f"Workflow interrupted for human review. Thread ID: {thread_id}")
            return current_state
        else:
            raise


def resume_inspection(
    thread_id: str,
    human_decision: str,
    human_notes: str = ""
) -> Dict[str, Any]:
    """
    Resume an interrupted inspection with human input.
    
    Args:
        thread_id: The thread ID from the interrupted inspection
        human_decision: APPROVE, REJECT, or MODIFY
        human_notes: Optional notes from the reviewer
    
    Returns:
        Final inspection state
    """
    if thread_id not in _active_workflows:
        raise ValueError(f"No active workflow found for thread_id: {thread_id}")
    
    workflow_info = _active_workflows[thread_id]
    app = workflow_info["app"]
    thread_config = workflow_info["config"]
    
    logger.info(f"Resuming workflow {thread_id} with decision: {human_decision}")
    
    # Resume with human input using Command
    human_input = {
        "decision": human_decision,
        "notes": human_notes
    }
    
    # Resume the workflow
    final_state = app.invoke(Command(resume=human_input), thread_config)
    
    # Clean up
    del _active_workflows[thread_id]
    
    return final_state


def run_multi_image_inspection(
    image_paths: List[str],
    criticality: str = "medium",
    domain: Optional[str] = None,
    user_notes: Optional[str] = None,
    session_id: Optional[str] = None,
    image_id_map: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Run inspection workflow for multiple images in a session.
    
    Args:
        image_paths: List of image file paths
        criticality: Criticality level for all images
        domain: Optional domain hint
        user_notes: Optional user notes
        session_id: Optional session ID for tracking
        image_id_map: Optional mapping of image_path -> image_id (to preserve IDs from uploaded_images)
    
    Returns:
        Aggregated session results with per-image results
    """
    from datetime import datetime
    from src.orchestration.session_aggregation import aggregate_session_results
    
    session_start_time = datetime.now()
    
    if not session_id:
        session_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Starting multi-image inspection session {session_id} with {len(image_paths)} images")
    
    # Per-image results storage (keyed by image_id matching uploaded_images)
    image_results = {}
    completed_count = 0
    failed_count = 0
    all_verdicts = []
    
    # Process each image sequentially
    for idx, image_path in enumerate(image_paths):
        # Use existing image_id if provided, otherwise generate new one
        if image_id_map and image_path in image_id_map:
            image_id = image_id_map[image_path]
        else:
            image_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Processing image {idx + 1}/{len(image_paths)}: {image_path} (ID: {image_id})")
        
        try:
            # Run single image inspection
            result = run_inspection(
                image_path=image_path,
                criticality=criticality,
                domain=domain,
                user_notes=user_notes
            )
            
            # Store per-image result
            image_results[image_id] = {
                "image_path": image_path,
                "inspector_result": result.get("inspector_result"),
                "auditor_result": result.get("auditor_result"),
                "consensus": result.get("consensus"),
                "safety_verdict": result.get("safety_verdict"),
                "clean_verification": result.get("clean_verification"),
                "report_path": result.get("report_path"),
                "processing_time": result.get("processing_time", 0),
                "error": result.get("error"),
                "failure_history": result.get("failure_history", []),
                "completed": True
            }
            
            # Aggregate metrics
            verdict = result.get("safety_verdict", {}).get("verdict", "UNKNOWN")
            all_verdicts.append(verdict)
            
            completed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}", exc_info=True)
            image_results[image_id] = {
                "image_path": image_path,
                "error": str(e),
                "failure_history": [str(e)],
                "completed": False
            }
            failed_count += 1
    
    # Aggregate session results using aggregation utility
    session_results_raw = aggregate_session_results(image_results)
    
    # Calculate session-level metrics
    session_end_time = datetime.now()
    session_duration = (session_end_time - session_start_time).total_seconds()
    
    # Enhance session results with timing and IDs
    session_results = {
        **session_results_raw,
        "session_id": session_id,
        "session_duration": session_duration,
        "session_start_time": session_start_time.isoformat(),
        "session_end_time": session_end_time.isoformat(),
        "per_image_verdicts": all_verdicts
    }
    
    logger.info(
        f"Multi-image session {session_id} complete: "
        f"{completed_count}/{len(image_paths)} images, "
        f"verdict: {session_results['aggregate_verdict']}"
    )
    
    return {
        "session_id": session_id,
        "image_results": image_results,
        "session_results": session_results,
        "processing_time": session_duration
    }


def get_pending_reviews() -> Dict[str, Dict[str, Any]]:
    """Get all workflows pending human review."""
    pending = {}
    for thread_id, info in _active_workflows.items():
        state = info.get("state", {})
        if state.get("current_step") == "awaiting_human_review":
            pending[thread_id] = {
                "thread_id": thread_id,
                "image_path": state.get("image_path"),
                "safety_verdict": state.get("safety_verdict"),
                "consensus": state.get("consensus"),
                "context": state.get("context")
            }
    return pending


async def run_inspection_streaming(
    image_path: str,
    criticality: str = "medium",
    domain: Optional[str] = None,
    user_notes: Optional[str] = None
):
    """
    Run inspection with streaming updates.
    
    Yields state updates for real-time UI feedback.
    """
    workflow = create_inspection_workflow()
    app = workflow.compile()
    
    initial_state: InspectionState = {
        "image_path": image_path,
        "context": {
            "image_id": str(uuid.uuid4())[:8],
            "criticality": criticality,
            "domain": domain,
            "user_notes": user_notes
        },
        "request_id": str(uuid.uuid4())[:8],
        "start_time": time.time(),
        "inspector_result": None,
        "auditor_result": None,
        "consensus": None,
        "safety_verdict": None,
        "clean_verification": None,
        "requires_human_review": False,
        "human_decision": None,
        "human_notes": None,
        "explanation": None,
        "report_path": None,
        "processing_time": None,
        "error": None,
        "failure_history": [],
        "has_critical_failure": False,
        "inspector_retry_count": 0,
        "auditor_retry_count": 0,
        "current_step": "pending"
    }
    
    # Stream workflow execution
    async for state in app.astream(initial_state):
        yield state
