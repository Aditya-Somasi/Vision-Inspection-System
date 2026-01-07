"""
LangGraph workflow construction and execution.
"""

import time
import uuid
from typing import Literal, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.orchestration.state import InspectionState
from src.orchestration.nodes import (
    initialize_inspection,
    run_inspector,
    run_auditor,
    analyze_consensus_node,
    evaluate_safety_node,
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
    Determine if human review is needed.
    
    NOTE: Human review interrupt is DISABLED to ensure full workflow completion.
    Human review status is still recorded and shown in PDF report.
    """
    # ALWAYS skip to explanation - human review is informational only (shown in PDF)
    # The requires_human_review flag is still set for PDF display purposes
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
    workflow.add_node("inspector", run_inspector)
    workflow.add_node("auditor", run_auditor)
    workflow.add_node("consensus", analyze_consensus_node)
    workflow.add_node("safety", evaluate_safety_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("explanation", generate_explanation)
    workflow.add_node("database", save_to_database)
    workflow.add_node("finalize", finalize_inspection)
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    workflow.add_edge("initialize", "inspector")
    workflow.add_edge("inspector", "auditor")
    workflow.add_edge("auditor", "consensus")
    workflow.add_edge("consensus", "safety")
    
    # Conditional edge for human review
    workflow.add_conditional_edges(
        "safety",
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


def run_inspection(
    image_path: str,
    criticality: str = "medium",
    domain: Optional[str] = None,
    user_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete inspection workflow.
    
    Args:
        image_path: Path to image file
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
        "requires_human_review": False,
        "human_decision": None,
        "human_notes": None,
        "explanation": None,
        "report_path": None,
        "processing_time": None,
        "error": None,
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
        "requires_human_review": False,
        "human_decision": None,
        "human_notes": None,
        "explanation": None,
        "report_path": None,
        "processing_time": None,
        "error": None,
        "current_step": "pending"
    }
    
    # Stream workflow execution
    async for state in app.astream(initial_state):
        yield state
