"""
LangGraph workflow construction and execution.
"""

import time
import uuid
from typing import Literal, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

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


def should_run_human_review(
    state: InspectionState
) -> Literal["human_review", "generate_explanation"]:
    """Determine if human review is needed."""
    if state.get("requires_human_review"):
        return "human_review"
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
        Final inspection state
    """
    # Create workflow
    workflow = create_inspection_workflow()
    
    # Compile with checkpointer for human-in-loop support
    if config.enable_chat_memory:
        checkpointer = SqliteSaver.from_conn_string(config.chat_history_db)
        app = workflow.compile(checkpointer=checkpointer)
    else:
        app = workflow.compile()
    
    # Initial state
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
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    return final_state


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
