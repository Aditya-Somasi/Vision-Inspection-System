"""
LangGraph workflow orchestration for vision inspection.
Implements the complete inspection pipeline with human-in-the-loop.
"""

import time
from pathlib import Path
from typing import TypedDict, Literal, Optional, Dict, Any, List
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from models import (
    InspectionContext, VLMAnalysisResult, ConsensusResult,
    SafetyVerdict, get_inspector, get_auditor, get_explainer
)
from safety import analyze_consensus, evaluate_safety
from database import InspectionRepository
from logger import setup_logger, set_request_id, print_processing_status
from config import config

logger = setup_logger(__name__, level=config.log_level, component="WORKFLOW")


# ============================================================================
# STATE DEFINITION
# ============================================================================

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


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def initialize_inspection(state: InspectionState) -> InspectionState:
    """Initialize inspection state."""
    logger.info("=" * 80)
    logger.info("STARTING NEW INSPECTION")
    logger.info("=" * 80)
    
    # Set request ID for logging correlation
    request_id = state.get("request_id") or str(uuid.uuid4())[:8]
    set_request_id(request_id)
    
    context_dict = state["context"]
    logger.info(f"Image: {state['image_path']}")
    logger.info(f"Criticality: {context_dict.get('criticality', 'unknown')}")
    logger.info(f"Domain: {context_dict.get('domain', 'unknown')}")
    
    state["request_id"] = request_id
    state["start_time"] = time.time()
    state["current_step"] = "initialized"
    state["requires_human_review"] = False
    
    return state


def run_inspector(state: InspectionState) -> InspectionState:
    """Run Inspector VLM analysis."""
    logger.info("Running Inspector (Qwen2-VL) analysis...")
    state["current_step"] = "inspector_analysis"
    
    try:
        # Create context object
        context_dict = state["context"]
        context = InspectionContext(**context_dict)
        
        # Get inspector agent
        inspector = get_inspector()
        
        # Analyze image
        result = inspector.analyze(Path(state["image_path"]), context)
        
        # Store result as dict
        state["inspector_result"] = result.model_dump()
        
        logger.info(f"Inspector found {len(result.defects)} defects")
        
    except Exception as e:
        logger.error(f"Inspector analysis failed: {e}", exc_info=True)
        state["error"] = f"Inspector failed: {str(e)}"
    
    return state


def run_auditor(state: InspectionState) -> InspectionState:
    """Run Auditor VLM verification."""
    logger.info("Running Auditor (Llama 3.2) verification...")
    state["current_step"] = "auditor_verification"
    
    try:
        # Create context object
        context_dict = state["context"]
        context = InspectionContext(**context_dict)
        
        # Reconstruct inspector result
        inspector_result = VLMAnalysisResult(**state["inspector_result"])
        
        # Get auditor agent
        auditor = get_auditor()
        
        # Verify
        result = auditor.verify(
            Path(state["image_path"]),
            context,
            inspector_result
        )
        
        # Store result as dict
        state["auditor_result"] = result.model_dump()
        
        logger.info(f"Auditor found {len(result.defects)} defects")
        
    except Exception as e:
        logger.error(f"Auditor verification failed: {e}", exc_info=True)
        state["error"] = f"Auditor failed: {str(e)}"
    
    return state


def analyze_consensus_node(state: InspectionState) -> InspectionState:
    """Analyze consensus between Inspector and Auditor."""
    logger.info("Analyzing consensus between models...")
    state["current_step"] = "consensus_analysis"
    
    try:
        # Reconstruct VLM results
        inspector_result = VLMAnalysisResult(**state["inspector_result"])
        auditor_result = VLMAnalysisResult(**state["auditor_result"])
        
        # Analyze consensus
        consensus = analyze_consensus(inspector_result, auditor_result)
        
        # Store as dict
        state["consensus"] = consensus.model_dump()
        
        logger.info(
            f"Consensus: {'AGREE' if consensus.models_agree else 'DISAGREE'} "
            f"(score: {consensus.agreement_score:.2f})"
        )
        
    except Exception as e:
        logger.error(f"Consensus analysis failed: {e}", exc_info=True)
        state["error"] = f"Consensus failed: {str(e)}"
    
    return state


def evaluate_safety_node(state: InspectionState) -> InspectionState:
    """Evaluate safety using safety gates."""
    logger.info("Evaluating safety gates...")
    state["current_step"] = "safety_evaluation"
    
    try:
        # Reconstruct objects
        context = InspectionContext(**state["context"])
        consensus = ConsensusResult(**state["consensus"])
        
        # Evaluate safety
        verdict = evaluate_safety(consensus, context)
        
        # Store as dict
        state["safety_verdict"] = verdict.model_dump()
        state["requires_human_review"] = verdict.requires_human
        
        logger.info(f"Safety verdict: {verdict.verdict}")
        logger.info(f"Requires human review: {verdict.requires_human}")
        
        if verdict.triggered_gates:
            logger.info(f"Triggered gates: {', '.join(verdict.triggered_gates)}")
        
    except Exception as e:
        logger.error(f"Safety evaluation failed: {e}", exc_info=True)
        state["error"] = f"Safety evaluation failed: {str(e)}"
    
    return state


def human_review_node(state: InspectionState) -> InspectionState:
    """
    Human review decision point.
    This node will interrupt the workflow for human input.
    """
    logger.warning("Human review required - workflow paused")
    state["current_step"] = "awaiting_human_review"
    
    # In actual implementation, this would wait for human input
    # For now, we'll just mark it and continue
    # The UI will handle the interrupt
    
    return state


def generate_explanation(state: InspectionState) -> InspectionState:
    """Generate human-readable explanation."""
    logger.info("Generating explanation...")
    state["current_step"] = "generating_explanation"
    
    try:
        # Reconstruct objects
        inspector_result = VLMAnalysisResult(**state["inspector_result"])
        auditor_result = VLMAnalysisResult(**state["auditor_result"])
        consensus = state["consensus"]
        verdict = state["safety_verdict"]
        
        # Get explainer
        explainer = get_explainer()
        
        # Generate explanation
        explanation = explainer.generate_explanation(
            inspector_result,
            auditor_result,
            consensus,
            verdict
        )
        
        state["explanation"] = explanation
        
        logger.info("Explanation generated successfully")
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}", exc_info=True)
        state["explanation"] = (
            f"Inspection complete. Verdict: {state['safety_verdict'].get('verdict', 'UNKNOWN')}. "
            f"See detailed findings in report."
        )
    
    return state


def save_to_database(state: InspectionState) -> InspectionState:
    """Save inspection record to database."""
    logger.info("Saving inspection to database...")
    state["current_step"] = "saving_to_database"
    
    try:
        repo = InspectionRepository()
        context = state["context"]
        verdict = state["safety_verdict"]
        consensus = state["consensus"]
        
        # Prepare inspection data
        inspection_data = {
            "inspection_id": state["request_id"],
            "image_path": state["image_path"],
            "image_filename": Path(state["image_path"]).name,
            "criticality": context.get("criticality"),
            "domain": context.get("domain"),
            "user_notes": context.get("user_notes"),
            "overall_verdict": verdict["verdict"],
            "defect_count": len(consensus["combined_defects"]),
            "critical_defect_count": sum(
                1 for d in consensus["combined_defects"]
                if d["safety_impact"] == "CRITICAL"
            ),
            "inspector_confidence": state["inspector_result"]["overall_confidence"],
            "auditor_confidence": state["auditor_result"]["overall_confidence"],
            "models_agree": consensus["models_agree"],
            "agreement_score": consensus["agreement_score"],
            "triggered_gates": verdict["triggered_gates"],
            "requires_human": verdict["requires_human"],
            "processing_time_seconds": time.time() - state["start_time"],
            "report_path": state.get("report_path")
        }
        
        # Prepare defects data
        defects_data = []
        for defect in consensus["combined_defects"]:
            defect_record = {
                "defect_id": defect["defect_id"],
                "defect_type": defect["type"],
                "location": defect["location"],
                "safety_impact": defect["safety_impact"],
                "reasoning": defect["reasoning"],
                "confidence": defect["confidence"],
                "recommended_action": defect["recommended_action"],
                "detected_by": "inspector"  # Could be enhanced to track which model found it
            }
            
            # Add bounding box if available
            if defect.get("bbox"):
                bbox = defect["bbox"]
                defect_record.update({
                    "bbox_x": bbox.get("x"),
                    "bbox_y": bbox.get("y"),
                    "bbox_width": bbox.get("width"),
                    "bbox_height": bbox.get("height")
                })
            
            defects_data.append(defect_record)
        
        # Save to database
        inspection = repo.create_inspection(inspection_data, defects_data)
        
        logger.info(f"Inspection saved with ID: {inspection.inspection_id}")
        
    except Exception as e:
        logger.error(f"Database save failed: {e}", exc_info=True)
        state["error"] = f"Database save failed: {str(e)}"
    
    return state


def finalize_inspection(state: InspectionState) -> InspectionState:
    """Finalize inspection and log results."""
    state["current_step"] = "completed"
    state["processing_time"] = time.time() - state["start_time"]
    
    logger.info("=" * 80)
    logger.info("INSPECTION COMPLETE")
    logger.info(f"Request ID: {state['request_id']}")
    logger.info(f"Verdict: {state['safety_verdict']['verdict']}")
    logger.info(f"Processing time: {state['processing_time']:.2f}s")
    logger.info("=" * 80)
    
    return state


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_run_human_review(state: InspectionState) -> Literal["human_review", "generate_explanation"]:
    """Determine if human review is needed."""
    if state.get("requires_human_review"):
        return "human_review"
    return "generate_explanation"


def has_error(state: InspectionState) -> Literal["error", "continue"]:
    """Check if an error occurred."""
    if state.get("error"):
        return "error"
    return "continue"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

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


# ============================================================================
# WORKFLOW EXECUTION
# ============================================================================

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


# ============================================================================
# STREAMING SUPPORT
# ============================================================================

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