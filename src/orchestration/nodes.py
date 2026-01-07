"""
Workflow node functions for inspection pipeline.
"""

import time
import uuid
from pathlib import Path

from src.orchestration.state import InspectionState
from src.schemas.models import VLMAnalysisResult, InspectionContext, ConsensusResult
from src.agents import get_inspector, get_auditor, get_explainer
from src.safety import analyze_consensus, evaluate_safety
from src.database import InspectionRepository
from utils.logger import setup_logger, set_request_id
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="WORKFLOW")


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
        
        # Apply agent-inferred criticality if provided
        if result.inferred_criticality:
            user_criticality = context.criticality
            inferred = result.inferred_criticality
            
            if user_criticality != inferred:
                logger.info(
                    f"Agent inferred criticality '{inferred}' differs from user's '{user_criticality}'"
                )
                # Update context with inferred criticality (agent overrides user if higher)
                criticality_order = {"low": 0, "medium": 1, "high": 2}
                if criticality_order.get(inferred, 1) > criticality_order.get(user_criticality, 1):
                    logger.warning(
                        f"Upgrading criticality from '{user_criticality}' to '{inferred}' "
                        f"based on agent analysis: {result.inferred_criticality_reasoning}"
                    )
                    state["context"]["criticality"] = inferred
                    state["context"]["criticality_upgraded"] = True
                    state["context"]["original_criticality"] = user_criticality
                    state["context"]["upgrade_reason"] = result.inferred_criticality_reasoning
        
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
    This node uses interrupt() to pause the workflow for human input.
    
    The workflow will pause here and wait for the user to provide:
    - decision: "APPROVE", "REJECT", or "MODIFY"
    - notes: Optional additional notes
    """
    from langgraph.types import interrupt
    
    logger.warning("Human review required - workflow pausing for input")
    state["current_step"] = "awaiting_human_review"
    
    # Prepare review context for the user
    safety_verdict = state.get("safety_verdict", {})
    consensus = state.get("consensus", {})
    defects = consensus.get("combined_defects", [])
    
    review_context = {
        "type": "human_review_required",
        "reason": safety_verdict.get("triggered_gates", ["Safety gate triggered"]),
        "verdict": safety_verdict.get("verdict", "UNKNOWN"),
        "defects": [
            {
                "type": d.get("type"),
                "severity": d.get("safety_impact"),
                "location": d.get("location"),
                "reasoning": d.get("reasoning")
            }
            for d in defects
        ],
        "models_agree": consensus.get("models_agree", False),
        "agreement_score": consensus.get("agreement_score", 0),
        "options": ["APPROVE", "REJECT", "MODIFY"],
        "message": "Please review the inspection findings and provide your decision."
    }
    
    # INTERRUPT - workflow pauses here until user provides input
    human_input = interrupt(review_context)
    
    # Process human input when workflow resumes
    if isinstance(human_input, dict):
        state["human_decision"] = human_input.get("decision", "APPROVE")
        state["human_notes"] = human_input.get("notes", "")
        
        # Update verdict based on human decision
        if human_input.get("decision") == "REJECT":
            state["safety_verdict"]["verdict"] = "UNSAFE"
            state["safety_verdict"]["human_override"] = True
        elif human_input.get("decision") == "APPROVE":
            state["safety_verdict"]["verdict"] = "SAFE"
            state["safety_verdict"]["human_override"] = True
        # MODIFY keeps original verdict but adds notes
    else:
        # Simple string input
        state["human_decision"] = str(human_input)
        state["human_notes"] = ""
    
    logger.info(f"Human review complete: {state['human_decision']}")
    state["current_step"] = "human_review_complete"
    
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
                "detected_by": "inspector"
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
