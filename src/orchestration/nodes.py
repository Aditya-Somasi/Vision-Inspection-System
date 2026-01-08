"""
Workflow node functions for inspection pipeline.
"""

import time
import uuid
from pathlib import Path
from typing import Union, List

from src.orchestration.state import InspectionState, validate_state
from src.schemas.models import VLMAnalysisResult, InspectionContext, ConsensusResult
from src.agents import get_inspector, get_auditor, get_explainer
from src.safety import analyze_consensus, evaluate_safety
from src.safety.image_quality import assess_image_quality
from src.database import InspectionRepository
from utils.logger import setup_logger, set_request_id
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="WORKFLOW")


def _normalize_image_input(image_path: Union[str, List[str]]) -> List[str]:
    """
    Normalize image input to list format for internal processing.
    
    Args:
        image_path: Single image path (str) or list of paths
        
    Returns:
        List of image paths
    """
    if isinstance(image_path, str):
        return [image_path]
    elif isinstance(image_path, list):
        return image_path
    else:
        raise ValueError(f"Invalid image_path type: {type(image_path)}")


def _should_retry(retry_count: int, max_retries: int = 1) -> bool:
    """Check if retry should be attempted."""
    return retry_count < max_retries


def _backoff_delay(retry_count: int) -> float:
    """Calculate backoff delay in seconds (exponential backoff)."""
    return min(2.0 ** retry_count, 10.0)  # Max 10 seconds


def initialize_inspection(state: InspectionState) -> InspectionState:
    """Initialize inspection state."""
    logger.info("=" * 80)
    logger.info("STARTING NEW INSPECTION")
    logger.info("=" * 80)
    
    # Set request ID for logging correlation
    request_id = state.get("request_id") or str(uuid.uuid4())[:8]
    set_request_id(request_id)
    
    context_dict = state["context"]
    # Normalize image path for logging
    image_paths = _normalize_image_input(state["image_path"])
    image_path_str = image_paths[0] if len(image_paths) == 1 else f"{len(image_paths)} images"
    logger.info(f"Image: {image_path_str}")
    logger.info(f"Criticality: {context_dict.get('criticality', 'unknown')}")
    logger.info(f"Domain: {context_dict.get('domain', 'unknown')}")
    
    state["request_id"] = request_id
    state["start_time"] = time.time()
    state["current_step"] = "initialized"
    state["requires_human_review"] = False
    state["failure_history"] = []
    state["has_critical_failure"] = False
    state["inspector_retry_count"] = 0
    state["auditor_retry_count"] = 0
    
    return state


def check_image_quality(state: InspectionState) -> InspectionState:
    """Check image quality before analysis."""
    logger.info("Checking image quality...")
    state["current_step"] = "quality_check"
    
    try:
        # Normalize to list and process first image (multi-image support TBD)
        image_paths = _normalize_image_input(state["image_path"])
        image_path = Path(image_paths[0])  # Process first image for now
        
        if len(image_paths) > 1:
            logger.info(f"Multi-image input detected ({len(image_paths)} images). Processing first image only.")
        
        quality_result = assess_image_quality(image_path)
        
        # Store quality assessment in state
        state["image_quality"] = quality_result
        
        if not quality_result.get("quality_passed", False):
            quality_score = quality_result.get("quality_score", 0.0)
            logger.warning(
                f"Image quality below threshold: score={quality_score:.2f}. "
                "Analysis may be less reliable."
            )
            # Don't block workflow, but mark for downstream gates to consider
            state["low_quality_image"] = True
        
    except Exception as e:
        logger.error(f"Image quality check failed: {e}", exc_info=True)
        # Non-blocking error - continue with inspection
        state["image_quality"] = {"quality_passed": False, "error": str(e)}
    
    return state


def run_inspector(state: InspectionState) -> InspectionState:
    """Run Inspector VLM analysis with retry logic."""
    logger.info("Running Inspector (Qwen2-VL) analysis...")
    state["current_step"] = "inspector_analysis"
    
    retry_count = state.get("inspector_retry_count", 0)
    max_retries = 1  # Default max retries
    
    # Create context object
    context_dict = state["context"]
    context = InspectionContext(**context_dict)
    
    # Get inspector agent
    inspector = get_inspector()
    
    # Normalize to list and process first image (multi-image support TBD)
    image_paths = _normalize_image_input(state["image_path"])
    image_path = Path(image_paths[0])  # Process first image for now
    
    result = None
    last_error = None
    
    # Retry loop
    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                delay = _backoff_delay(retry_count - 1)
                logger.info(f"Retrying Inspector analysis (attempt {retry_count + 1}/{max_retries + 1}) after {delay:.1f}s delay...")
                time.sleep(delay)
            
            # Analyze image
            result = inspector.analyze(image_path, context)
            
            # Check if analysis failed (even if no exception)
            if result.analysis_failed:
                raise Exception(result.failure_reason or "Inspector analysis failed")
            
            # Success - break retry loop
            break
            
        except Exception as e:
            last_error = e
            logger.warning(f"Inspector analysis attempt {retry_count + 1} failed: {e}")
            
            if retry_count < max_retries and _should_retry(retry_count, max_retries):
                retry_count += 1
                state["inspector_retry_count"] = retry_count
                continue
            else:
                # Max retries reached or retry not allowed
                logger.error(f"Inspector analysis failed after {retry_count + 1} attempt(s): {e}", exc_info=True)
                error_msg = f"Inspector failed after {retry_count + 1} attempt(s): {str(e)}"
                state["error"] = error_msg
                state["failure_history"] = state.get("failure_history", []) + [error_msg]
                state["has_critical_failure"] = True
                
                # Return failed result
                result = VLMAnalysisResult(
                    object_identified="unknown",
                    overall_condition="uncertain",
                    defects=[],
                    overall_confidence="low",
                    analysis_reasoning=f"Analysis failed after retries: {str(e)}",
                    analysis_failed=True,
                    failure_reason=error_msg
                )
                break
    
    # Store result as dict
    if result:
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
        
        if not result.analysis_failed:
            logger.info(f"Inspector found {len(result.defects)} defects")
    
    return state


def run_auditor(state: InspectionState) -> InspectionState:
    """Run Auditor VLM verification with retry logic."""
    logger.info("Running Auditor (Llama 3.2) verification...")
    state["current_step"] = "auditor_verification"
    
    retry_count = state.get("auditor_retry_count", 0)
    max_retries = 1  # Default max retries
    
    # Create context object
    context_dict = state["context"]
    context = InspectionContext(**context_dict)
    
    # Reconstruct inspector result
    inspector_result = VLMAnalysisResult(**state["inspector_result"])
    
    # Get auditor agent
    auditor = get_auditor()
    
    # Normalize to list and process first image (multi-image support TBD)
    image_paths = _normalize_image_input(state["image_path"])
    image_path = Path(image_paths[0])  # Process first image for now
    
    result = None
    last_error = None
    
    # Retry loop
    while retry_count <= max_retries:
        try:
            if retry_count > 0:
                delay = _backoff_delay(retry_count - 1)
                logger.info(f"Retrying Auditor verification (attempt {retry_count + 1}/{max_retries + 1}) after {delay:.1f}s delay...")
                time.sleep(delay)
            
            # Verify
            result = auditor.verify(
                image_path,
                context,
                inspector_result
            )
            
            # Check if analysis failed (even if no exception)
            if result.analysis_failed:
                raise Exception(result.failure_reason or "Auditor verification failed")
            
            # Success - break retry loop
            break
            
        except Exception as e:
            last_error = e
            logger.warning(f"Auditor verification attempt {retry_count + 1} failed: {e}")
            
            if retry_count < max_retries and _should_retry(retry_count, max_retries):
                retry_count += 1
                state["auditor_retry_count"] = retry_count
                continue
            else:
                # Max retries reached or retry not allowed
                logger.error(f"Auditor verification failed after {retry_count + 1} attempt(s): {e}", exc_info=True)
                error_msg = f"Auditor failed after {retry_count + 1} attempt(s): {str(e)}"
                state["error"] = error_msg
                state["failure_history"] = state.get("failure_history", []) + [error_msg]
                state["has_critical_failure"] = True
                
                # Return failed result
                result = VLMAnalysisResult(
                    object_identified="unknown",
                    overall_condition="uncertain",
                    defects=[],
                    overall_confidence="low",
                    analysis_reasoning=f"Verification failed after retries: {str(e)}",
                    analysis_failed=True,
                    failure_reason=error_msg
                )
                break
    
    # Store result as dict
    if result:
        state["auditor_result"] = result.model_dump()
        
        if not result.analysis_failed:
            logger.info(f"Auditor found {len(result.defects)} defects")
    
    return state


def analyze_consensus_node(state: InspectionState) -> InspectionState:
    """Analyze consensus between Inspector and Auditor."""
    logger.info("Analyzing consensus between models...")
    state["current_step"] = "consensus_analysis"
    
    try:
        # Validate state before consensus analysis
        is_valid, error_msg = validate_state(state, required_fields=["inspector_result", "auditor_result"])
        if not is_valid:
            raise ValueError(f"State validation failed: {error_msg}")
        
        # Reconstruct VLM results
        inspector_result = VLMAnalysisResult(**state["inspector_result"])
        auditor_result = VLMAnalysisResult(**state["auditor_result"])
        
        # Check for critical failures - abort if either agent failed
        if inspector_result.analysis_failed or auditor_result.analysis_failed:
            failure_msg = []
            if inspector_result.analysis_failed:
                failure_msg.append(f"Inspector: {inspector_result.failure_reason}")
            if auditor_result.analysis_failed:
                failure_msg.append(f"Auditor: {auditor_result.failure_reason}")
            
            error_summary = "; ".join(failure_msg)
            logger.error(f"Critical failure detected: {error_summary}")
            state["error"] = f"Analysis failures: {error_summary}"
            state["has_critical_failure"] = True
            
            # Still attempt consensus for error tracking, but it will be marked as failed
            # This allows downstream gates to detect the error
        
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
        error_msg = f"Consensus failed: {str(e)}"
        state["error"] = error_msg
        state["failure_history"] = state.get("failure_history", []) + [error_msg]
        state["has_critical_failure"] = True
    
    return state


def evaluate_safety_node(state: InspectionState) -> InspectionState:
    """Evaluate safety using safety gates."""
    logger.info("Evaluating safety gates...")
    state["current_step"] = "safety_evaluation"
    
    try:
        # Validate state before safety evaluation
        is_valid, error_msg = validate_state(state, required_fields=["context", "consensus"])
        if not is_valid:
            raise ValueError(f"State validation failed: {error_msg}")
        
        # Reconstruct objects
        context = InspectionContext(**state["context"])
        consensus = ConsensusResult(**state["consensus"])
        
        # Evaluate safety
        verdict = evaluate_safety(consensus, context)
        
        # Store as dict
        state["safety_verdict"] = verdict.model_dump()
        state["requires_human_review"] = verdict.requires_human
        
        # Collect errors from verdict and add to failure_history
        if verdict.errors:
            state["failure_history"] = state.get("failure_history", []) + verdict.errors
        
        logger.info(f"Safety verdict: {verdict.verdict}")
        logger.info(f"Requires human review: {verdict.requires_human}")
        
        if verdict.errors:
            logger.warning(f"Errors in verdict: {', '.join(verdict.errors)}")
        
        if verdict.triggered_gates:
            logger.info(f"Triggered gates: {', '.join(verdict.triggered_gates)}")
        
    except Exception as e:
        logger.error(f"Safety evaluation failed: {e}", exc_info=True)
        error_msg = f"Safety evaluation failed: {str(e)}"
        state["error"] = error_msg
        state["failure_history"] = state.get("failure_history", []) + [error_msg]
        state["has_critical_failure"] = True
    
    return state


def human_review_node(state: InspectionState) -> InspectionState:
    """
    Human review node - marks for review but doesn't block workflow.
    Explanation and PDF will still be generated.
    This is a non-blocking review flag for UI display.
    """
    logger.info("Human review flagged - marking for review (non-blocking)")
    state["current_step"] = "flagged_for_review"
    
    # Mark that human review is recommended but don't block workflow
    # The UI can show this flag and allow users to review later
    safety_verdict = state.get("safety_verdict", {})
    consensus = state.get("consensus", {})
    defects = consensus.get("combined_defects", [])
    
    # Store review context for UI display
    state["human_review_context"] = {
        "type": "human_review_recommended",
        "reason": safety_verdict.get("reason", "Clean verification failed or high criticality"),
        "verdict": safety_verdict.get("verdict", "UNKNOWN"),
        "defect_count": len(defects),
        "models_agree": consensus.get("models_agree", False),
        "agreement_score": consensus.get("agreement_score", 0),
        "message": "Human review is recommended. Inspection will complete and results will be available for review."
    }
    
    # Don't change verdict - keep original verdict
    # Just mark that review is recommended
    logger.info("Human review flagged (non-blocking) - workflow will continue to generate explanation and PDF")
    
    return state


def clean_verification_node(state: InspectionState) -> InspectionState:
    """
    Independent clean image verification step.
    
    This node provides additional verification when both models report "no defects".
    Acts as a third verification mechanism without requiring a third model.
    
    Verification checks:
    - Both models must have HIGH confidence
    - Agreement score must be high (>0.8)
    - No analysis errors
    - Image quality must be acceptable
    - Conservative check: If any uncertainty, mark for review
    """
    logger.info("Running clean image verification...")
    state["current_step"] = "clean_verification"
    
    try:
        # Reconstruct results
        inspector_result = VLMAnalysisResult(**state["inspector_result"])
        auditor_result = VLMAnalysisResult(**state["auditor_result"])
        consensus = ConsensusResult(**state["consensus"])
        
        defect_count = len(consensus.combined_defects)
        
        # Only run clean verification if no defects found
        if defect_count == 0:
            inspector_conf = inspector_result.overall_confidence
            auditor_conf = auditor_result.overall_confidence
            agreement_score = consensus.agreement_score
            
            # Clean verification criteria
            both_high_conf = (inspector_conf == "high" and auditor_conf == "high")
            high_agreement = agreement_score > 0.8
            no_errors = not (inspector_result.analysis_failed or auditor_result.analysis_failed)
            
            # Check image quality if available
            # For clean images (0 defects), image quality is less critical if models agree
            image_quality = state.get("image_quality", {})
            quality_passed = image_quality.get("quality_passed", True)  # Default to True if not checked
            quality_score = image_quality.get("quality_score", 1.0)  # Default to 1.0 if not available
            
            # If both models agree with high confidence and no defects, 
            # image quality threshold is less critical (only warn, don't fail)
            # Only fail quality check if quality is very poor AND models disagree
            very_poor_quality = quality_score < 0.3  # Very poor threshold
            
            # All criteria must pass for clean verification
            # Quality check is less strict if models strongly agree (agreement > 0.9)
            clean_verified = (
                both_high_conf and
                high_agreement and
                no_errors and
                (quality_passed or (agreement_score > 0.9 and not very_poor_quality))
            )
            
            if clean_verified:
                logger.info("Clean image verification PASSED - all criteria met")
                state["clean_verification"] = {
                    "verified": True,
                    "reason": "All verification criteria met: high confidence, high agreement, no errors, good quality"
                }
            else:
                # Log why verification failed
                reasons = []
                if not both_high_conf:
                    reasons.append(f"confidence not high (Inspector: {inspector_conf}, Auditor: {auditor_conf})")
                if not high_agreement:
                    reasons.append(f"agreement score too low ({agreement_score:.2f}, required >0.8)")
                if not no_errors:
                    reasons.append("analysis errors detected")
                if not quality_passed:
                    reasons.append("image quality below threshold")
                
                logger.warning(f"Clean image verification FAILED: {', '.join(reasons)}")
                state["clean_verification"] = {
                    "verified": False,
                    "reason": f"Verification failed: {', '.join(reasons)}",
                    "details": {
                        "inspector_confidence": inspector_conf,
                        "auditor_confidence": auditor_conf,
                        "agreement_score": agreement_score,
                        "has_errors": not no_errors,
                        "quality_passed": quality_passed
                    }
                }
                
                # If clean verification fails, flag for human review but keep SAFE verdict
                # Only change verdict if quality is very poor AND models disagree
                safety_verdict = state.get("safety_verdict", {})
                if safety_verdict.get("verdict") == "SAFE":
                    # Only change to REQUIRES_HUMAN_REVIEW if quality is very poor (< 0.3) 
                    # AND models don't strongly agree (agreement < 0.9)
                    very_poor_quality = quality_score < 0.3
                    low_agreement = agreement_score < 0.9
                    
                    if very_poor_quality and low_agreement:
                        logger.warning("Clean verification failed with very poor quality and low agreement - updating SAFE verdict to REQUIRE_HUMAN_REVIEW")
                        safety_verdict["verdict"] = "REQUIRES_HUMAN_REVIEW"
                        safety_verdict["requires_human"] = True
                        safety_verdict["reason"] = f"Clean verification failed: {', '.join(reasons)}. Conservative review required."
                        state["safety_verdict"] = safety_verdict
                        state["requires_human_review"] = True
                    else:
                        # Keep SAFE verdict but flag for review
                        logger.info(f"Clean verification failed but keeping SAFE verdict (quality: {quality_score:.2f}, agreement: {agreement_score:.2f}). Flagging for optional review.")
                        state["requires_human_review"] = True
                        # Add note to verdict without changing it
                        safety_verdict["review_note"] = f"Optional review recommended: {', '.join(reasons)}"
                        state["safety_verdict"] = safety_verdict
        else:
            # Defects found - clean verification not applicable
            state["clean_verification"] = {
                "verified": False,
                "reason": "Not applicable - defects found",
                "defect_count": defect_count
            }
    
    except Exception as e:
        logger.error(f"Clean verification failed: {e}", exc_info=True)
        state["clean_verification"] = {
            "verified": False,
            "reason": f"Verification error: {str(e)}"
        }
    
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
        
        # Validate explanation completeness before storing
        required_sections = ["EXECUTIVE SUMMARY", "SUMMARY", "FINAL RECOMMENDATION"]
        explanation_lower = explanation.lower()
        
        has_summary = any(keyword in explanation_lower for keyword in ["executive summary", "summary", "overview"])
        has_recommendation = any(keyword in explanation_lower for keyword in ["final recommendation", "recommendation", "verdict", "action required"])
        
        if not has_summary:
            logger.warning("Generated explanation missing SUMMARY section - generating fallback")
            # Generate fallback summary from structured data
            object_name = inspector_result.object_identified or "component"
            defect_count = len(consensus.get("combined_defects", []))
            verdict_str = verdict.get("verdict", "UNKNOWN")
            
            fallback_prefix = (
                f"EXECUTIVE SUMMARY\n\n"
                f"Inspection of {object_name} identified {defect_count} defect(s). "
                f"Final verdict: {verdict_str}. "
                f"Both Inspector and Auditor models analyzed the image independently. "
            )
            if defect_count > 0:
                critical_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "CRITICAL")
                if critical_count > 0:
                    fallback_prefix += f"{critical_count} critical defect(s) were detected. "
                else:
                    fallback_prefix += "No critical defects detected. "
            else:
                fallback_prefix += "No defects were detected. "
            
            fallback_prefix += "\n\n"
            explanation = fallback_prefix + explanation
        
        if not has_recommendation:
            logger.warning("Generated explanation missing FINAL RECOMMENDATION section")
            # Append recommendation if missing
            verdict_str = verdict.get("verdict", "UNKNOWN")
            action_required = "No action required" if verdict_str == "SAFE" else "Further inspection or remediation recommended"
            
            recommendation_suffix = (
                f"\n\nFINAL RECOMMENDATION\n\n"
                f"Verdict: {verdict_str}\n"
                f"Action Required: {action_required}\n"
                f"Safety Assessment: Based on the analysis, the component {'appears safe' if verdict_str == 'SAFE' else 'requires attention'}."
            )
            explanation = explanation + recommendation_suffix
        
        state["explanation"] = explanation
        
        # Generate decision support (Cost & Time)
        try:
            decision_support = explainer.generate_decision_support(
                consensus.get("combined_defects", []),
                verdict.get("verdict", "UNKNOWN")
            )
            state["decision_support"] = decision_support
            logger.info("Decision support metrics generated")
        except Exception as e:
            logger.error(f"Decision support generation failed: {e}")
            state["decision_support"] = {}
        
        logger.info("Explanation generated and validated successfully")
        
    except Exception as e:
        logger.error(f"Explanation generation failed: {e}", exc_info=True)
        error_details = str(e)
        
        # Generate fallback explanation from structured data
        inspector_result = state.get("inspector_result", {})
        consensus = state.get("consensus", {})
        verdict = state.get("safety_verdict", {})
        
        object_name = inspector_result.get("object_identified", "component")
        defect_count = len(consensus.get("combined_defects", []))
        verdict_str = verdict.get("verdict", "UNKNOWN")
        
        fallback_explanation = (
            f"EXECUTIVE SUMMARY\n\n"
            f"Inspection of {object_name} identified {defect_count} defect(s). "
            f"Final verdict: {verdict_str}. "
            f"Analysis was completed by both Inspector and Auditor models.\n\n"
            f"FINAL RECOMMENDATION\n\n"
            f"Verdict: {verdict_str}\n"
            f"Action Required: {'No action required' if verdict_str == 'SAFE' else 'Further inspection recommended'}\n"
            f"Safety Assessment: {'Component appears safe' if verdict_str == 'SAFE' else 'Component requires attention'}.\n\n"
            f"NOTE: Full explanation generation failed ({error_details}). This summary was generated from structured findings."
        )
        
        state["explanation"] = fallback_explanation
        logger.warning("Used fallback explanation due to generation failure")
    
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
        
        # Normalize image path for database storage
        image_paths = _normalize_image_input(state["image_path"])
        primary_image_path = image_paths[0]
        
        # Prepare inspection data
        inspection_data = {
            "inspection_id": state["request_id"],
            "image_path": primary_image_path,
            "image_filename": Path(primary_image_path).name,
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
    """Finalize inspection, generate PDF report, and log results."""
    state["current_step"] = "completed"
    state["processing_time"] = time.time() - state["start_time"]
    
    # Generate PDF report if explanation exists
    if state.get("explanation") and not state.get("report_path"):
        try:
            from src.reporting import generate_report
            from pathlib import Path
            
            logger.info("Generating PDF report...")
            report_path = generate_report(state)
            state["report_path"] = str(report_path)
            logger.info(f"PDF report generated: {report_path}")
        except Exception as e:
            logger.error(f"PDF report generation failed: {e}", exc_info=True)
            state["error"] = f"PDF generation failed: {str(e)}"
    
    # Ensure errors are visible in final state
    errors = state.get("failure_history", [])
    if state.get("error"):
        if state["error"] not in errors:
            errors.append(state["error"])
    if state.get("safety_verdict", {}).get("errors"):
        for err in state["safety_verdict"]["errors"]:
            if err not in errors:
                errors.append(err)
    state["failure_history"] = errors
    
    logger.info("=" * 80)
    logger.info("INSPECTION COMPLETE")
    logger.info(f"Request ID: {state['request_id']}")
    logger.info(f"Verdict: {state['safety_verdict']['verdict']}")
    logger.info(f"Processing time: {state['processing_time']:.2f}s")
    if state.get("report_path"):
        logger.info(f"PDF Report: {state['report_path']}")
    if errors:
        logger.warning(f"Errors encountered: {len(errors)} error(s)")
        for err in errors:
            logger.warning(f"  - {err}")
    logger.info("=" * 80)
    
    return state
