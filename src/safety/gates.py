"""
Safety gate engine for deterministic safety evaluation.
Shows ALL gates evaluated (pass/fail) not just triggered ones.
Trusts agent severity assessment with configurable domain-specific overrides.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from src.schemas.models import (
    ConsensusResult,
    SafetyVerdict,
    InspectionContext,
)
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="SAFETY")

# Load safety rules from YAML
SAFETY_RULES_PATH = Path(__file__).parent.parent.parent / "config" / "safety_rules.yaml"


# ============================================================================
# GATE CONSTANTS - Use these consistently in code and tests
# ============================================================================
GATE_ERROR_STATE = "GATE_0_ERROR_STATE"
GATE_CRITICAL_DEFECT = "GATE_1_CRITICAL_DEFECT"
GATE_DOMAIN_ZERO_TOLERANCE = "GATE_2_DOMAIN_ZERO_TOLERANCE"
GATE_MODEL_DISAGREEMENT = "GATE_3_MODEL_DISAGREEMENT"
GATE_LOW_CONFIDENCE = "GATE_4_LOW_CONFIDENCE"
GATE_DEFECT_COUNT = "GATE_5_DEFECT_COUNT"
GATE_HIGH_CRITICALITY = "GATE_6_HIGH_CRITICALITY"
GATE_NO_DEFECTS = "GATE_7_NO_DEFECTS"
GATE_AUDITOR_UNCERTAIN = "GATE_8_AUDITOR_UNCERTAIN"
GATE_DEFAULT_CONSERVATIVE = "GATE_DEFAULT_CONSERVATIVE"

# Mapping for display names
GATE_DISPLAY_NAMES = {
    GATE_ERROR_STATE: "Error State Check",
    GATE_CRITICAL_DEFECT: "Critical Defect Check",
    GATE_DOMAIN_ZERO_TOLERANCE: "Domain Zero Tolerance",
    GATE_MODEL_DISAGREEMENT: "Model Agreement Check",
    GATE_LOW_CONFIDENCE: "Confidence Threshold",
    GATE_DEFECT_COUNT: "Defect Count Limit",
    GATE_HIGH_CRITICALITY: "High Criticality Check", 
    GATE_NO_DEFECTS: "No Defects Verification",
    GATE_AUDITOR_UNCERTAIN: "Auditor Certainty Check",
    GATE_DEFAULT_CONSERVATIVE: "Conservative Fallback",
}


def load_safety_rules() -> Dict[str, Any]:
    """Load safety rules from YAML config."""
    try:
        if SAFETY_RULES_PATH.exists():
            with open(SAFETY_RULES_PATH, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load safety_rules.yaml: {e}")
    return {}


class GateResult:
    """Result of evaluating a single gate."""
    
    def __init__(
        self,
        gate_id: str,
        passed: bool,
        message: str = "",
        details: Optional[Dict[str, Any]] = None
    ):
        self.gate_id = gate_id
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    @property
    def display_name(self) -> str:
        return GATE_DISPLAY_NAMES.get(self.gate_id, self.gate_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "display_name": self.display_name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details
        }


class SafetyGateEngine:
    """
    Deterministic safety gate engine.
    Evaluates ALL gates and tracks pass/fail for each.
    Trusts agent severity assessment with domain-specific flagging.
    """
    
    def __init__(self):
        self.logger = logger
        self.rules = load_safety_rules()
        self.domains = self.rules.get("domains", {})
        self.agent_trust = self.rules.get("agent_trust", {"trust_agent_severity": True})
    
    def _get_domain_rules(self, domain: Optional[str]) -> Dict[str, Any]:
        """Get domain-specific rules or fall back to general."""
        if domain and domain.lower() in self.domains:
            return self.domains[domain.lower()]
        return self.domains.get("general", {})
    
    def _should_flag_for_domain(
        self,
        defect_type: str,
        domain: Optional[str]
    ) -> bool:
        """Check if defect type should be flagged for this domain."""
        domain_rules = self._get_domain_rules(domain)
        zero_tolerance = domain_rules.get("zero_tolerance_types", [])
        
        defect_lower = defect_type.lower()
        for zt_type in zero_tolerance:
            if zt_type.lower() in defect_lower or defect_lower in zt_type.lower():
                return True
        return False
    
    def _confidence_to_numeric(self, conf: str) -> float:
        """Convert confidence string to numeric value."""
        mapping = {"high": 0.9, "medium": 0.6, "low": 0.3}
        return mapping.get(conf.lower() if conf else "low", 0.5)
    
    def evaluate(
        self,
        consensus: ConsensusResult,
        context: InspectionContext
    ) -> SafetyVerdict:
        """
        Evaluate ALL safety gates and return verdict.
        Now tracks every gate's pass/fail status.
        
        Args:
            consensus: Consensus result from VLMs
            context: Inspection context
        
        Returns:
            Final safety verdict with ALL gate results
        """
        self.logger.info("Evaluating ALL safety gates")
        
        all_gate_results: List[GateResult] = []
        triggered_gates: List[str] = []
        blocking_result: Optional[Tuple[str, str, str, bool]] = None  # verdict, reason, confidence, requires_human
        
        # Collect errors from both models
        error_messages = []
        if consensus.inspector_result.analysis_failed:
            error_messages.append(f"Inspector: {consensus.inspector_result.failure_reason or 'Analysis failed'}")
        if consensus.auditor_result.analysis_failed:
            error_messages.append(f"Auditor: {consensus.auditor_result.failure_reason or 'Analysis failed'}")
        
        # ====================================================================
        # GATE 0: Error State Check (evaluated FIRST)
        # ====================================================================
        has_errors = len(error_messages) > 0
        gate0_passed = not has_errors
        gate0_result = GateResult(
            gate_id=GATE_ERROR_STATE,
            passed=gate0_passed,
            message="No analysis errors" if gate0_passed else f"{len(error_messages)} analysis error(s)",
            details={"errors": error_messages} if error_messages else {}
        )
        all_gate_results.append(gate0_result)
        
        if not gate0_passed:
            triggered_gates.append(GATE_ERROR_STATE)
            error_summary = "; ".join(error_messages)
            blocking_result = (
                "UNSAFE",
                f"Analysis failed: {error_summary}",
                "low",
                True  # Require human review for errors
            )
            self.logger.error(f"Gate 0 FAILED: Analysis errors detected: {error_summary}")
        
        # Filter defects: remove invalid bboxes and low-confidence defects (unless high criticality)
        valid_defects = []
        for defect in consensus.combined_defects:
            # Check bbox validity if present
            if defect.bbox:
                # Validate bbox is in valid percentage range (0-100)
                if (defect.bbox.x < 0 or defect.bbox.x > 100 or
                    defect.bbox.y < 0 or defect.bbox.y > 100 or
                    defect.bbox.width <= 0 or defect.bbox.width > 100 or
                    defect.bbox.height <= 0 or defect.bbox.height > 100):
                    logger.warning(f"Defect {defect.type} has invalid bbox coordinates - filtering out")
                    continue
                # Check bbox doesn't exceed image bounds
                if defect.bbox.x + defect.bbox.width > 100 or defect.bbox.y + defect.bbox.height > 100:
                    logger.warning(f"Defect {defect.type} bbox exceeds image bounds - filtering out")
                    continue
                # Check reasonableness (area between 0.1% and 50%)
                area_percent = (defect.bbox.width * defect.bbox.height) / 100.0
                if area_percent < 0.1 or area_percent > 50.0:
                    logger.warning(f"Defect {defect.type} bbox unreasonable size (area={area_percent:.2f}%) - filtering out")
                    continue
            
            # Filter low-confidence defects unless criticality is high (conservative)
            if defect.confidence == "low" and context.criticality != "high":
                logger.debug(f"Filtering low-confidence defect: {defect.type} (criticality={context.criticality})")
                continue
            
            valid_defects.append(defect)
        
        defects = valid_defects
        defect_count = len(defects)
        
        # Categorize defects by severity
        critical_defects = [d for d in defects if d.safety_impact == "CRITICAL"]
        moderate_defects = [d for d in defects if d.safety_impact == "MODERATE"]
        cosmetic_defects = [d for d in defects if d.safety_impact == "COSMETIC"]
        
        critical_count = len(critical_defects)
        moderate_count = len(moderate_defects)
        cosmetic_count = len(cosmetic_defects)
        
        inspector_conf = consensus.inspector_result.overall_confidence
        auditor_conf = consensus.auditor_result.overall_confidence
        auditor_condition = consensus.auditor_result.overall_condition
        
        domain_rules = self._get_domain_rules(context.domain)
        
        # ====================================================================
        # GATE 1: Critical Defect Check
        # ====================================================================
        gate1_passed = critical_count == 0
        gate1_result = GateResult(
            gate_id=GATE_CRITICAL_DEFECT,
            passed=gate1_passed,
            message=f"{'No' if gate1_passed else critical_count} critical defects",
            details={"critical_count": critical_count, "types": [d.type for d in critical_defects]}
        )
        all_gate_results.append(gate1_result)
        
        if not gate1_passed and blocking_result is None:
            triggered_gates.append(GATE_CRITICAL_DEFECT)
            blocking_result = (
                "UNSAFE",
                f"Agent detected {critical_count} critical safety defect(s): "
                f"{', '.join(d.type for d in critical_defects)}",
                "high" if consensus.models_agree else "medium",
                False
            )
            self.logger.warning(f"Gate 1 FAILED: {critical_count} critical defects")
        
        # ====================================================================
        # GATE 2: Domain-Specific Zero Tolerance
        # ====================================================================
        flagged_defects = [
            d for d in defects 
            if self._should_flag_for_domain(d.type, context.domain)
        ]
        gate2_passed = not (flagged_defects and domain_rules.get("require_human_review_always", False))
        gate2_result = GateResult(
            gate_id=GATE_DOMAIN_ZERO_TOLERANCE,
            passed=gate2_passed,
            message=f"{'Passed' if gate2_passed else f'{len(flagged_defects)} domain violations'}",
            details={"domain": context.domain, "flagged": [d.type for d in flagged_defects]}
        )
        all_gate_results.append(gate2_result)
        
        if not gate2_passed and blocking_result is None:
            triggered_gates.append(GATE_DOMAIN_ZERO_TOLERANCE)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"Domain '{context.domain}' requires review for: "
                f"{', '.join(d.type for d in flagged_defects)}",
                "medium",
                True
            )
            self.logger.warning(f"Gate 2 FAILED: Domain flags triggered")
        
        # ====================================================================
        # GATE 3: VLM Agreement Check
        # ====================================================================
        gate3_passed = consensus.models_agree
        gate3_result = GateResult(
            gate_id=GATE_MODEL_DISAGREEMENT,
            passed=gate3_passed,
            message=f"Agreement: {consensus.agreement_score:.0%}",
            details={"agreement_score": consensus.agreement_score, "models_agree": consensus.models_agree}
        )
        all_gate_results.append(gate3_result)
        
        if not gate3_passed and blocking_result is None:
            triggered_gates.append(GATE_MODEL_DISAGREEMENT)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"Inspector and Auditor disagree. {consensus.disagreement_details}",
                "low",
                True
            )
            self.logger.warning(f"Gate 3 FAILED: Models disagree ({consensus.agreement_score:.0%})")
        
        # ====================================================================
        # GATE 4: Confidence Threshold Check
        # ====================================================================
        low_confidence = inspector_conf == "low" or auditor_conf == "low"
        gate4_passed = not low_confidence
        gate4_result = GateResult(
            gate_id=GATE_LOW_CONFIDENCE,
            passed=gate4_passed,
            message=f"Inspector: {inspector_conf}, Auditor: {auditor_conf}",
            details={"inspector_confidence": inspector_conf, "auditor_confidence": auditor_conf}
        )
        all_gate_results.append(gate4_result)
        
        if not gate4_passed and blocking_result is None:
            triggered_gates.append(GATE_LOW_CONFIDENCE)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"Low confidence in analysis (Inspector: {inspector_conf}, Auditor: {auditor_conf})",
                "low",
                True
            )
            self.logger.warning("Gate 4 FAILED: Low confidence")
        
        # ====================================================================
        # GATE 5: Defect Count Threshold
        # ====================================================================
        gate5_passed = defect_count <= config.max_defects_auto
        gate5_result = GateResult(
            gate_id=GATE_DEFECT_COUNT,
            passed=gate5_passed,
            message=f"{defect_count} defects (limit: {config.max_defects_auto})",
            details={"defect_count": defect_count, "limit": config.max_defects_auto}
        )
        all_gate_results.append(gate5_result)
        
        if not gate5_passed and blocking_result is None:
            triggered_gates.append(GATE_DEFECT_COUNT)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"Multiple defects detected ({defect_count} found, limit: {config.max_defects_auto})",
                "medium",
                True
            )
            self.logger.warning(f"Gate 5 FAILED: Too many defects ({defect_count})")
        
        # ====================================================================
        # GATE 6: High Criticality Context
        # ====================================================================
        # High criticality + zero defects requires BOTH models HIGH confidence
        high_crit_zero_defects = (
            context.criticality == "high" and
            defect_count == 0
        )
        high_crit_with_defects = (
            context.criticality == "high" and
            defect_count > 0 and
            config.high_criticality_requires_review
        )
        
        # For high criticality + zero defects, both must have HIGH confidence
        if high_crit_zero_defects:
            both_high_conf = (inspector_conf == "high" and auditor_conf == "high")
            if not both_high_conf:
                gate6_passed = False
                gate6_message = f"High criticality, no defects, but insufficient confidence (Inspector: {inspector_conf}, Auditor: {auditor_conf})"
            else:
                gate6_passed = True
                gate6_message = f"High criticality, no defects, both models HIGH confidence - verified"
        else:
            gate6_passed = not high_crit_with_defects
            gate6_message = f"Criticality: {context.criticality}, Defects: {defect_count}"
        
        gate6_result = GateResult(
            gate_id=GATE_HIGH_CRITICALITY,
            passed=gate6_passed,
            message=gate6_message,
            details={
                "criticality": context.criticality,
                "defect_count": defect_count,
                "inspector_confidence": inspector_conf,
                "auditor_confidence": auditor_conf
            }
        )
        all_gate_results.append(gate6_result)
        
        if not gate6_passed and blocking_result is None:
            triggered_gates.append(GATE_HIGH_CRITICALITY)
            if high_crit_zero_defects:
                blocking_result = (
                    "UNSAFE",
                    f"High-criticality component with zero defects but insufficient confidence "
                    f"(Inspector: {inspector_conf}, Auditor: {auditor_conf}) - conservative verdict",
                    "low",
                    True
                )
            else:
                blocking_result = (
                    "REQUIRES_HUMAN_REVIEW",
                    f"High-criticality component with {defect_count} defect(s) - human verification required",
                    "medium",
                    True
                )
            self.logger.warning("Gate 6 FAILED: High criticality requirement not met")
        
        # ====================================================================
        # GATE 7: Clean Image Verification Gate (No Defects Found)
        # ====================================================================
        # For "no defects" -> SAFE, require:
        # - defect_count == 0 (after filtering invalid/low-confidence defects)
        # - All reported defects (if any) have valid bboxes
        # - BOTH models HIGH confidence
        # - models_agree (agreement_score > 0.8)
        # - no errors (checked in GATE_0)
        no_defects = defect_count == 0
        
        # Additional validation: Check if any defects from original consensus have invalid bboxes
        # (This catches defects that were filtered but might indicate model hallucination)
        original_defects = consensus.combined_defects
        invalid_bbox_defects = []
        for defect in original_defects:
            if defect.bbox:
                # Check for invalid coordinates
                if (defect.bbox.x < 0 or defect.bbox.x > 100 or
                    defect.bbox.y < 0 or defect.bbox.y > 100 or
                    defect.bbox.width <= 0 or defect.bbox.width > 100 or
                    defect.bbox.height <= 0 or defect.bbox.height > 100 or
                    defect.bbox.x + defect.bbox.width > 100 or
                    defect.bbox.y + defect.bbox.height > 100):
                    invalid_bbox_defects.append(defect.type)
        
        has_invalid_bboxes = len(invalid_bbox_defects) > 0
        
        both_high_conf = (inspector_conf == "high" and auditor_conf == "high")
        high_agreement = consensus.agreement_score > 0.8
        no_errors = gate0_result.passed  # From GATE_0
        
        gate7_passed = (
            no_defects and
            not has_invalid_bboxes and  # No invalid bboxes indicate hallucinations
            both_high_conf and
            high_agreement and
            no_errors
        )
        
        gate7_details = {
            "defect_count": defect_count,
            "has_invalid_bboxes": has_invalid_bboxes,
            "invalid_bbox_defects": invalid_bbox_defects,
            "inspector_confidence": inspector_conf,
            "auditor_confidence": auditor_conf,
            "both_high_confidence": both_high_conf,
            "agreement_score": consensus.agreement_score,
            "high_agreement": high_agreement,
            "no_errors": no_errors
        }
        
        if no_defects and not gate7_passed:
            # Missing requirements for SAFE verdict
            missing = []
            if has_invalid_bboxes:
                missing.append(f"Invalid bbox coordinates detected: {', '.join(invalid_bbox_defects)}")
            if not both_high_conf:
                missing.append(f"Both models HIGH confidence (Inspector: {inspector_conf}, Auditor: {auditor_conf})")
            if not high_agreement:
                missing.append(f"High agreement (score: {consensus.agreement_score:.2f}, required: >0.8)")
            if not no_errors:
                missing.append("No analysis errors")
            gate7_message = f"No defects but missing requirements: {', '.join(missing)}"
        elif gate7_passed:
            gate7_message = "No defects, valid bboxes, both HIGH confidence, high agreement, no errors - verified clean"
        else:
            gate7_message = f"{defect_count} valid defects found"
        
        gate7_result = GateResult(
            gate_id=GATE_NO_DEFECTS,
            passed=gate7_passed,
            message=gate7_message,
            details=gate7_details
        )
        all_gate_results.append(gate7_result)
        
        # ====================================================================
        # GATE 8: Auditor Certainty Check (NEW)
        # ====================================================================
        auditor_uncertain = (
            auditor_condition == "uncertain" or
            auditor_conf == "low" or
            self._confidence_to_numeric(auditor_conf) < 0.4
        )
        gate8_passed = not auditor_uncertain
        gate8_result = GateResult(
            gate_id=GATE_AUDITOR_UNCERTAIN,
            passed=gate8_passed,
            message=f"Auditor condition: {auditor_condition}, confidence: {auditor_conf}",
            details={"auditor_condition": auditor_condition, "auditor_confidence": auditor_conf}
        )
        all_gate_results.append(gate8_result)
        
        if not gate8_passed and blocking_result is None:
            triggered_gates.append(GATE_AUDITOR_UNCERTAIN)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"Auditor analysis inconclusive (condition: {auditor_condition}, confidence: {auditor_conf})",
                "low",
                True
            )
            self.logger.warning("Gate 8 FAILED: Auditor uncertain")
        
        # ====================================================================
        # DETERMINE FINAL VERDICT
        # ====================================================================
        
        # If no blocking result and Gate 7 passed (verified clean) -> SAFE
        if blocking_result is None and gate7_result.passed:
            triggered_gates.append(GATE_NO_DEFECTS)
            self.logger.info("Gate 7 PASSED: Verified clean image -> SAFE")
            
            return SafetyVerdict(
                verdict="SAFE",
                reason="No defects detected by Inspector or Auditor - all safety gates passed with HIGH confidence verification",
                requires_human=False,
                confidence_level="high",
                triggered_gates=triggered_gates,
                errors=error_messages,  # Include any errors even if gate passed
                defect_summary={
                    "total_defects": 0,
                    "verification_passed": True,
                    "all_gate_results": [g.to_dict() for g in all_gate_results]
                }
            )
        
        # If blocking result exists, return it
        if blocking_result:
            verdict, reason, confidence, requires_human = blocking_result
            return SafetyVerdict(
                verdict=verdict,
                reason=reason,
                requires_human=requires_human,
                confidence_level=confidence,
                triggered_gates=triggered_gates,
                errors=error_messages,  # Include errors
                defect_summary={
                    "total_defects": defect_count,
                    "critical": critical_count,
                    "moderate": moderate_count,
                    "cosmetic": cosmetic_count,
                    "all_gate_results": [g.to_dict() for g in all_gate_results]
                }
            )
        
        # ====================================================================
        # DEFAULT: CONSERVATIVE APPROACH - Non-critical defects
        # ====================================================================
        # If only MODERATE or COSMETIC defects and all other gates pass,
        # we can consider it SAFE with monitoring recommendation
        # BUT: High criticality + cosmetic defects -> REQUIRES_HUMAN_REVIEW (not UNSAFE)
        if critical_count == 0 and moderate_count == 0 and cosmetic_count > 0:
            # Check if high criticality - if so, require human review (not UNSAFE)
            if context.criticality == "high":
                # High criticality + cosmetic defects -> REQUIRES_HUMAN_REVIEW (conservative but not UNSAFE)
                triggered_gates.append(GATE_DEFAULT_CONSERVATIVE)
                gate_default_result = GateResult(
                    gate_id=GATE_DEFAULT_CONSERVATIVE,
                    passed=False,
                    message=f"High criticality with {cosmetic_count} cosmetic defects - requires human review",
                    details={"criticality": context.criticality, "cosmetic_count": cosmetic_count}
                )
                all_gate_results.append(gate_default_result)
                self.logger.warning(f"High criticality + {cosmetic_count} cosmetic defects -> REQUIRES_HUMAN_REVIEW")
                
                return SafetyVerdict(
                    verdict="REQUIRES_HUMAN_REVIEW",
                    reason=f"High-criticality component with {cosmetic_count} cosmetic defect(s) - human verification recommended",
                    requires_human=True,
                    confidence_level="high" if consensus.models_agree else "medium",
                    triggered_gates=triggered_gates,
                    errors=error_messages,
                    defect_summary={
                        "total_defects": defect_count,
                        "cosmetic": cosmetic_count,
                        "all_gate_results": [g.to_dict() for g in all_gate_results]
                    }
                )
            
            # Low/medium criticality + cosmetic only -> SAFE with note
            triggered_gates.append(GATE_NO_DEFECTS)  # Using this as "safe" indicator
            self.logger.info(f"Only cosmetic defects ({cosmetic_count}) on {context.criticality} criticality -> SAFE")
            
            return SafetyVerdict(
                verdict="SAFE",
                reason=f"Only cosmetic defects detected ({cosmetic_count}). No safety impact.",
                requires_human=False,
                confidence_level="high" if consensus.models_agree else "medium",
                triggered_gates=triggered_gates,
                errors=error_messages,
                defect_summary={
                    "total_defects": defect_count,
                    "cosmetic": cosmetic_count,
                    "all_gate_results": [g.to_dict() for g in all_gate_results]
                }
            )
        
        # Moderate defects -> UNSAFE (conservative)
        triggered_gates.append(GATE_DEFAULT_CONSERVATIVE)
        gate_default_result = GateResult(
            gate_id=GATE_DEFAULT_CONSERVATIVE,
            passed=False,
            message=f"Conservative: {moderate_count} moderate, {cosmetic_count} cosmetic defects",
            details={"moderate": moderate_count, "cosmetic": cosmetic_count}
        )
        all_gate_results.append(gate_default_result)
        
        if moderate_count > 0:
            severity_msg = f"{moderate_count} MODERATE"
        else:
            severity_msg = f"{defect_count} unclassified"
        
        self.logger.warning(f"Default gate (CONSERVATIVE): {severity_msg} defects -> UNSAFE")
        
        return SafetyVerdict(
            verdict="UNSAFE",
            reason=(
                f"Defects detected: {severity_msg} defect(s). "
                f"Types: {', '.join(d.type for d in defects[:3])}{'...' if len(defects) > 3 else ''}"
            ),
            requires_human=False,
            confidence_level="high" if consensus.models_agree else "medium",
            triggered_gates=triggered_gates,
            errors=error_messages,  # Include errors
            defect_summary={
                "total_defects": defect_count,
                "moderate": moderate_count,
                "cosmetic": cosmetic_count,
                "defect_types": [d.type for d in defects],
                "all_gate_results": [g.to_dict() for g in all_gate_results]
            }
        )


def evaluate_safety(
    consensus: ConsensusResult,
    context: InspectionContext
) -> SafetyVerdict:
    """Evaluate safety using deterministic gates with agent trust."""
    engine = SafetyGateEngine()
    return engine.evaluate(consensus, context)
