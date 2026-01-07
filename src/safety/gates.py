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
        
        defects = consensus.combined_defects
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
        high_crit_issue = (
            context.criticality == "high" and
            defect_count > 0 and
            config.high_criticality_requires_review
        )
        gate6_passed = not high_crit_issue
        gate6_result = GateResult(
            gate_id=GATE_HIGH_CRITICALITY,
            passed=gate6_passed,
            message=f"Criticality: {context.criticality}, Defects: {defect_count}",
            details={"criticality": context.criticality, "defect_count": defect_count}
        )
        all_gate_results.append(gate6_result)
        
        if not gate6_passed and blocking_result is None:
            triggered_gates.append(GATE_HIGH_CRITICALITY)
            blocking_result = (
                "REQUIRES_HUMAN_REVIEW",
                f"High-criticality component with {defect_count} defect(s) - human verification required",
                "medium",
                True
            )
            self.logger.warning("Gate 6 FAILED: High criticality with defects")
        
        # ====================================================================
        # GATE 7: No Defects Found (Verification Pass)
        # ====================================================================
        gate7_passed = defect_count == 0
        gate7_result = GateResult(
            gate_id=GATE_NO_DEFECTS,
            passed=gate7_passed,  # This is "passed" meaning condition met for SAFE
            message="No defects found" if gate7_passed else f"{defect_count} defects found",
            details={"defect_count": defect_count}
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
        
        # If no blocking result and no defects -> SAFE
        if blocking_result is None and defect_count == 0:
            triggered_gates.append(GATE_NO_DEFECTS)
            self.logger.info("All gates passed: No defects found -> SAFE")
            
            return SafetyVerdict(
                verdict="SAFE",
                reason="No defects detected by Inspector or Auditor - all safety gates passed",
                requires_human=False,
                confidence_level="high",
                triggered_gates=triggered_gates,
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
        if critical_count == 0 and moderate_count == 0 and cosmetic_count > 0:
            # Cosmetic only -> SAFE with note
            triggered_gates.append(GATE_NO_DEFECTS)  # Using this as "safe" indicator
            self.logger.info(f"Only cosmetic defects ({cosmetic_count}) -> SAFE")
            
            return SafetyVerdict(
                verdict="SAFE",
                reason=f"Only cosmetic defects detected ({cosmetic_count}). No safety impact.",
                requires_human=False,
                confidence_level="high" if consensus.models_agree else "medium",
                triggered_gates=triggered_gates,
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
