"""
Safety gate engine for deterministic safety evaluation.
"""

from src.schemas.models import (
    ConsensusResult,
    SafetyVerdict,
    InspectionContext,
)
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="SAFETY")


class SafetyGateEngine:
    """
    Deterministic safety gate engine.
    Applies universal thresholds without domain-specific hardcoded rules.
    """
    
    def __init__(self):
        self.logger = logger
    
    def evaluate(
        self,
        consensus: ConsensusResult,
        context: InspectionContext
    ) -> SafetyVerdict:
        """
        Evaluate safety using deterministic gates.
        
        Args:
            consensus: Consensus result from VLMs
            context: Inspection context
        
        Returns:
            Final safety verdict
        """
        self.logger.info("Evaluating safety gates")
        
        triggered_gates = []
        defects = consensus.combined_defects
        
        # Extract key metrics
        defect_count = len(defects)
        critical_defects = [d for d in defects if d.safety_impact == "CRITICAL"]
        critical_count = len(critical_defects)
        
        inspector_conf = consensus.inspector_result.overall_confidence
        auditor_conf = consensus.auditor_result.overall_confidence
        
        # ====================================================================
        # GATE 1: Critical Defect Detection
        # ====================================================================
        if critical_count > 0:
            triggered_gates.append("GATE_1_CRITICAL_DEFECT")
            self.logger.warning(
                f"Gate 1 triggered: {critical_count} critical defects found"
            )
            
            return SafetyVerdict(
                verdict="UNSAFE",
                reason=f"Critical safety defects detected ({critical_count} found)",
                requires_human=False,
                confidence_level="high" if consensus.models_agree else "medium",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "critical_defects": critical_count,
                    "critical_types": [d.type for d in critical_defects]
                }
            )
        
        # ====================================================================
        # GATE 2: VLM Agreement Check
        # ====================================================================
        if not consensus.models_agree:
            triggered_gates.append("GATE_2_MODEL_DISAGREEMENT")
            self.logger.warning(
                f"Gate 2 triggered: Models disagree (score: {consensus.agreement_score:.2f})"
            )
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=(
                    f"Inspector and Auditor disagree on findings. "
                    f"{consensus.disagreement_details}"
                ),
                requires_human=True,
                confidence_level="low",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "agreement_score": consensus.agreement_score
                }
            )
        
        # ====================================================================
        # GATE 3: Confidence Threshold Check
        # ====================================================================
        low_confidence = (
            inspector_conf == "low" or
            auditor_conf == "low"
        )
        
        if low_confidence:
            triggered_gates.append("GATE_3_LOW_CONFIDENCE")
            self.logger.warning("Gate 3 triggered: Low confidence detected")
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=(
                    f"Low confidence in analysis (Inspector: {inspector_conf}, "
                    f"Auditor: {auditor_conf})"
                ),
                requires_human=True,
                confidence_level="low",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "inspector_confidence": inspector_conf,
                    "auditor_confidence": auditor_conf
                }
            )
        
        # ====================================================================
        # GATE 4: Defect Count Threshold
        # ====================================================================
        if defect_count > config.max_defects_auto:
            triggered_gates.append("GATE_4_DEFECT_COUNT")
            self.logger.warning(
                f"Gate 4 triggered: Too many defects ({defect_count} > {config.max_defects_auto})"
            )
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=f"Multiple defects detected ({defect_count} found)",
                requires_human=True,
                confidence_level="medium",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "defect_types": [d.type for d in defects]
                }
            )
        
        # ====================================================================
        # GATE 5: High Criticality Context
        # ====================================================================
        if (context.criticality == "high" and
            defect_count > 0 and
            config.high_criticality_requires_review):
            triggered_gates.append("GATE_5_HIGH_CRITICALITY")
            self.logger.warning(
                "Gate 5 triggered: High criticality component with defects"
            )
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=(
                    f"High-criticality component with {defect_count} defects - "
                    f"human verification required"
                ),
                requires_human=True,
                confidence_level="medium",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "criticality": context.criticality,
                    "defect_types": [d.type for d in defects]
                }
            )
        
        # ====================================================================
        # GATE 6: No Defects Found (Verification Pass)
        # ====================================================================
        if defect_count == 0:
            triggered_gates.append("GATE_6_NO_DEFECTS")
            self.logger.info("Gate 6: No defects found by both models")
            
            # Extra verification for high criticality
            if context.criticality == "high":
                self.logger.info("High criticality - double-checked by Auditor")
            
            return SafetyVerdict(
                verdict="SAFE",
                reason="No defects detected by Inspector or Auditor",
                requires_human=False,
                confidence_level="high",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": 0,
                    "verification_passed": True
                }
            )
        
        # ====================================================================
        # DEFAULT: Safe with Notes (Minor Defects Only)
        # ====================================================================
        triggered_gates.append("GATE_DEFAULT_MINOR_DEFECTS")
        self.logger.info(f"Default gate: {defect_count} minor defects found")
        
        # Check if all defects are cosmetic
        all_cosmetic = all(d.safety_impact == "COSMETIC" for d in defects)
        
        return SafetyVerdict(
            verdict="SAFE" if all_cosmetic else "REQUIRES_HUMAN_REVIEW",
            reason=(
                f"{defect_count} minor defect(s) found - "
                f"{'cosmetic only' if all_cosmetic else 'moderate severity'}"
            ),
            requires_human=not all_cosmetic,
            confidence_level="high" if all_cosmetic else "medium",
            triggered_gates=triggered_gates,
            defect_summary={
                "total_defects": defect_count,
                "severity_breakdown": {
                    "critical": 0,
                    "moderate": sum(1 for d in defects if d.safety_impact == "MODERATE"),
                    "cosmetic": sum(1 for d in defects if d.safety_impact == "COSMETIC")
                }
            }
        )


def evaluate_safety(
    consensus: ConsensusResult,
    context: InspectionContext
) -> SafetyVerdict:
    """Evaluate safety using deterministic gates."""
    engine = SafetyGateEngine()
    return engine.evaluate(consensus, context)
