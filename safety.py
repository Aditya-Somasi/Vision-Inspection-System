"""
Safety gates and consensus analysis.
Implements deterministic safety rules and VLM agreement checking.
"""

from typing import List, Tuple
from models import (
    VLMAnalysisResult,
    ConsensusResult,
    SafetyVerdict,
    InspectionContext,
    DefectInfo
)
from logger import setup_logger
from config import config

logger = setup_logger(__name__, level=config.log_level, component="SAFETY")


# ============================================================================
# CONSENSUS ANALYZER
# ============================================================================

class ConsensusAnalyzer:
    """Analyzes agreement between Inspector and Auditor."""
    
    def __init__(self):
        self.logger = logger
    
    def analyze(
        self,
        inspector_result: VLMAnalysisResult,
        auditor_result: VLMAnalysisResult
    ) -> ConsensusResult:
        """
        Analyze consensus between two VLM results.
        
        Args:
            inspector_result: Inspector's analysis
            auditor_result: Auditor's analysis
        
        Returns:
            Consensus result with agreement metrics
        """
        self.logger.info("Analyzing consensus between Inspector and Auditor")
        
        # Compare overall conditions
        conditions_agree = (
            inspector_result.overall_condition == auditor_result.overall_condition
        )
        
        # Compare defect counts
        inspector_defect_count = len(inspector_result.defects)
        auditor_defect_count = len(auditor_result.defects)
        
        # Compare defect types
        inspector_types = set(inspector_result.defect_types)
        auditor_types = set(auditor_result.defect_types)
        
        common_types = inspector_types & auditor_types
        all_types = inspector_types | auditor_types
        
        # Calculate agreement score
        type_agreement = len(common_types) / len(all_types) if all_types else 1.0
        
        # Count agreement (allow ±1 difference)
        count_diff = abs(inspector_defect_count - auditor_defect_count)
        count_agreement = 1.0 if count_diff <= 1 else max(0, 1 - (count_diff / max(inspector_defect_count, auditor_defect_count, 1)))
        
        # Confidence agreement
        confidence_levels = {"high": 3, "medium": 2, "low": 1}
        inspector_conf = confidence_levels.get(inspector_result.overall_confidence, 2)
        auditor_conf = confidence_levels.get(auditor_result.overall_confidence, 2)
        confidence_agreement = 1.0 - (abs(inspector_conf - auditor_conf) / 2)
        
        # Overall agreement score (weighted average)
        agreement_score = (
            0.4 * (1.0 if conditions_agree else 0.0) +
            0.3 * type_agreement +
            0.2 * count_agreement +
            0.1 * confidence_agreement
        )
        
        # Determine if models agree (threshold: 0.7)
        models_agree = agreement_score >= 0.7
        
        # Build disagreement details
        disagreement_details = None
        if not models_agree:
            details = []
            if not conditions_agree:
                details.append(
                    f"Condition: Inspector says '{inspector_result.overall_condition}', "
                    f"Auditor says '{auditor_result.overall_condition}'"
                )
            if inspector_defect_count != auditor_defect_count:
                details.append(
                    f"Count: Inspector found {inspector_defect_count} defects, "
                    f"Auditor found {auditor_defect_count}"
                )
            
            unique_to_inspector = inspector_types - auditor_types
            unique_to_auditor = auditor_types - inspector_types
            
            if unique_to_inspector:
                details.append(f"Inspector found: {', '.join(unique_to_inspector)}")
            if unique_to_auditor:
                details.append(f"Auditor found: {', '.join(unique_to_auditor)}")
            
            disagreement_details = "; ".join(details)
        
        self.logger.info(
            f"Consensus: {'AGREE' if models_agree else 'DISAGREE'} "
            f"(score: {agreement_score:.2f})"
        )
        
        if disagreement_details:
            self.logger.warning(f"Disagreement details: {disagreement_details}")
        
        return ConsensusResult(
            models_agree=models_agree,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=agreement_score,
            disagreement_details=disagreement_details
        )


# ============================================================================
# SAFETY GATE ENGINE
# ============================================================================

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
            self.logger.warning(f"Gate 1 triggered: {critical_count} critical defects found")
            
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


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_consensus(
    inspector_result: VLMAnalysisResult,
    auditor_result: VLMAnalysisResult
) -> ConsensusResult:
    """Analyze consensus between Inspector and Auditor."""
    analyzer = ConsensusAnalyzer()
    return analyzer.analyze(inspector_result, auditor_result)


def evaluate_safety(
    consensus: ConsensusResult,
    context: InspectionContext
) -> SafetyVerdict:
    """Evaluate safety using deterministic gates."""
    engine = SafetyGateEngine()
    return engine.evaluate(consensus, context)