"""
Consensus analysis between Inspector and Auditor.
"""

from src.schemas.models import VLMAnalysisResult, ConsensusResult
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="CONSENSUS")


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
        
        # Special case: Both report "no defects" - require HIGH confidence for true agreement
        # If both have 0 defects but LOW confidence, treat as disagreement (conservative)
        both_no_defects = (inspector_defect_count == 0 and auditor_defect_count == 0)
        inspector_high_conf = inspector_result.overall_confidence == "high"
        auditor_high_conf = auditor_result.overall_confidence == "high"
        
        if both_no_defects:
            # For "no defects" agreement, both must have HIGH confidence
            if not (inspector_high_conf and auditor_high_conf):
                # Low confidence "no defects" -> treat as disagreement (conservative)
                self.logger.warning(
                    f"Both models report 'no defects' but confidence is not HIGH for both "
                    f"(Inspector: {inspector_result.overall_confidence}, "
                    f"Auditor: {auditor_result.overall_confidence}) - treating as disagreement"
                )
                type_agreement = 0.0  # Force disagreement
                conditions_agree = False  # Override condition agreement
        
        # Count agreement (allow ±1 difference)
        count_diff = abs(inspector_defect_count - auditor_defect_count)
        count_agreement = (
            1.0 if count_diff <= 1
            else max(0, 1 - (count_diff / max(inspector_defect_count, auditor_defect_count, 1)))
        )
        
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
        
        # Round to 4 decimals to avoid floating point precision issues (e.g., 99.99999%)
        agreement_score = round(agreement_score, 4)
        
        # Clamp to 1.0 if very close
        if agreement_score >= 0.9999:
            agreement_score = 1.0
        
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


def analyze_consensus(
    inspector_result: VLMAnalysisResult,
    auditor_result: VLMAnalysisResult
) -> ConsensusResult:
    """Analyze consensus between Inspector and Auditor."""
    analyzer = ConsensusAnalyzer()
    return analyzer.analyze(inspector_result, auditor_result)
