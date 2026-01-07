"""
Safety gate engine for deterministic safety evaluation.
Trusts agent severity assessment with configurable domain-specific overrides.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

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


def load_safety_rules() -> Dict[str, Any]:
    """Load safety rules from YAML config."""
    try:
        if SAFETY_RULES_PATH.exists():
            with open(SAFETY_RULES_PATH, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Could not load safety_rules.yaml: {e}")
    return {}


class SafetyGateEngine:
    """
    Deterministic safety gate engine.
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
        
        # Check if defect type matches any zero-tolerance type
        defect_lower = defect_type.lower()
        for zt_type in zero_tolerance:
            if zt_type.lower() in defect_lower or defect_lower in zt_type.lower():
                return True
        return False
    
    def evaluate(
        self,
        consensus: ConsensusResult,
        context: InspectionContext
    ) -> SafetyVerdict:
        """
        Evaluate safety using deterministic gates.
        Trusts agent severity assessment for CRITICAL/MODERATE/COSMETIC.
        
        Args:
            consensus: Consensus result from VLMs
            context: Inspection context
        
        Returns:
            Final safety verdict
        """
        self.logger.info("Evaluating safety gates (agent-trust mode)")
        
        triggered_gates = []
        defects = consensus.combined_defects
        
        # Extract key metrics
        defect_count = len(defects)
        
        # TRUST AGENT'S SEVERITY CLASSIFICATION
        critical_defects = [d for d in defects if d.safety_impact == "CRITICAL"]
        moderate_defects = [d for d in defects if d.safety_impact == "MODERATE"]
        cosmetic_defects = [d for d in defects if d.safety_impact == "COSMETIC"]
        
        critical_count = len(critical_defects)
        moderate_count = len(moderate_defects)
        cosmetic_count = len(cosmetic_defects)
        
        inspector_conf = consensus.inspector_result.overall_confidence
        auditor_conf = consensus.auditor_result.overall_confidence
        
        domain_rules = self._get_domain_rules(context.domain)
        
        # ====================================================================
        # GATE 1: Agent-Determined Critical Defects
        # ====================================================================
        # Trust the agent's CRITICAL classification
        if critical_count > 0:
            triggered_gates.append("GATE_1_AGENT_CRITICAL")
            self.logger.warning(
                f"Gate 1 triggered: Agent classified {critical_count} defects as CRITICAL"
            )
            
            return SafetyVerdict(
                verdict="UNSAFE",
                reason=(
                    f"Agent detected {critical_count} critical safety defect(s): "
                    f"{', '.join(d.type for d in critical_defects)}"
                ),
                requires_human=False,
                confidence_level="high" if consensus.models_agree else "medium",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "critical": critical_count,
                    "moderate": moderate_count,
                    "cosmetic": cosmetic_count,
                    "critical_types": [d.type for d in critical_defects]
                }
            )
        
        # ====================================================================
        # GATE 2: Domain-Specific Zero Tolerance
        # ====================================================================
        # Check if any defects trigger domain-specific flags
        flagged_defects = [
            d for d in defects 
            if self._should_flag_for_domain(d.type, context.domain)
        ]
        
        if flagged_defects and domain_rules.get("require_human_review_always", False):
            triggered_gates.append("GATE_2_DOMAIN_FLAG")
            self.logger.warning(
                f"Gate 2 triggered: Domain-specific flag for {context.domain}"
            )
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=(
                    f"Domain '{context.domain}' requires review for: "
                    f"{', '.join(d.type for d in flagged_defects)}"
                ),
                requires_human=True,
                confidence_level="medium",
                triggered_gates=triggered_gates,
                defect_summary={
                    "total_defects": defect_count,
                    "flagged_for_domain": [d.type for d in flagged_defects]
                }
            )
        
        # ====================================================================
        # GATE 3: VLM Agreement Check
        # ====================================================================
        if not consensus.models_agree:
            triggered_gates.append("GATE_3_MODEL_DISAGREEMENT")
            self.logger.warning(
                f"Gate 3 triggered: Models disagree (score: {consensus.agreement_score:.2f})"
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
        # GATE 4: Confidence Threshold Check
        # ====================================================================
        low_confidence = (
            inspector_conf == "low" or
            auditor_conf == "low"
        )
        
        if low_confidence:
            triggered_gates.append("GATE_4_LOW_CONFIDENCE")
            self.logger.warning("Gate 4 triggered: Low confidence detected")
            
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
        # GATE 5: Defect Count Threshold
        # ====================================================================
        if defect_count > config.max_defects_auto:
            triggered_gates.append("GATE_5_DEFECT_COUNT")
            self.logger.warning(
                f"Gate 5 triggered: Too many defects ({defect_count} > {config.max_defects_auto})"
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
        # GATE 6: High Criticality Context
        # ====================================================================
        if (context.criticality == "high" and
            defect_count > 0 and
            config.high_criticality_requires_review):
            triggered_gates.append("GATE_6_HIGH_CRITICALITY")
            self.logger.warning(
                "Gate 6 triggered: High criticality component with defects"
            )
            
            return SafetyVerdict(
                verdict="REQUIRES_HUMAN_REVIEW",
                reason=(
                    f"High-criticality component with {defect_count} defect(s) - "
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
        # GATE 7: No Defects Found (Verification Pass)
        # ====================================================================
        if defect_count == 0:
            triggered_gates.append("GATE_7_NO_DEFECTS")
            self.logger.info("Gate 7: No defects found by both models")
            
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
        # DEFAULT: CONSERVATIVE APPROACH - Any Defect = UNSAFE
        # ====================================================================
        # EMERGENCY FIX: Until model accuracy is verified, treat ANY defect as UNSAFE
        # This prevents false negatives (marking faulty parts as SAFE)
        triggered_gates.append("GATE_DEFAULT_CONSERVATIVE")
        
        # Determine severity for messaging
        if moderate_count > 0:
            severity_msg = f"{moderate_count} MODERATE"
        elif cosmetic_count > 0:
            severity_msg = f"{cosmetic_count} COSMETIC"
        else:
            severity_msg = f"{defect_count} unclassified"
            
        self.logger.warning(f"Default gate (CONSERVATIVE): {severity_msg} defects - marking UNSAFE")
        
        return SafetyVerdict(
            verdict="UNSAFE",
            reason=(
                f"Defects detected: {severity_msg} defect(s). "
                f"Types: {', '.join(d.type for d in defects[:3])}{'...' if len(defects) > 3 else ''}"
            ),
            requires_human=False,  # Auto-mark unsafe, don't wait for human
            confidence_level="high" if consensus.models_agree else "medium",
            triggered_gates=triggered_gates,
            defect_summary={
                "total_defects": defect_count,
                "moderate": moderate_count,
                "cosmetic": cosmetic_count,
                "defect_types": [d.type for d in defects]
            }
        )


def evaluate_safety(
    consensus: ConsensusResult,
    context: InspectionContext
) -> SafetyVerdict:
    """Evaluate safety using deterministic gates with agent trust."""
    engine = SafetyGateEngine()
    return engine.evaluate(consensus, context)
