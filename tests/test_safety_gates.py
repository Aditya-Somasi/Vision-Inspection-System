"""
Unit tests for safety gate logic.
Uses gate constants from gates.py for consistency.
"""

import pytest
from src.schemas.models import (
    VLMAnalysisResult,
    DefectInfo,
    ConsensusResult,
    InspectionContext,
    SafetyVerdict
)
from src.safety.consensus import ConsensusAnalyzer, analyze_consensus
from src.safety.gates import (
    SafetyGateEngine, 
    evaluate_safety,
    # Import gate constants
    GATE_CRITICAL_DEFECT,
    GATE_MODEL_DISAGREEMENT,
    GATE_NO_DEFECTS,
    GATE_LOW_CONFIDENCE,
)


class TestConsensusAnalyzer:
    """Tests for ConsensusAnalyzer."""
    
    def test_models_agree_on_no_defects(self):
        """Test consensus when both models find no defects."""
        inspector_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        result = analyze_consensus(inspector_result, auditor_result)
        
        assert result.models_agree is True
        assert result.agreement_score >= 0.9
        assert len(result.combined_defects) == 0
    
    def test_models_disagree_on_condition(self):
        """Test consensus when models disagree on condition."""
        inspector_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="damaged",
            defects=[],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        result = analyze_consensus(inspector_result, auditor_result)
        
        assert result.models_agree is False
        assert "Condition" in result.disagreement_details
    
    def test_combined_defects(self, mock_defect):
        """Test that defects from both models are combined."""
        defect1 = DefectInfo(
            type="crack",
            location="corner",
            safety_impact="CRITICAL",
            reasoning="Test",
            confidence="high",
            recommended_action="Fix"
        )
        
        defect2 = DefectInfo(
            type="rust",
            location="center",
            safety_impact="MODERATE",
            reasoning="Test",
            confidence="medium",
            recommended_action="Monitor"
        )
        
        inspector_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="damaged",
            defects=[defect1],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="fastener",
            overall_condition="damaged",
            defects=[defect2],
            overall_confidence="high"
        )
        
        result = analyze_consensus(inspector_result, auditor_result)
        
        # Should combine both defects
        assert len(result.combined_defects) == 2
        defect_types = {d.type for d in result.combined_defects}
        assert "crack" in defect_types
        assert "rust" in defect_types


class TestSafetyGateEngine:
    """Tests for SafetyGateEngine."""
    
    def test_gate_1_critical_defect(self):
        """Test Gate 1: Critical defects trigger UNSAFE verdict."""
        critical_defect = DefectInfo(
            type="crack",
            location="structural beam",
            safety_impact="CRITICAL",
            reasoning="Structural integrity compromised",
            confidence="high",
            recommended_action="Replace immediately"
        )
        
        inspector_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="damaged",
            defects=[critical_defect],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="damaged",
            defects=[critical_defect],
            overall_confidence="high"
        )
        
        consensus = ConsensusResult(
            models_agree=True,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=1.0
        )
        
        context = InspectionContext(
            image_id="test-123",
            criticality="high"
        )
        
        verdict = evaluate_safety(consensus, context)
        
        assert verdict.verdict == "UNSAFE"
        assert GATE_CRITICAL_DEFECT in verdict.triggered_gates
    
    def test_gate_3_model_disagreement(self):
        """Test Gate 3: Model disagreement triggers review."""
        inspector_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="damaged",
            defects=[],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        consensus = ConsensusResult(
            models_agree=False,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=0.4,
            disagreement_details="Condition disagreement"
        )
        
        context = InspectionContext(image_id="test-123")
        
        verdict = evaluate_safety(consensus, context)
        
        assert verdict.verdict == "REQUIRES_HUMAN_REVIEW"
        assert GATE_MODEL_DISAGREEMENT in verdict.triggered_gates
        assert verdict.requires_human is True
    
    def test_gate_7_no_defects_safe(self):
        """Test Gate 7: No defects results in SAFE verdict."""
        inspector_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        consensus = ConsensusResult(
            models_agree=True,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=1.0
        )
        
        context = InspectionContext(image_id="test-123")
        
        verdict = evaluate_safety(consensus, context)
        
        assert verdict.verdict == "SAFE"
        assert GATE_NO_DEFECTS in verdict.triggered_gates
        assert verdict.requires_human is False
    
    def test_cosmetic_defects_safe(self):
        """Test that cosmetic defects result in SAFE verdict."""
        cosmetic_defect = DefectInfo(
            type="scratch",
            location="surface",
            safety_impact="COSMETIC",
            reasoning="Minor surface blemish",
            confidence="high",
            recommended_action="No action needed"
        )
        
        inspector_result = VLMAnalysisResult(
            object_identified="panel",
            overall_condition="good",
            defects=[cosmetic_defect],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="panel",
            overall_condition="good",
            defects=[cosmetic_defect],
            overall_confidence="high"
        )
        
        consensus = ConsensusResult(
            models_agree=True,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=0.95
        )
        
        context = InspectionContext(image_id="test-123")
        
        verdict = evaluate_safety(consensus, context)
        
        # Cosmetic-only defects should be SAFE
        assert verdict.verdict == "SAFE"
    
    def test_all_gate_results_included(self):
        """Test that all gate results are included in defect_summary."""
        inspector_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        auditor_result = VLMAnalysisResult(
            object_identified="beam",
            overall_condition="good",
            defects=[],
            overall_confidence="high"
        )
        
        consensus = ConsensusResult(
            models_agree=True,
            inspector_result=inspector_result,
            auditor_result=auditor_result,
            agreement_score=1.0
        )
        
        context = InspectionContext(image_id="test-123")
        
        verdict = evaluate_safety(consensus, context)
        
        # Check that all gate results are in defect_summary
        assert "all_gate_results" in verdict.defect_summary
        gate_results = verdict.defect_summary["all_gate_results"]
        assert len(gate_results) >= 7  # At least 7 gates evaluated


class TestPydanticSchemas:
    """Tests for Pydantic schema validation."""
    
    def test_defect_info_normalization(self):
        """Test that defect type is normalized."""
        defect = DefectInfo(
            type="  CRACK  ",
            location="corner",
            safety_impact="CRITICAL",
            reasoning="Test",
            confidence="high",
            recommended_action="Fix"
        )
        
        assert defect.type == "crack"
    
    def test_vlm_result_critical_count(self):
        """Test critical defect counting."""
        critical = DefectInfo(
            type="crack",
            location="a",
            safety_impact="CRITICAL",
            reasoning="x",
            confidence="high",
            recommended_action="x"
        )
        
        moderate = DefectInfo(
            type="rust",
            location="b",
            safety_impact="MODERATE",
            reasoning="y",
            confidence="medium",
            recommended_action="y"
        )
        
        result = VLMAnalysisResult(
            object_identified="test",
            overall_condition="damaged",
            defects=[critical, moderate, critical],
            overall_confidence="high"
        )
        
        assert result.critical_defect_count == 2
        assert len(result.defect_types) == 2
        assert "crack" in result.defect_types
        assert "rust" in result.defect_types
