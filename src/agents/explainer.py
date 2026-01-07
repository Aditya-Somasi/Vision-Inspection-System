"""
Explainer Agent using Groq for fast text generation.
Uses LangChain's ChatGroq wrapper for natural language explanations.
"""

import json

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from src.schemas.models import VLMAnalysisResult
from utils.config import config
from utils.logger import setup_logger
from utils.prompts import EXPLAINER_PROMPT


class ExplainerAgent:
    """
    Text-based LLM for generating human-readable explanations.
    Uses Groq for ultra-fast inference.
    """
    
    def __init__(self):
        # Initialize Groq LLM via LangChain
        self.llm = ChatGroq(
            model=config.explainer_model,
            api_key=config.groq_api_key,
            temperature=config.explainer_temperature,
            max_tokens=config.explainer_max_tokens
        )
        
        self.logger = setup_logger(
            "agent.explainer",
            level=config.log_level,
            component="EXPLAINER"
        )
        
        self.logger.info(f"Initialized Explainer with Groq model: {config.explainer_model}")
    
    def generate_explanation(
        self,
        inspector_result: VLMAnalysisResult,
        auditor_result: VLMAnalysisResult,
        consensus: dict,
        safety_verdict: dict
    ) -> str:
        """
        Generate human-readable explanation of findings.
        
        Args:
            inspector_result: Inspector analysis
            auditor_result: Auditor analysis
            consensus: Consensus result
            safety_verdict: Final safety verdict
        
        Returns:
            Natural language explanation
        """
        self.logger.info("Generating explanation...")
        
        try:
            # Build structured findings for LLM
            findings = {
                "inspector": {
                    "object": inspector_result.object_identified,
                    "condition": inspector_result.overall_condition,
                    "defects": [
                        {
                            "type": d.type,
                            "location": d.location,
                            "safety_impact": d.safety_impact,
                            "reasoning": d.reasoning
                        }
                        for d in inspector_result.defects
                    ],
                    "confidence": inspector_result.overall_confidence
                },
                "auditor": {
                    "object": auditor_result.object_identified,
                    "condition": auditor_result.overall_condition,
                    "defects": [
                        {
                            "type": d.type,
                            "location": d.location,
                            "safety_impact": d.safety_impact
                        }
                        for d in auditor_result.defects
                    ],
                    "confidence": auditor_result.overall_confidence
                },
                "consensus": consensus,
                "verdict": safety_verdict
            }
            
            # Build prompt with datetime-safe serialization
            def serialize_safe(obj):
                """Convert to JSON-serializable format."""
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                if hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            prompt = EXPLAINER_PROMPT.format(
                findings=json.dumps(findings, indent=2, default=serialize_safe)
            )
            
            self.logger.debug("Calling Groq API...")
            
            # Call LLM via LangChain
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            explanation = response.content
            
            self.logger.info("Explanation generated successfully")
            
            return explanation.strip()
        
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            
            return (
                f"Inspection complete. The system detected "
                f"{len(inspector_result.defects)} defects. "
                f"Final verdict: {safety_verdict.get('verdict', 'UNKNOWN')}. "
                f"Please review the detailed findings in the report."
            )
    
    def health_check(self) -> bool:
        """Perform health check on Groq API."""
        try:
            self.logger.info(f"Health check: {config.explainer_model} (Groq)")
            
            message = HumanMessage(content="Respond with only the word 'OK'")
            response = self.llm.invoke([message])
            
            if response.content and len(response.content) > 0:
                self.logger.info("✓ Explainer (Groq) is healthy")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"✗ Explainer health check failed: {e}")
            return False
