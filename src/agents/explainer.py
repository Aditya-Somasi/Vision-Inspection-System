"""
Explainer Agent using Llama 3.1 for natural language explanations.
"""

import json
import time
import requests

from src.schemas.models import VLMAnalysisResult
from utils.config import config
from utils.logger import setup_logger
from utils.prompts import EXPLAINER_PROMPT


class ExplainerAgent:
    """
    Text-based LLM for generating human-readable explanations.
    Takes structured findings and produces natural language reports.
    """
    
    def __init__(self):
        self.model_id = config.explainer_model
        self.temperature = config.explainer_temperature
        self.max_tokens = config.explainer_max_tokens
        self.api_endpoint = f"{config.huggingface_api_endpoint}/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {config.huggingface_api_key}"}
        self.logger = setup_logger(
            "agent.explainer",
            level=config.log_level,
            component="EXPLAINER"
        )
    
    def _call_api(self, payload: dict, retries: int = None) -> dict:
        """Call HuggingFace API with retry logic."""
        retries = retries or config.api_max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                self.logger.debug(f"API call attempt {attempt + 1}/{retries}")
                
                response = requests.post(
                    self.api_endpoint,
                    headers=self.headers,
                    json=payload,
                    timeout=config.api_timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    wait_time = config.api_retry_backoff ** attempt
                    self.logger.warning(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    last_error = Exception(error_msg)
            
            except Exception as e:
                self.logger.error(f"API call error: {e}")
                last_error = e
                time.sleep(config.api_retry_backoff ** attempt)
        
        raise last_error or Exception("API call failed")
    
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
            
            # Build prompt
            prompt = EXPLAINER_PROMPT.format(
                findings=json.dumps(findings, indent=2)
            )
            
            # Prepare payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False
                }
            }
            
            self.logger.debug("Calling LLM API...")
            
            # Call API
            response = self._call_api(payload)
            
            # Extract text
            if isinstance(response, list) and len(response) > 0:
                explanation = response[0].get("generated_text", "")
            elif isinstance(response, dict):
                explanation = response.get("generated_text", "")
            else:
                explanation = (
                    "Unable to generate explanation due to unexpected response format."
                )
            
            self.logger.info("Explanation generated successfully")
            
            return explanation.strip()
        
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            
            # Fallback explanation
            return (
                f"Inspection complete. The system detected "
                f"{len(inspector_result.defects)} defects. "
                f"Final verdict: {safety_verdict.get('verdict', 'UNKNOWN')}. "
                f"Please review the detailed findings in the report."
            )
    
    def health_check(self) -> bool:
        """Perform health check."""
        try:
            self.logger.info(f"Health check: {self.model_id}")
            
            payload = {
                "inputs": "test",
                "parameters": {"max_new_tokens": 10}
            }
            
            response = self._call_api(payload, retries=1)
            
            if response:
                self.logger.info("✓ Explainer is healthy")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"✗ Explainer health check failed: {e}")
            return False
