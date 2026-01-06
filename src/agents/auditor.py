"""
Auditor Agent using Llama 3.2 Vision for verification.
"""

import json
import time
from pathlib import Path

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext
from utils.config import config
from utils.prompts import AUDITOR_PROMPT


class AuditorAgent(BaseVLMAgent):
    """
    Verification agent using Llama 3.2 Vision.
    Provides independent second opinion with skeptical approach.
    """
    
    def __init__(self):
        super().__init__(
            model_id=config.vlm_auditor_model,
            temperature=config.vlm_auditor_temperature,
            max_tokens=config.vlm_auditor_max_tokens,
            nickname="Auditor"
        )
    
    def verify(
        self,
        image_path: Path,
        context: InspectionContext,
        inspector_findings: VLMAnalysisResult
    ) -> VLMAnalysisResult:
        """
        Verify inspector findings with independent analysis.
        
        Args:
            image_path: Path to image file
            context: Inspection context
            inspector_findings: Results from inspector agent
        
        Returns:
            Independent analysis result
        """
        self.logger.info(f"Starting verification for image: {context.image_id}")
        
        try:
            # Encode image
            image_b64 = self._encode_image(image_path)
            
            # Summarize inspector findings for context
            inspector_summary = {
                "overall_condition": inspector_findings.overall_condition,
                "defect_count": len(inspector_findings.defects),
                "defect_types": inspector_findings.defect_types,
                "confidence": inspector_findings.overall_confidence
            }
            
            # Build prompt with inspector context
            prompt = AUDITOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "unknown",
                inspector_findings=json.dumps(inspector_summary, indent=2)
            )
            
            # Prepare payload
            payload = {
                "inputs": {
                    "text": prompt,
                    "image": image_b64
                },
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False
                }
            }
            
            self.logger.debug("Calling VLM API for verification...")
            start_time = time.time()
            
            # Call API
            response = self._call_api(payload)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Auditor response received in {elapsed:.2f}s")
            
            # Extract text from response
            if isinstance(response, list) and len(response) > 0:
                response_text = response[0].get("generated_text", "")
            elif isinstance(response, dict):
                response_text = response.get("generated_text", "")
            else:
                raise ValueError(f"Unexpected response format: {type(response)}")
            
            self.logger.debug(f"Raw response: {response_text[:200]}...")
            
            # Parse JSON
            result_dict = self._parse_json_response(response_text)
            
            # Validate and create Pydantic model
            result = VLMAnalysisResult(**result_dict)
            
            self.logger.info(
                f"Verification complete: {len(result.defects)} defects found, "
                f"confidence: {result.overall_confidence}"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Auditor verification failed: {e}", exc_info=True)
            
            # Return uncertain result on error
            return VLMAnalysisResult(
                object_identified="unknown",
                overall_condition="uncertain",
                defects=[],
                overall_confidence="low",
                analysis_reasoning=f"Verification failed due to error: {str(e)}"
            )
