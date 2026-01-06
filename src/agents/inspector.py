"""
Inspector Agent using Qwen2-VL for primary defect detection.
"""

import time
from pathlib import Path

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext
from utils.config import config
from utils.prompts import INSPECTOR_PROMPT


class InspectorAgent(BaseVLMAgent):
    """
    Primary inspection agent using Qwen2-VL.
    Performs initial defect detection and safety assessment.
    """
    
    def __init__(self):
        super().__init__(
            model_id=config.vlm_inspector_model,
            temperature=config.vlm_inspector_temperature,
            max_tokens=config.vlm_inspector_max_tokens,
            nickname="Inspector"
        )
    
    def analyze(
        self,
        image_path: Path,
        context: InspectionContext
    ) -> VLMAnalysisResult:
        """
        Analyze image for defects.
        
        Args:
            image_path: Path to image file
            context: Inspection context
        
        Returns:
            Structured analysis result
        """
        self.logger.info(f"Starting inspection for image: {context.image_id}")
        
        try:
            # Encode image
            image_b64 = self._encode_image(image_path)
            
            # Build prompt with context
            prompt = INSPECTOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "unknown",
                user_notes=context.user_notes or "None provided"
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
            
            self.logger.debug("Calling VLM API...")
            start_time = time.time()
            
            # Call API
            response = self._call_api(payload)
            
            elapsed = time.time() - start_time
            self.logger.info(f"VLM response received in {elapsed:.2f}s")
            
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
                f"Analysis complete: {len(result.defects)} defects found, "
                f"confidence: {result.overall_confidence}"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Inspector analysis failed: {e}", exc_info=True)
            
            # Return uncertain result on error
            return VLMAnalysisResult(
                object_identified="unknown",
                overall_condition="uncertain",
                defects=[],
                overall_confidence="low",
                analysis_reasoning=f"Analysis failed due to error: {str(e)}"
            )
