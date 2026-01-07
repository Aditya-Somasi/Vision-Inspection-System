"""
VLM Inspector Agent using HuggingFace with Qwen2.5-VL.
Uses LangChain's ChatHuggingFace wrapper for vision analysis.
"""

import time
from pathlib import Path

from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext
from utils.config import config
from utils.prompts import INSPECTOR_PROMPT


class VLMInspectorAgent(BaseVLMAgent):
    """
    Primary inspection agent using HuggingFace with Qwen2.5-VL.
    Performs initial defect detection and safety assessment.
    """
    
    def __init__(self):
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(api_key=config.huggingface_api_key)
        self.model_id = config.vlm_inspector_model
        self.temperature = config.vlm_inspector_temperature
        self.max_tokens = config.vlm_inspector_max_tokens
        
        # We don't use LangChain for HuggingFace vision models
        # (langchain-huggingface doesn't support vision well yet)
        super().__init__(
            llm=None,  # We'll use InferenceClient directly
            nickname="Inspector",
            is_vision=True
        )
        
        self.logger.info(f"Initialized Inspector with HuggingFace model: {self.model_id}")
    
    def analyze(
        self,
        image_path: Path,
        context: InspectionContext
    ) -> VLMAnalysisResult:
        """
        Analyze image for defects using Qwen2.5-VL.
        
        Args:
            image_path: Path to image file
            context: Inspection context
        
        Returns:
            Structured analysis result
        """
        self.logger.info(f"Starting inspection for image: {context.image_id}")
        
        try:
            # Build prompt with context
            prompt = INSPECTOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "general",
                user_notes=context.user_notes or "None provided"
            )
            
            # Encode image to base64 for vision
            image_data = self._encode_image_to_base64(image_path)
            
            # Build messages for chat completions API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ]
            
            self.logger.debug("Calling HuggingFace API...")
            start_time = time.time()
            
            # Call HuggingFace InferenceClient
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = completion.choices[0].message.content
            
            elapsed = time.time() - start_time
            self.logger.info(f"HuggingFace response received in {elapsed:.2f}s")
            
            self.logger.debug(f"Raw response: {response_text[:300]}...")
            
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
            
            return VLMAnalysisResult(
                object_identified="unknown",
                overall_condition="uncertain",
                defects=[],
                overall_confidence="low",
                analysis_reasoning=f"Analysis failed due to error: {str(e)}"
            )
    
    def health_check(self) -> bool:
        """Perform health check on HuggingFace API."""
        try:
            self.logger.info(f"Health check: {self.model_id} (HuggingFace)")
            
            # Simple text-only test
            messages = [
                {"role": "user", "content": "Respond with only the word 'OK'"}
            ]
            
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=10
            )
            
            response = completion.choices[0].message.content
            
            if response and len(response) > 0:
                self.logger.info(f"✓ Inspector (HuggingFace) is healthy")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"✗ Inspector health check failed: {e}")
            return False


# Backward compatibility alias
InspectorAgent = VLMInspectorAgent
