"""
VLM Auditor Agent using HuggingFace with Llama 3.2 Vision.
Uses HuggingFace InferenceClient for independent verification.
"""

import time
from pathlib import Path

from huggingface_hub import InferenceClient

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext
from utils.config import config
from utils.prompts import AUDITOR_PROMPT


class VLMAuditorAgent(BaseVLMAgent):
    """
    Auditor agent using HuggingFace with Llama 3.2 Vision.
    Performs independent verification of Inspector findings.
    """
    
    def __init__(self):
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(api_key=config.huggingface_api_key)
        self.model_id = config.vlm_auditor_model
        self.temperature = config.vlm_auditor_temperature
        self.max_tokens = config.vlm_auditor_max_tokens
        
        super().__init__(
            llm=None,  # Using InferenceClient directly
            nickname="Auditor",
            is_vision=True
        )
        
        self.logger.info(f"Initialized Auditor with HuggingFace model: {self.model_id}")
    
    def verify(
        self,
        image_path: Path,
        context: InspectionContext,
        inspector_result: VLMAnalysisResult
    ) -> VLMAnalysisResult:
        """
        Verify Inspector's findings with independent analysis.
        
        Args:
            image_path: Path to image file
            context: Inspection context
            inspector_result: Inspector's findings to verify
        
        Returns:
            Auditor's independent analysis
        """
        self.logger.info(f"Starting audit verification for: {context.image_id}")
        
        try:
            # Format inspector findings
            inspector_findings = self._format_inspector_findings(inspector_result)
            
            # Build prompt
            prompt = AUDITOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "general",
                inspector_findings=inspector_findings
            )
            
            # Encode image
            image_data = self._encode_image_to_base64(image_path)
            
            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ]
            
            self.logger.debug("Calling HuggingFace API for verification...")
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
            self.logger.info(f"Auditor response received in {elapsed:.2f}s")
            
            result_dict = self._parse_json_response(response_text)
            result = VLMAnalysisResult(**result_dict)
            
            self.logger.info(
                f"Audit complete: {len(result.defects)} defects found, "
                f"confidence: {result.overall_confidence}"
            )
            
            return result
        
        except Exception as e:
            self.logger.error(f"Auditor verification failed: {e}", exc_info=True)
            
            return VLMAnalysisResult(
                object_identified="unknown",
                overall_condition="uncertain",
                defects=[],
                overall_confidence="low",
                analysis_reasoning=f"Audit verification failed: {str(e)}"
            )
    
    def _format_inspector_findings(self, result: VLMAnalysisResult) -> str:
        """Format inspector findings for the auditor prompt."""
        lines = [
            f"Object: {result.object_identified}",
            f"Condition: {result.overall_condition}",
            f"Confidence: {result.overall_confidence}",
            f"Number of defects: {len(result.defects)}",
            ""
        ]
        
        if result.defects:
            lines.append("Defects found:")
            for i, defect in enumerate(result.defects, 1):
                lines.append(f"  {i}. {defect.type} ({defect.safety_impact})")
                lines.append(f"     Location: {defect.location}")
                lines.append(f"     Reasoning: {defect.reasoning}")
        else:
            lines.append("No defects were reported by the Inspector.")
        
        lines.append("")
        lines.append(f"Inspector's reasoning: {result.analysis_reasoning}")
        
        return "\n".join(lines)
    
    def health_check(self) -> bool:
        """Perform health check on HuggingFace API."""
        try:
            self.logger.info(f"Health check: {self.model_id} (HuggingFace)")
            
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
                self.logger.info(f"✓ Auditor (HuggingFace) is healthy")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"✗ Auditor health check failed: {e}")
            return False


# Backward compatibility
AuditorAgent = VLMAuditorAgent
