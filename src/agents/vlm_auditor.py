"""
VLM Auditor Agent using Groq with Llama 4 Maverick for independent verification.
Uses native Groq SDK with fallback to HuggingFace for resilience.
"""

import time
from pathlib import Path
from typing import Optional
import base64
import io

from PIL import Image

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext
from utils.config import config
from utils.prompts import AUDITOR_PROMPT


class VLMAuditorAgent(BaseVLMAgent):
    """
    Auditor agent using Groq's Llama 4 Maverick for independent verification.
    Uses DIFFERENT model from Inspector for TRUE consensus architecture.
    Falls back to HuggingFace if Groq unavailable.
    """
    
    def __init__(self):
        self.model_id = config.vlm_auditor_model
        self.temperature = config.vlm_auditor_temperature
        self.max_tokens = config.vlm_auditor_max_tokens
        self.provider = getattr(config, 'vlm_auditor_provider', 'groq')
        
        # Initialize appropriate client based on provider
        self._init_client()
        
        super().__init__(
            llm=None,  # Using direct client
            nickname="Auditor",
            is_vision=True
        )
        
        self.logger.info(
            f"Initialized Auditor with {self.provider.upper()} model: {self.model_id}"
        )
    
    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        self.use_groq = False
        self.use_huggingface = False
        
        if self.provider == "groq":
            try:
                from groq import Groq
                self.client = Groq(api_key=config.groq_api_key)
                self.use_groq = True
                return
            except ImportError:
                self.logger.warning("groq package not installed, trying langchain-groq")
                try:
                    from langchain_groq import ChatGroq
                    self.langchain_client = ChatGroq(
                        model=self.model_id,
                        api_key=config.groq_api_key,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    self.use_groq = True
                    return
                except Exception as e:
                    self.logger.warning(f"Groq init failed ({e}), falling back to HuggingFace")
            except Exception as e:
                self.logger.warning(f"Groq SDK failed ({e}), falling back to HuggingFace")
        
        # Fallback to HuggingFace
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=config.huggingface_api_key)
        self.use_huggingface = True
        # Use a different HF model for auditor if Groq not available
        if "llama-4" in self.model_id.lower():
            self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
            self.logger.info(f"Using HuggingFace fallback model: {self.model_id}")
    
    def _encode_image_optimized(self, image_path: Path, max_size: int = 1024) -> str:
        """Encode image with resize and compression to prevent oversized payloads."""
        img = Image.open(image_path)
        
        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image to {img.size}")
        
        # Convert to RGB if needed (for JPEG)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Compress to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        
        # Check size and reduce quality if needed
        if buffer.tell() > 5_000_000:  # 5MB limit
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=60, optimize=True)
        
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{base64_str}"
    
    def _call_groq_vision(self, prompt: str, image_data: str) -> str:
        """Call Groq vision API with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Groq API attempt {attempt+1} failed: {e}, retrying...")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
    
    def _call_huggingface_vision(self, prompt: str, image_data: str) -> str:
        """Call HuggingFace vision API with retry logic."""
        max_retries = 3
        retry_delay = 1
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        }]
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"HuggingFace API attempt {attempt+1} failed: {e}, retrying...")
                    time.sleep(retry_delay * (2 ** attempt))
                else:
                    raise
    
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
        self.logger.info(f"Using provider: {self.provider} with model: {self.model_id}")
        
        try:
            # Format inspector findings
            inspector_findings = self._format_inspector_findings(inspector_result)
            
            # Build prompt
            prompt = AUDITOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "general",
                inspector_findings=inspector_findings
            )
            
            # Encode image with optimization
            image_data = self._encode_image_optimized(image_path)
            
            self.logger.debug(f"Calling {self.provider.upper()} API for verification...")
            start_time = time.time()
            
            # Call appropriate API
            if self.use_groq:
                response_text = self._call_groq_vision(prompt, image_data)
            else:
                response_text = self._call_huggingface_vision(prompt, image_data)
            
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
        """Perform health check on the API."""
        try:
            provider_name = "Groq" if self.use_groq else "HuggingFace"
            self.logger.info(f"Health check: {self.model_id} ({provider_name})")
            
            if self.use_groq:
                response = self.client.chat.completions.create(
                    model=self.model_id.replace("-Instruct", "").replace("meta-llama/llama-4-maverick-17b-128e-instruct", "llama-3.3-70b-versatile"),  # Use text model for health check
                    messages=[{"role": "user", "content": "Respond with only the word 'OK'"}],
                    max_tokens=10
                )
                content = response.choices[0].message.content
            else:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": "Respond with only the word 'OK'"}],
                    max_tokens=10
                )
                content = completion.choices[0].message.content
            
            if content and len(content) > 0:
                self.logger.info(f"✓ Auditor ({provider_name}) is healthy")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"✗ Auditor health check failed: {e}")
            return False


# Backward compatibility
AuditorAgent = VLMAuditorAgent
