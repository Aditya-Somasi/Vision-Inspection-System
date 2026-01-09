"""
VLM Auditor Agent using Groq with Llama 4 Maverick for independent verification.
Uses native Groq SDK with fallback to HuggingFace for resilience.
"""

import time
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any
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
            # Build prompt (Auditor works independently, no inspector findings in prompt per new design)
            prompt = AUDITOR_PROMPT.format(
                criticality=context.criticality,
                domain=context.domain or "general"
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
            
            # Use robust JSON parser like Inspector
            result_dict = self._parse_json_robust(response_text)
            
            # Validate and fix result (same as Inspector)
            result_dict = self._validate_and_fix_result(result_dict)
            
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
                analysis_reasoning=f"Audit verification failed: {str(e)}",
                analysis_failed=True,
                failure_reason=f"Auditor verification failed: {str(e)}"
            )
    
    def _parse_json_robust(self, text: str) -> Dict[str, Any]:
        """
        Robustly parse JSON from model response (same as Inspector).
        Handles markdown fences, nested braces, and malformed output.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed JSON dict
        """
        text = text.strip()
        
        # Try 1: Extract from markdown code fence
        fence_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        fence_matches = re.findall(fence_pattern, text)
        if fence_matches:
            for match in fence_matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # Try 2: Find balanced JSON object using stack-based parsing
        def find_balanced_json(s: str) -> Optional[str]:
            """Find the largest balanced JSON object in text."""
            best_json = None
            best_length = 0
            
            i = 0
            while i < len(s):
                if s[i] == '{':
                    depth = 0
                    start = i
                    in_string = False
                    escape_next = False
                    
                    for j in range(i, len(s)):
                        char = s[j]
                        
                        if escape_next:
                            escape_next = False
                            continue
                        
                        if char == '\\':
                            escape_next = True
                            continue
                        
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                        
                        if not in_string:
                            if char == '{':
                                depth += 1
                            elif char == '}':
                                depth -= 1
                                if depth == 0:
                                    candidate = s[start:j+1]
                                    if len(candidate) > best_length:
                                        try:
                                            json.loads(candidate)
                                            best_json = candidate
                                            best_length = len(candidate)
                                        except json.JSONDecodeError:
                                            pass
                                    break
                i += 1
            
            return best_json
        
        balanced_json = find_balanced_json(text)
        if balanced_json:
            try:
                return json.loads(balanced_json)
            except json.JSONDecodeError:
                pass
        
        # Try 3: Simple extraction (fallback)
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            try:
                return json.loads(text[start_idx:end_idx])
            except json.JSONDecodeError:
                pass
        
        # Failed to parse - log and raise
        self.logger.error(f"JSON parsing failed. Raw text (first 500 chars): {text[:500]}")
        raise ValueError(f"Failed to parse JSON from model response")
    
    def _validate_and_fix_result(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix common issues in model output (same as Inspector).
        
        Args:
            result_dict: Parsed JSON dict from model
            
        Returns:
            Validated and fixed dict
        """
        # Ensure required fields exist
        if "object_identified" not in result_dict:
            result_dict["object_identified"] = "unknown"
        
        if "overall_condition" not in result_dict:
            result_dict["overall_condition"] = "uncertain"
        
        if "overall_confidence" not in result_dict:
            result_dict["overall_confidence"] = "low"
        
        if "defects" not in result_dict:
            result_dict["defects"] = []
        
        # Confidence boosting: If no defects found and condition is "good", boost confidence
        defect_count = len(result_dict.get("defects", []))
        overall_condition = result_dict.get("overall_condition", "uncertain")
        current_confidence = result_dict.get("overall_confidence", "low")
        
        if defect_count == 0 and overall_condition == "good":
            # Boost confidence if model correctly identifies no defects
            if current_confidence == "low":
                result_dict["overall_confidence"] = "medium"
                self.logger.info("Boosted auditor confidence from 'low' to 'medium' for clean image")
            elif current_confidence == "medium":
                result_dict["overall_confidence"] = "high"
                self.logger.info("Boosted auditor confidence from 'medium' to 'high' for clean image")
        
        # Validate and fix defects
        valid_defects = []
        for defect in result_dict.get("defects", []):
            if isinstance(defect, dict):
                # Ensure required defect fields
                defect.setdefault("type", "unspecified")
                defect.setdefault("location", "unspecified")
                defect.setdefault("safety_impact", "MODERATE")
                defect.setdefault("reasoning", "No reasoning provided")
                defect.setdefault("confidence", "low")  # Conservative: default to low
                defect.setdefault("recommended_action", "Further inspection recommended")
                
                # Validate safety_impact
                if defect["safety_impact"] not in ["CRITICAL", "MODERATE", "COSMETIC"]:
                    defect["safety_impact"] = "MODERATE"
                
                # Validate confidence - default to "low" if invalid
                if defect["confidence"] not in ["high", "medium", "low"]:
                    defect["confidence"] = "low"
                
                # Additional validation: Filter out suspicious defects that may be false positives
                defect_type_lower = defect.get("type", "").lower()
                defect_confidence = defect.get("confidence", "low")
                defect_reasoning = defect.get("reasoning", "").lower()
                
                # Filter out low-confidence defects with vague reasoning
                vague_indicators = ["possible", "might be", "appears to be", "could be", "uncertain", "unclear"]
                has_vague_reasoning = any(indicator in defect_reasoning for indicator in vague_indicators)
                
                if defect_confidence == "low" and has_vague_reasoning:
                    self.logger.warning(
                        f"Filtering out auditor low-confidence defect with vague reasoning: {defect.get('type')} - "
                        f"'{defect.get('reasoning', '')[:50]}'"
                    )
                    continue  # Skip this defect
                
                # Validate bbox if present (same logic as Inspector)
                if "bbox" in defect and defect["bbox"]:
                    bbox = defect["bbox"]
                    if isinstance(bbox, dict):
                        required_keys = ["x", "y", "width", "height"]
                        if all(k in bbox for k in required_keys):
                            raw_x = bbox.get("x", 0)
                            raw_y = bbox.get("y", 0)
                            raw_w = bbox.get("width", 0)
                            raw_h = bbox.get("height", 0)
                            
                            has_large_values = any(v > 100 for v in [raw_x, raw_y, raw_w, raw_h] if v > 0)
                            
                            if has_large_values:
                                self.logger.warning(f"Bbox values > 100 detected, assuming pixel format: {bbox}")
                                defect["bbox"] = None
                                defect["bbox_approximate"] = True
                                self.logger.warning("Cannot reliably convert pixel coordinates - bbox removed")
                            else:
                                # Validate percentage range (0-100)
                                if (raw_x < 0 or raw_x > 100 or raw_y < 0 or raw_y > 100 or
                                    raw_w <= 0 or raw_w > 100 or raw_h <= 0 or raw_h > 100):
                                    self.logger.warning(f"Bbox values out of valid percentage range: {bbox}")
                                    defect["bbox"] = None
                                    defect["bbox_approximate"] = True
                                elif raw_x + raw_w > 100 or raw_y + raw_h > 100:
                                    self.logger.warning(f"Bbox exceeds image bounds: {bbox}")
                                    defect["bbox"] = None
                                    defect["bbox_approximate"] = True
                                else:
                                    # Validate reasonableness - reduced minimum to 0.05% to include smaller defects
                                    area_percent = (raw_w * raw_h) / 100.0
                                    if area_percent < 0.05:
                                        self.logger.warning(f"Bbox very small (area={area_percent:.2f}%) - may be noise: {bbox}")
                                        # Only filter out very low-confidence defects with extremely small bbox (< 0.02%)
                                        if defect_confidence == "low" and area_percent < 0.02:
                                            self.logger.warning(
                                                f"Filtering out auditor very low-confidence defect with extremely tiny bbox: {defect.get('type')}"
                                            )
                                            continue  # Skip this defect
                                        defect["bbox_approximate"] = True
                                    elif area_percent > 50.0:
                                        self.logger.warning(f"Bbox too large (area={area_percent:.2f}%) - likely error: {bbox}")
                                        defect["bbox"] = None
                                        defect["bbox_approximate"] = True
                                    else:
                                        defect["bbox"] = {
                                            "x": max(0, min(100, raw_x)),
                                            "y": max(0, min(100, raw_y)),
                                            "width": max(0.1, min(100, raw_w)),
                                            "height": max(0.1, min(100, raw_h))
                                        }
                        else:
                            defect["bbox"] = None
                    else:
                        defect["bbox"] = None
                
                # If defect has no bbox AND low confidence AND vague location, filter it out
                if not defect.get("bbox") and defect_confidence == "low":
                    location = defect.get("location", "").lower()
                    vague_locations = ["somewhere", "various", "multiple", "general", "areas"]
                    if any(vague in location for vague in vague_locations):
                        self.logger.warning(
                            f"Filtering out auditor low-confidence defect with no bbox and vague location: {defect.get('type')}"
                        )
                        continue  # Skip this defect
                
                valid_defects.append(defect)
        
        result_dict["defects"] = valid_defects
        
        return result_dict
    
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
                # For Groq, use a text-only model for health check since Groq vision models may not be available
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
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
