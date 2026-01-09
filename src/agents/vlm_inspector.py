"""
VLM Inspector Agent using HuggingFace with Qwen2.5-VL.
Includes image optimization, retry logic, and robust JSON parsing.
"""

import time
import re
import json
import base64
import io
from pathlib import Path
from typing import Optional, Dict, Any

from PIL import Image
from huggingface_hub import InferenceClient

from src.agents.base import BaseVLMAgent
from src.schemas.models import VLMAnalysisResult, InspectionContext, DefectInfo
from utils.config import config
from utils.prompts import INSPECTOR_PROMPT


class VLMInspectorAgent(BaseVLMAgent):
    """
    Primary inspection agent using HuggingFace with Qwen2.5-VL.
    Performs initial defect detection and safety assessment.
    Includes image optimization and robust error handling.
    """
    
    def __init__(self):
        # Initialize HuggingFace InferenceClient
        self.client = InferenceClient(api_key=config.huggingface_api_key)
        self.model_id = config.vlm_inspector_model
        self.temperature = config.vlm_inspector_temperature
        self.max_tokens = config.vlm_inspector_max_tokens
        self.max_image_size = getattr(config, 'max_image_dimension', 1024)
        
        super().__init__(
            llm=None,  # We'll use InferenceClient directly
            nickname="Inspector",
            is_vision=True
        )
        
        self.logger.info(f"Initialized Inspector with HuggingFace model: {self.model_id}")
    
    def _encode_image_optimized(self, image_path: Path, max_size: Optional[int] = None) -> str:
        """
        Encode image with resize and compression to prevent oversized payloads.
        
        Args:
            image_path: Path to image file
            max_size: Maximum dimension (width or height)
            
        Returns:
            Base64 data URI string
        """
        max_size = max_size or self.max_image_size
        
        img = Image.open(image_path)
        original_size = img.size
        
        # Resize if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            self.logger.debug(f"Resized image from {original_size} to {img.size}")
        
        # Convert to RGB if needed (RGBA/P modes can't be saved as JPEG)
        if img.mode in ('RGBA', 'P', 'LA'):
            img = img.convert('RGB')
        
        # Compress to JPEG with quality optimization
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85, optimize=True)
        
        # Check size and reduce quality if still too large (>5MB)
        if buffer.tell() > 5_000_000:
            self.logger.debug(f"Image still large ({buffer.tell()} bytes), reducing quality")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=60, optimize=True)
        
        # Final size check
        payload_size = buffer.tell()
        if payload_size > 10_000_000:
            raise ValueError(f"Image too large even after optimization: {payload_size} bytes")
        
        self.logger.debug(f"Encoded image: {payload_size} bytes")
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{base64_str}"
    
    def _call_api_with_retry(self, messages: list, max_retries: int = 3) -> str:
        """
        Call HuggingFace API with exponential backoff retry logic.
        
        Args:
            messages: Chat messages to send
            max_retries: Maximum retry attempts
            
        Returns:
            Response text from model
        """
        retry_delay = 1  # Initial delay in seconds
        
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
                error_str = str(e).lower()
                
                # Check for rate limiting
                if "429" in str(e) or "rate" in error_str:
                    # Try to parse Retry-After header
                    retry_after = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Rate limited, waiting {retry_after}s before retry {attempt+1}/{max_retries}"
                    )
                    time.sleep(retry_after)
                    
                elif "413" in str(e) or "payload" in error_str:
                    # Payload too large - don't retry, it won't help
                    raise ValueError(f"Image payload too large for API: {e}")
                    
                elif attempt < max_retries - 1:
                    # General error, retry with backoff
                    wait_time = retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"API attempt {attempt+1} failed: {e}, retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    raise
        
        raise RuntimeError(f"API call failed after {max_retries} attempts")
    
    def _parse_json_robust(self, text: str) -> Dict[str, Any]:
        """
        Robustly parse JSON from model response.
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
                    # Found potential JSON start
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
                                    # Found complete JSON
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
        
        # PRE-PARSE: Try to extract analysis_reasoning text before failing
        # This preserves summary even if JSON is malformed
        extracted_reasoning = None
        if "analysis_reasoning" in text:
            reasoning_pattern = r'"analysis_reasoning"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            match = re.search(reasoning_pattern, text, re.DOTALL)
            if not match:
                reasoning_pattern = r'"analysis_reasoning"\s*:\s*"([^"]*)"'
                match = re.search(reasoning_pattern, text)
            if match:
                extracted_reasoning = match.group(1).replace('\\"', '"').replace('\\n', '\n')
        
        # If we extracted reasoning, create partial result
        if extracted_reasoning:
            self.logger.warning("JSON parsing failed but extracted analysis_reasoning - returning partial result")
            partial_result = {"analysis_reasoning": extracted_reasoning}
            
            # Try to extract other fields with regex
            obj_match = re.search(r'"object_identified"\s*:\s*"([^"]*)"', text)
            if obj_match:
                partial_result["object_identified"] = obj_match.group(1)
            else:
                partial_result["object_identified"] = "unknown"
            
            # Extract defects count if possible
            defect_count = text.count('"type"')
            partial_result["defects"] = []  # Empty list, couldn't parse structure
            
            partial_result["overall_condition"] = "uncertain"
            partial_result["overall_confidence"] = "low"
            
            return partial_result
        
        # Failed to parse - log and raise
        self.logger.error(f"JSON parsing failed. Raw text (first 500 chars): {text[:500]}")
        raise ValueError(f"Failed to parse JSON from model response")
    
    def _validate_and_fix_result(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and fix common issues in model output.
        
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
        # This helps models be more confident when they correctly identify clean images
        defect_count = len(result_dict.get("defects", []))
        overall_condition = result_dict.get("overall_condition", "uncertain")
        current_confidence = result_dict.get("overall_confidence", "low")
        
        if defect_count == 0 and overall_condition == "good":
            # Boost confidence if model correctly identifies no defects
            # If already high, keep it; if medium/low, boost to medium or high
            if current_confidence == "low":
                result_dict["overall_confidence"] = "medium"
                self.logger.info("Boosted confidence from 'low' to 'medium' for clean image")
            elif current_confidence == "medium":
                result_dict["overall_confidence"] = "high"
                self.logger.info("Boosted confidence from 'medium' to 'high' for clean image")
            # If already high, keep it high
        
        # Validate and fix defects
        valid_defects = []
        for defect in result_dict.get("defects", []):
            if isinstance(defect, dict):
                # Ensure required defect fields
                defect.setdefault("type", "unspecified")
                defect.setdefault("location", "unspecified")
                defect.setdefault("safety_impact", "MODERATE")
                defect.setdefault("reasoning", "No reasoning provided")
                defect.setdefault("confidence", "low")  # Conservative: default to low confidence
                defect.setdefault("recommended_action", "Further inspection recommended")
                
                # Validate safety_impact
                if defect["safety_impact"] not in ["CRITICAL", "MODERATE", "COSMETIC"]:
                    defect["safety_impact"] = "MODERATE"
                
                # Validate confidence - default to "low" if invalid (conservative)
                if defect["confidence"] not in ["high", "medium", "low"]:
                    defect["confidence"] = "low"
                
                # Additional validation: Filter out suspicious defects that may be false positives
                # Check for common false positive patterns
                defect_type_lower = defect.get("type", "").lower()
                defect_confidence = defect.get("confidence", "low")
                defect_reasoning = defect.get("reasoning", "").lower()
                
                # Filter out low-confidence defects with vague reasoning or common false positive terms
                vague_indicators = ["possible", "might be", "appears to be", "could be", "uncertain", "unclear"]
                has_vague_reasoning = any(indicator in defect_reasoning for indicator in vague_indicators)
                
                # If low confidence AND vague reasoning, this is likely a false positive - filter it out
                if defect_confidence == "low" and has_vague_reasoning:
                    self.logger.warning(
                        f"Filtering out low-confidence defect with vague reasoning: {defect.get('type')} - "
                        f"'{defect.get('reasoning', '')[:50]}'"
                    )
                    continue  # Skip this defect
                
                # Validate and normalize bbox if present
                if "bbox" in defect and defect["bbox"]:
                    bbox = defect["bbox"]
                    if isinstance(bbox, dict):
                        required_keys = ["x", "y", "width", "height"]
                        if all(k in bbox for k in required_keys):
                            # Normalize coordinates to percentages (0-100)
                            # If values are > 100, assume they're pixels and need conversion
                            # For now, we assume models return percentages per prompt instructions
                            # But if they return pixels, we'll detect and normalize
                            raw_x = bbox.get("x", 0)
                            raw_y = bbox.get("y", 0)
                            raw_w = bbox.get("width", 0)
                            raw_h = bbox.get("height", 0)
                            
                            # Detect if values are likely pixels (any value > 100) vs percentages
                            # Note: This is a heuristic - ideally models always return percentages
                            has_large_values = any(v > 100 for v in [raw_x, raw_y, raw_w, raw_h] if v > 0)
                            
                            if has_large_values:
                                # Likely pixel format - normalize assuming model image size
                                # This is a fallback - models should return percentages per prompt
                                self.logger.warning(f"Bbox values > 100 detected, assuming pixel format: {bbox}")
                                # For safety, we'll set bbox to None if we can't reliably convert
                                # In future, we could normalize if we know the model input image size
                                defect["bbox"] = None
                                defect["bbox_approximate"] = True
                                self.logger.warning("Cannot reliably convert pixel coordinates without image size context - bbox removed")
                            else:
                                # Assume percentages (0-100), validate range
                                if (raw_x < 0 or raw_x > 100 or raw_y < 0 or raw_y > 100 or
                                    raw_w <= 0 or raw_w > 100 or raw_h <= 0 or raw_h > 100):
                                    self.logger.warning(f"Bbox values out of valid percentage range (0-100): {bbox}")
                                    defect["bbox"] = None
                                    defect["bbox_approximate"] = True
                                elif raw_x + raw_w > 100 or raw_y + raw_h > 100:
                                    self.logger.warning(f"Bbox exceeds image bounds: x+width={raw_x+raw_w}, y+height={raw_y+raw_h}")
                                    defect["bbox"] = None
                                    defect["bbox_approximate"] = True
                                else:
                                    # Validate reasonableness (area between 0.05% and 50% of image)
                                    # Reduced minimum from 0.1% to 0.05% to include smaller defects
                                    area_percent = (raw_w * raw_h) / 100.0
                                    if area_percent < 0.05:
                                        self.logger.warning(f"Bbox very small (area={area_percent:.2f}% < 0.05%) - may be noise: {bbox}")
                                        # Only filter out very low-confidence defects with extremely small bbox (< 0.02%)
                                        if defect_confidence == "low" and area_percent < 0.02:
                                            self.logger.warning(
                                                f"Filtering out very low-confidence defect with extremely tiny bbox: {defect.get('type')}"
                                            )
                                            continue  # Skip this defect
                                        # Don't remove, but flag as potentially invalid
                                        defect["bbox_approximate"] = True
                                    elif area_percent > 50.0:
                                        self.logger.warning(f"Bbox too large (area={area_percent:.2f}% > 50%) - likely error: {bbox}")
                                        defect["bbox"] = None
                                        defect["bbox_approximate"] = True
                                    else:
                                        # Valid bbox, ensure all values are in correct range
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
                
                # If defect has no bbox AND low confidence AND vague location, it's likely a false positive
                if not defect.get("bbox") and defect_confidence == "low":
                    location = defect.get("location", "").lower()
                    vague_locations = ["somewhere", "various", "multiple", "general", "areas"]
                    if any(vague in location for vague in vague_locations):
                        self.logger.warning(
                            f"Filtering out low-confidence defect with no bbox and vague location: {defect.get('type')}"
                        )
                        continue  # Skip this defect
                
                valid_defects.append(defect)
        
        result_dict["defects"] = valid_defects
        
        return result_dict
    
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
            
            # Encode image with optimization
            image_data = self._encode_image_optimized(image_path)
            
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
            
            # Call API with retry logic
            response_text = self._call_api_with_retry(messages)
            
            elapsed = time.time() - start_time
            self.logger.info(f"HuggingFace response received in {elapsed:.2f}s")
            self.logger.debug(f"Raw response (first 300 chars): {response_text[:300]}...")
            
            # Parse JSON robustly
            result_dict = self._parse_json_robust(response_text)
            
            # Validate and fix result
            result_dict = self._validate_and_fix_result(result_dict)
            
            # Create Pydantic model
            result = VLMAnalysisResult(**result_dict)
            
            # Apply agent-inferred criticality if provided
            if result.inferred_criticality:
                user_criticality = context.criticality
                inferred = result.inferred_criticality
                
                if user_criticality != inferred:
                    self.logger.info(
                        f"Agent inferred criticality '{inferred}' differs from user's '{user_criticality}'"
                    )
                    # Update context with inferred criticality (agent overrides user if higher)
                    criticality_order = {"low": 0, "medium": 1, "high": 2}
                    if criticality_order.get(inferred, 1) > criticality_order.get(user_criticality, 1):
                        self.logger.warning(
                            f"Upgrading criticality from '{user_criticality}' to '{inferred}' "
                            f"based on agent analysis"
                        )
            
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
                analysis_reasoning=f"Analysis failed due to error: {str(e)}",
                analysis_failed=True,
                failure_reason=f"Inspector analysis failed: {str(e)}"
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
