"""
Base VLM Agent class using LangChain for unified provider interface.
Provides common functionality for all VLM agents with LangChain integration.
"""

import base64
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from utils.logger import setup_logger
from utils.config import config


class BaseVLMAgent(ABC):
    """
    Base class for VLM agents using LangChain.
    Provides unified interface for different LLM providers.
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        nickname: str,
        is_vision: bool = True
    ):
        """
        Initialize the VLM agent.
        
        Args:
            llm: LangChain chat model instance
            nickname: Human-readable name for logging
            is_vision: Whether this is a vision model
        """
        self.llm = llm
        self.nickname = nickname
        self.is_vision = is_vision
        
        self.logger = setup_logger(
            f"agent.{nickname}",
            level=config.log_level,
            component=nickname.upper()
        )
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode local image file to base64 data URI.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 data URI string
        """
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        suffix = image_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        mime_type = mime_types.get(suffix, "image/jpeg")
        
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_str}"
    
    def _build_vision_message(
        self,
        prompt: str,
        image_source: Union[str, Path]
    ) -> HumanMessage:
        """
        Build a vision message with text and image.
        
        Args:
            prompt: Text prompt
            image_source: Either a URL string or Path to local file
            
        Returns:
            LangChain HumanMessage with multimodal content
        """
        # Determine if it's a URL or local file
        if isinstance(image_source, Path) or (
            isinstance(image_source, str) and not image_source.startswith("http")
        ):
            image_path = Path(image_source)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            image_url = self._encode_image_to_base64(image_path)
        else:
            image_url = image_source
        
        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ]
        )
    
    def _parse_json_response(self, text: str) -> dict:
        """
        Parse JSON from model response, handling markdown code blocks.
        
        Args:
            text: Raw response text
            
        Returns:
            Parsed JSON dict
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Try to find JSON in text
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1
        
        if start_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.debug(f"Raw text: {text[:500]}...")
            raise ValueError(f"Failed to parse JSON response: {e}")

    def _normalize_bbox_percentages(
        self,
        bbox: Optional[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> Optional[Dict[str, float]]:
        """
        Ensure bbox coordinates are expressed as percentages (0-100).

        Args:
            bbox: Raw bbox dict from model
            image_size: Tuple of (width, height) in pixels

        Returns:
            Normalized bbox dict or None if invalid
        """
        if not bbox:
            return None

        try:
            x = float(bbox.get("x"))
            y = float(bbox.get("y"))
            width = float(bbox.get("width"))
            height = float(bbox.get("height"))
        except (TypeError, ValueError):
            return None

        img_width, img_height = image_size
        if img_width <= 0 or img_height <= 0:
            return None

        def _clamp(value: float, minimum: float, maximum: float) -> float:
            return max(minimum, min(maximum, value))

        is_percentage = (
            0.0 <= x <= 100.0
            and 0.0 <= y <= 100.0
            and 0.0 < width <= 100.0
            and 0.0 < height <= 100.0
            and x + width <= 100.0 + 1e-3
            and y + height <= 100.0 + 1e-3
        )

        if not is_percentage:
            # Treat as pixel values and normalize
            x = (x / img_width) * 100.0
            y = (y / img_height) * 100.0
            width = (width / img_width) * 100.0
            height = (height / img_height) * 100.0

        x = _clamp(x, 0.0, 100.0)
        y = _clamp(y, 0.0, 100.0)
        width = _clamp(width, 0.01, 100.0)
        height = _clamp(height, 0.01, 100.0)

        if x + width > 100.0:
            width = 100.0 - x
        if y + height > 100.0:
            height = 100.0 - y

        if width <= 0 or height <= 0:
            return None

        return {
            "x": round(x, 4),
            "y": round(y, 4),
            "width": round(width, 4),
            "height": round(height, 4),
        }
    
    @abstractmethod
    def health_check(self) -> bool:
        """Perform health check on the model API."""
        pass
