"""
Base VLM Agent class with retry logic and error handling.
"""

import base64
import json
import time
import requests
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger
from utils.config import config


class BaseVLMAgent:
    """Base class for VLM agents with retry logic and error handling."""
    
    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        nickname: str
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.nickname = nickname
        self.api_endpoint = f"{config.huggingface_api_endpoint}/{model_id}"
        self.headers = {"Authorization": f"Bearer {config.huggingface_api_key}"}
        self.logger = setup_logger(
            f"agent.{nickname}",
            level=config.log_level,
            component=nickname.upper()
        )
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _call_api(self, payload: dict, retries: Optional[int] = None) -> dict:
        """
        Call HuggingFace API with retry logic.
        
        Args:
            payload: Request payload
            retries: Number of retries (defaults to config)
        
        Returns:
            API response
        
        Raises:
            Exception if all retries fail
        """
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
                    self.logger.debug("API call successful")
                    return response.json()
                
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    wait_time = config.api_retry_backoff ** attempt
                    self.logger.warning(
                        f"Model loading, waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                    continue
                
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    last_error = Exception(error_msg)
            
            except requests.Timeout:
                wait_time = config.api_retry_backoff ** attempt
                self.logger.warning(f"Request timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
                last_error = Exception("Request timeout")
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                last_error = e
        
        # All retries failed
        self.logger.error(f"All {retries} API call attempts failed")
        raise last_error or Exception("API call failed after all retries")
    
    def _parse_json_response(self, text: str) -> dict:
        """
        Parse JSON from model response, handling markdown code blocks.
        
        Args:
            text: Raw response text
        
        Returns:
            Parsed JSON dict
        """
        # Remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.debug(f"Raw text: {text}")
            raise ValueError(f"Failed to parse JSON response: {e}")
    
    def health_check(self) -> bool:
        """
        Perform health check on the model API.
        
        Returns:
            True if model is healthy
        """
        try:
            self.logger.info(f"Health check: {self.model_id}")
            
            # Simple API call to check availability
            payload = {
                "inputs": "test",
                "parameters": {"max_new_tokens": 10}
            }
            
            response = self._call_api(payload, retries=1)
            
            if response:
                self.logger.info(f"✓ {self.nickname} is healthy")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"✗ {self.nickname} health check failed: {e}")
            return False
