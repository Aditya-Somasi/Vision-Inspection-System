"""
Pydantic schemas for data validation and VLM agent implementations.
Defines structured outputs and agent wrappers for Inspector, Auditor, and Explainer.
"""

from typing import List, Literal, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import base64
import json
import time
import requests
from pathlib import Path
from logger import setup_logger, set_request_id
from config import config
from prompts import INSPECTOR_PROMPT, AUDITOR_PROMPT, EXPLAINER_PROMPT

logger = setup_logger(__name__, level=config.log_level, component="MODELS")


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: float = Field(..., description="X coordinate (0-1 normalized or pixel)")
    y: float = Field(..., description="Y coordinate (0-1 normalized or pixel)")
    width: float = Field(..., description="Width")
    height: float = Field(..., description="Height")
    
    @field_validator("x", "y", "width", "height")
    @classmethod
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Coordinates must be non-negative")
        return v


class DefectInfo(BaseModel):
    """Structured defect information."""
    defect_id: str = Field(default_factory=lambda: f"defect_{int(time.time()*1000)}")
    type: str = Field(..., description="Defect type (e.g., crack, rust, deformation)")
    location: str = Field(..., description="Human-readable location description")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box if available")
    safety_impact: Literal["CRITICAL", "MODERATE", "COSMETIC"] = Field(
        ..., description="Safety impact level"
    )
    reasoning: str = Field(..., description="Why this defect is concerning")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Detection confidence")
    recommended_action: str = Field(..., description="Suggested action to take")
    
    @field_validator("type")
    @classmethod
    def normalize_defect_type(cls, v: str) -> str:
        """Normalize defect type to lowercase."""
        return v.lower().strip()
    
    def is_critical(self) -> bool:
        """Check if defect is critical."""
        return self.safety_impact == "CRITICAL"


class VLMAnalysisResult(BaseModel):
    """VLM analysis result with structured defects."""
    object_identified: str = Field(..., description="What object/component was identified")
    overall_condition: Literal["damaged", "good", "uncertain"] = Field(
        ..., description="Overall condition assessment"
    )
    defects: List[DefectInfo] = Field(default_factory=list, description="List of detected defects")
    overall_confidence: Literal["high", "medium", "low"] = Field(
        ..., description="Overall analysis confidence"
    )
    analysis_reasoning: Optional[str] = Field(None, description="General reasoning about the image")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def has_defects(self) -> bool:
        """Check if any defects were found."""
        return len(self.defects) > 0
    
    @property
    def critical_defect_count(self) -> int:
        """Count critical defects."""
        return sum(1 for d in self.defects if d.is_critical())
    
    @property
    def defect_types(self) -> List[str]:
        """Get list of unique defect types."""
        return list(set(d.type for d in self.defects))


class ConsensusResult(BaseModel):
    """Result of consensus analysis between two VLMs."""
    models_agree: bool = Field(..., description="Whether models agree on findings")
    inspector_result: VLMAnalysisResult
    auditor_result: VLMAnalysisResult
    agreement_score: float = Field(..., ge=0, le=1, description="Agreement score 0-1")
    disagreement_details: Optional[str] = Field(None, description="Details of disagreements")
    combined_defects: List[DefectInfo] = Field(default_factory=list)
    
    @model_validator(mode="after")
    def compute_combined_defects(self):
        """Combine defects from both models."""
        # Use inspector as primary, add unique defects from auditor
        inspector_types = set(d.type for d in self.inspector_result.defects)
        
        self.combined_defects = self.inspector_result.defects.copy()
        
        for defect in self.auditor_result.defects:
            if defect.type not in inspector_types:
                self.combined_defects.append(defect)
        
        return self


class SafetyVerdict(BaseModel):
    """Final safety verdict after all checks."""
    verdict: Literal["SAFE", "UNSAFE", "REQUIRES_HUMAN_REVIEW"] = Field(
        ..., description="Final safety verdict"
    )
    reason: str = Field(..., description="Reason for verdict")
    requires_human: bool = Field(..., description="Whether human review is required")
    confidence_level: Literal["high", "medium", "low"] = Field(
        ..., description="Confidence in verdict"
    )
    triggered_gates: List[str] = Field(
        default_factory=list, description="Which safety gates were triggered"
    )
    defect_summary: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class InspectionContext(BaseModel):
    """Context information for inspection."""
    image_id: str
    criticality: Literal["low", "medium", "high"] = "medium"
    domain: Optional[str] = None
    reference_standards: Optional[List[str]] = None
    user_notes: Optional[str] = None


# ============================================================================
# VLM AGENT BASE CLASS
# ============================================================================

class BaseVLMAgent:
    """Base class for VLM agents with retry logic and error handling."""
    
    def __init__(self, model_id: str, temperature: float, max_tokens: int, nickname: str):
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.nickname = nickname
        self.api_endpoint = f"{config.huggingface_api_endpoint}/{model_id}"
        self.headers = {"Authorization": f"Bearer {config.huggingface_api_key}"}
        self.logger = setup_logger(f"agent.{nickname}", level=config.log_level, component=nickname.upper())
    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    def _call_api(self, payload: dict, retries: int = None) -> dict:
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
                    self.logger.warning(f"Model loading, waiting {wait_time}s before retry...")
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


# Export schemas
__all__ = [
    "BoundingBox",
    "DefectInfo",
    "VLMAnalysisResult",
    "ConsensusResult",
    "SafetyVerdict",
    "InspectionContext",
    "BaseVLMAgent",
]

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


# ============================================================================
# AUDITOR AGENT (Llama 3.2 Vision)
# ============================================================================

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


# ============================================================================
# EXPLAINER AGENT (Llama 3.1 Text)
# ============================================================================

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
        self.logger = setup_logger("agent.explainer", level=config.log_level, component="EXPLAINER")
    
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
                explanation = "Unable to generate explanation due to unexpected response format."
            
            self.logger.info("Explanation generated successfully")
            
            return explanation.strip()
        
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            
            # Fallback explanation
            return (
                f"Inspection complete. The system detected {len(inspector_result.defects)} defects. "
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


# ============================================================================
# AGENT FACTORY
# ============================================================================

def get_inspector() -> InspectorAgent:
    """Get Inspector agent instance."""
    return InspectorAgent()


def get_auditor() -> AuditorAgent:
    """Get Auditor agent instance."""
    return AuditorAgent()


def get_explainer() -> ExplainerAgent:
    """Get Explainer agent instance."""
    return ExplainerAgent()


def health_check_agents() -> dict:
    """
    Perform health checks on all agents.
    
    Returns:
        Dict of agent_name -> (status: bool, details: str)
    """
    results = {}
    
    # Inspector
    try:
        inspector = get_inspector()
        status = inspector.health_check()
        results["Inspector (Qwen2-VL)"] = (
            status,
            f"Model: {config.vlm_inspector_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Inspector (Qwen2-VL)"] = (False, f"Error: {e}")
    
    # Auditor
    try:
        auditor = get_auditor()
        status = auditor.health_check()
        results["Auditor (Llama 3.2)"] = (
            status,
            f"Model: {config.vlm_auditor_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Auditor (Llama 3.2)"] = (False, f"Error: {e}")
    
    # Explainer
    try:
        explainer = get_explainer()
        status = explainer.health_check()
        results["Explainer (Llama 3.1)"] = (
            status,
            f"Model: {config.explainer_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Explainer (Llama 3.1)"] = (False, f"Error: {e}")
    
    return results


