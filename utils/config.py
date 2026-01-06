"""
Unified configuration management with Pydantic validation.
Loads and validates all environment variables.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Application configuration with validation."""
    
    # ========================
    # HuggingFace Configuration
    # ========================
    huggingface_api_key: str = Field(..., alias="HUGGINGFACE_API_KEY")
    huggingface_api_endpoint: str = Field(
        default="https://api-inference.huggingface.co/models",
        alias="HUGGINGFACE_API_ENDPOINT"
    )
    
    # ========================
    # VLM Model Configuration
    # ========================
    vlm_inspector_model: str = Field(
        default="Qwen/Qwen2-VL-7B-Instruct",
        alias="VLM_INSPECTOR_MODEL"
    )
    vlm_inspector_temperature: float = Field(
        default=0.1,
        alias="VLM_INSPECTOR_TEMPERATURE"
    )
    vlm_inspector_max_tokens: int = Field(
        default=1024,
        alias="VLM_INSPECTOR_MAX_TOKENS"
    )
    
    vlm_auditor_model: str = Field(
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        alias="VLM_AUDITOR_MODEL"
    )
    vlm_auditor_temperature: float = Field(
        default=0.2,
        alias="VLM_AUDITOR_TEMPERATURE"
    )
    vlm_auditor_max_tokens: int = Field(
        default=1024,
        alias="VLM_AUDITOR_MAX_TOKENS"
    )
    
    explainer_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        alias="EXPLAINER_MODEL"
    )
    explainer_temperature: float = Field(
        default=0.3,
        alias="EXPLAINER_TEMPERATURE"
    )
    explainer_max_tokens: int = Field(
        default=2048,
        alias="EXPLAINER_MAX_TOKENS"
    )
    
    # ========================
    # Safety Configuration
    # ========================
    confidence_threshold: float = Field(
        default=0.7,
        alias="CONFIDENCE_THRESHOLD"
    )
    max_defects_auto: int = Field(
        default=2,
        alias="MAX_DEFECTS_AUTO"
    )
    vlm_agreement_required: bool = Field(
        default=True,
        alias="VLM_AGREEMENT_REQUIRED"
    )
    high_criticality_requires_review: bool = Field(
        default=True,
        alias="HIGH_CRITICALITY_REQUIRES_REVIEW"
    )
    low_confidence_threshold: float = Field(
        default=0.5,
        alias="LOW_CONFIDENCE_THRESHOLD"
    )
    critical_defect_types: str = Field(
        default="crack,fracture,corrosion,structural_damage,deformation",
        alias="CRITICAL_DEFECT_TYPES"
    )
    
    # ========================
    # LangSmith Configuration
    # ========================
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langchain_tracing_v2: str = Field(default="false", alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="vision-inspection", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGCHAIN_ENDPOINT"
    )
    
    # ========================
    # Database Configuration
    # ========================
    database_path: str = Field(default="inspections.db", alias="DATABASE_PATH")
    database_echo: bool = Field(default=False, alias="DATABASE_ECHO")
    chat_history_db: str = Field(default="chat_history.db", alias="CHAT_HISTORY_DB")
    
    # ========================
    # File Storage Configuration
    # ========================
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    report_dir: str = Field(default="reports", alias="REPORT_DIR")
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    max_file_size_mb: int = Field(default=10, alias="MAX_FILE_SIZE_MB")
    allowed_extensions: str = Field(
        default="jpg,jpeg,png,bmp,tiff,webp",
        alias="ALLOWED_EXTENSIONS"
    )
    
    # ========================
    # Logging Configuration
    # ========================
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="detailed", alias="LOG_FORMAT")
    log_to_console: bool = Field(default=True, alias="LOG_TO_CONSOLE")
    log_to_file: bool = Field(default=True, alias="LOG_TO_FILE")
    log_to_langsmith: bool = Field(default=True, alias="LOG_TO_LANGSMITH")
    
    # ========================
    # API Configuration
    # ========================
    api_timeout: int = Field(default=60, alias="API_TIMEOUT")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    api_retry_backoff: int = Field(default=2, alias="API_RETRY_BACKOFF")
    
    # ========================
    # Chat Memory Configuration
    # ========================
    enable_chat_memory: bool = Field(default=True, alias="ENABLE_CHAT_MEMORY")
    max_chat_history: int = Field(default=50, alias="MAX_CHAT_HISTORY")
    
    # ========================
    # UI Configuration
    # ========================
    app_title: str = Field(default="Vision Inspection System", alias="APP_TITLE")
    default_criticality: str = Field(default="medium", alias="DEFAULT_CRITICALITY")
    show_debug_info: bool = Field(default=False, alias="SHOW_DEBUG_INFO")
    enable_analytics: bool = Field(default=True, alias="ENABLE_ANALYTICS")
    
    # ========================
    # Performance Configuration
    # ========================
    enable_streaming: bool = Field(default=True, alias="ENABLE_STREAMING")
    max_concurrent_calls: int = Field(default=3, alias="MAX_CONCURRENT_CALLS")
    max_image_dimension: int = Field(default=2048, alias="MAX_IMAGE_DIMENSION")
    
    # ========================
    # Development Configuration
    # ========================
    environment: str = Field(default="development", alias="ENVIRONMENT")
    skip_health_checks: bool = Field(default=False, alias="SKIP_HEALTH_CHECKS")
    use_mock_responses: bool = Field(default=False, alias="USE_MOCK_RESPONSES")
    verbose_errors: bool = Field(default=True, alias="VERBOSE_ERRORS")
    
    # ========================
    # Validators
    # ========================
    
    @field_validator("huggingface_api_key")
    @classmethod
    def validate_hf_key(cls, v: str) -> str:
        """Validate HuggingFace API key format."""
        if not v or v == "hf_xxxxxxxxxxxxx":
            raise ValueError(
                "HUGGINGFACE_API_KEY is required. Get one from: "
                "https://huggingface.co/settings/tokens"
            )
        if not v.startswith("hf_"):
            raise ValueError("HUGGINGFACE_API_KEY must start with 'hf_'")
        return v
    
    @field_validator("confidence_threshold", "low_confidence_threshold")
    @classmethod
    def validate_confidence(cls, v: float, info) -> float:
        """Validate confidence thresholds."""
        if not 0 <= v <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return v
    
    @field_validator("langchain_tracing_v2")
    @classmethod
    def validate_langsmith(cls, v: str, info) -> str:
        """Validate LangSmith configuration."""
        if v.lower() == "true":
            api_key = os.getenv("LANGSMITH_API_KEY")
            if not api_key:
                print(
                    "WARNING: LANGCHAIN_TRACING_V2 is enabled but LANGSMITH_API_KEY is missing. "
                    "Tracing will be disabled."
                )
                return "false"
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("default_criticality")
    @classmethod
    def validate_criticality(cls, v: str) -> str:
        """Validate criticality level."""
        valid_levels = ["low", "medium", "high"]
        if v.lower() not in valid_levels:
            raise ValueError(f"DEFAULT_CRITICALITY must be one of {valid_levels}")
        return v.lower()
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"ENVIRONMENT must be one of {valid_envs}")
        return v.lower()
    
    # ========================
    # Helper Properties
    # ========================
    
    @property
    def critical_defect_types_list(self) -> List[str]:
        """Get critical defect types as list."""
        return [t.strip() for t in self.critical_defect_types.split(",")]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        """Get allowed extensions as list."""
        return [ext.strip() for ext in self.allowed_extensions.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    @property
    def langsmith_enabled(self) -> bool:
        """Check if LangSmith tracing is enabled."""
        return self.langchain_tracing_v2.lower() == "true" and self.langsmith_api_key is not None
    
    def get_upload_dir(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_report_dir(self) -> Path:
        """Get report directory as Path object."""
        path = Path(self.report_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_log_dir(self) -> Path:
        """Get log directory as Path object."""
        path = Path(self.log_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


def get_config() -> Config:
    """
    Load and validate configuration.
    Exits if configuration is invalid.
    """
    try:
        config = Config()
        
        # Set LangSmith environment variables if enabled
        if config.langsmith_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = config.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = config.langchain_endpoint
        
        return config
    
    except ValidationError as e:
        print("\n❌ Configuration Error:")
        print("=" * 60)
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            print(f"\n Field: {field}")
            print(f"  Error: {error['msg']}")
            if "input" in error:
                print(f"  Value: {error['input']}")
        print("\n" + "=" * 60)
        print("\nPlease check your .env file and fix the errors above.")
        print("See .env.example for reference.\n")
        raise SystemExit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected configuration error: {e}\n")
        raise SystemExit(1)


# Global configuration instance
config = get_config()


# Export commonly used paths
UPLOAD_DIR = config.get_upload_dir()
REPORT_DIR = config.get_report_dir()
LOG_DIR = config.get_log_dir()