"""
Agent factory and exports for Vision Inspection System.
"""

from src.agents.vlm_inspector import VLMInspectorAgent, InspectorAgent
from src.agents.vlm_auditor import VLMAuditorAgent, AuditorAgent
from src.agents.explainer import ExplainerAgent
from utils.config import config


def get_inspector() -> VLMInspectorAgent:
    """Get Inspector agent instance."""
    return VLMInspectorAgent()


def get_auditor() -> VLMAuditorAgent:
    """Get Auditor agent instance."""
    return VLMAuditorAgent()


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
        results["Inspector (HuggingFace)"] = (
            status,
            f"Model: {config.vlm_inspector_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Inspector (HuggingFace)"] = (False, f"Error: {e}")
    
    # Auditor
    try:
        auditor = get_auditor()
        status = auditor.health_check()
        results["Auditor (HuggingFace)"] = (
            status,
            f"Model: {config.vlm_auditor_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Auditor (HuggingFace)"] = (False, f"Error: {e}")
    
    # Explainer
    try:
        explainer = get_explainer()
        status = explainer.health_check()
        results["Explainer (Groq)"] = (
            status,
            f"Model: {config.explainer_model}" if status else "Connection failed"
        )
    except Exception as e:
        results["Explainer (Groq)"] = (False, f"Error: {e}")
    
    return results


__all__ = [
    "VLMInspectorAgent",
    "VLMAuditorAgent",
    "InspectorAgent",
    "AuditorAgent",
    "ExplainerAgent",
    "get_inspector",
    "get_auditor",
    "get_explainer",
    "health_check_agents",
]
