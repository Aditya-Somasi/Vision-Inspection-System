"""
Agent factory and exports for Vision Inspection System.
"""

from src.agents.inspector import InspectorAgent
from src.agents.auditor import AuditorAgent
from src.agents.explainer import ExplainerAgent
from utils.config import config


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


__all__ = [
    "InspectorAgent",
    "AuditorAgent",
    "ExplainerAgent",
    "get_inspector",
    "get_auditor",
    "get_explainer",
    "health_check_agents",
]
