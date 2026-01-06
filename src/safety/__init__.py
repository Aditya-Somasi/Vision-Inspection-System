"""
Safety module for Vision Inspection System.
"""

from src.safety.consensus import ConsensusAnalyzer, analyze_consensus
from src.safety.gates import SafetyGateEngine, evaluate_safety

__all__ = [
    "ConsensusAnalyzer",
    "SafetyGateEngine",
    "analyze_consensus",
    "evaluate_safety",
]
