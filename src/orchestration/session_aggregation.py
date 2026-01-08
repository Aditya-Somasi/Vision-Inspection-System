"""
Session aggregation utilities for multi-image inspection sessions.
Aggregates per-image results into session-level summaries.
"""

from typing import Dict, Any, List
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="SESSION_AGGREGATION")


def aggregate_session_results(image_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-image results into session-level summary.
    
    Args:
        image_results: Dictionary of image_id -> inspection result dict
    
    Returns:
        Aggregated session results dictionary
    """
    if not image_results:
        return {
            "total_images": 0,
            "completed_images": 0,
            "failed_images": 0,
            "aggregate_verdict": "UNKNOWN",
            "total_defects": 0,
            "critical_defects": 0,
            "moderate_defects": 0,
            "cosmetic_defects": 0
        }
    
    completed_count = 0
    failed_count = 0
    total_defects = 0
    critical_defects = 0
    moderate_defects = 0
    cosmetic_defects = 0
    all_verdicts = []
    
    for image_id, result in image_results.items():
        if result.get("completed", False):
            completed_count += 1
            
            # Get verdict
            safety_verdict = result.get("safety_verdict", {})
            verdict = safety_verdict.get("verdict", "UNKNOWN")
            all_verdicts.append(verdict)
            
            # Count defects
            consensus = result.get("consensus", {})
            defects = consensus.get("combined_defects", [])
            total_defects += len(defects)
            
            for defect in defects:
                severity = defect.get("safety_impact", "COSMETIC")
                if severity == "CRITICAL":
                    critical_defects += 1
                elif severity == "MODERATE":
                    moderate_defects += 1
                elif severity == "COSMETIC":
                    cosmetic_defects += 1
        else:
            failed_count += 1
    
    # Determine aggregate verdict
    aggregate_verdict = determine_aggregate_verdict(all_verdicts, total_defects)
    
    return {
        "total_images": len(image_results),
        "completed_images": completed_count,
        "failed_images": failed_count,
        "aggregate_verdict": aggregate_verdict,
        "total_defects": total_defects,
        "critical_defects": critical_defects,
        "moderate_defects": moderate_defects,
        "cosmetic_defects": cosmetic_defects,
        "verdict_distribution": {
            "SAFE": sum(1 for v in all_verdicts if v == "SAFE"),
            "UNSAFE": sum(1 for v in all_verdicts if v == "UNSAFE"),
            "REQUIRES_HUMAN_REVIEW": sum(1 for v in all_verdicts if v == "REQUIRES_HUMAN_REVIEW")
        }
    }


def determine_aggregate_verdict(verdicts: List[str], total_defects: int) -> str:
    """
    Determine aggregate verdict from per-image verdicts.
    
    Conservative approach:
    - If any UNSAFE → UNSAFE
    - If any REQUIRES_HUMAN_REVIEW → REQUIRES_HUMAN_REVIEW
    - If all SAFE → SAFE
    - If mixed with defects → REQUIRES_HUMAN_REVIEW
    """
    if not verdicts:
        return "UNKNOWN"
    
    # If any image is UNSAFE, session is UNSAFE (conservative)
    if any(v == "UNSAFE" for v in verdicts):
        return "UNSAFE"
    
    # If any image requires human review, session requires review
    if any(v == "REQUIRES_HUMAN_REVIEW" for v in verdicts):
        return "REQUIRES_HUMAN_REVIEW"
    
    # If all are SAFE, session is SAFE
    if all(v == "SAFE" for v in verdicts):
        return "SAFE"
    
    # Mixed verdicts or defects found → requires review
    if total_defects > 0:
        return "REQUIRES_HUMAN_REVIEW"
    
    # Default to review for safety
    return "REQUIRES_HUMAN_REVIEW"
