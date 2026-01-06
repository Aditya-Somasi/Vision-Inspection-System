"""
Reporting module for Vision Inspection System.
"""

from src.reporting.pdf import generate_report, InspectionReport

__all__ = [
    "generate_report",
    "InspectionReport",
]
