"""
Reporting module for Vision Inspection System.
"""

from src.reporting.pdf_generator import generate_report, InspectionReport

__all__ = [
    "generate_report",
    "InspectionReport",
]
