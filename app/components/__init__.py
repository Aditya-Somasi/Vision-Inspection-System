"""
UI Components for Vision Inspection System.
"""

from app.components.verdict_display import (
    render_verdict_banner,
    render_confidence_bar,
    render_gate_results,
    render_confidence_metrics,
    render_defect_summary
)
from app.components.decision_support import render_decision_support
from app.components.chat_widget import chat_widget, clear_chat
from app.components.sidebar import render_sidebar
from app.components.image_upload import (
    render_multi_image_upload_zone,
    render_image_preview_gallery,
    render_batch_config_form,
    process_uploaded_files
)
from app.components.inspection_progress import (
    render_live_inspection_tab,
    render_per_image_progress_card,
    render_session_progress_dashboard
)
from app.components.results_view import (
    render_results_review_tab,
    render_image_verdict_card,
    render_session_summary
)

__all__ = [
    # Verdict Display
    "render_verdict_banner",
    "render_confidence_bar",
    "render_gate_results",
    "render_confidence_metrics",
    "render_defect_summary",
    # Decision Support
    "render_decision_support",
    # Chat Widget
    "chat_widget",
    "clear_chat",
    # Sidebar
    "render_sidebar",
    # Image Upload
    "render_multi_image_upload_zone",
    "render_image_preview_gallery",
    "render_batch_config_form",
    "process_uploaded_files",
    # Inspection Progress
    "render_live_inspection_tab",
    "render_per_image_progress_card",
    "render_session_progress_dashboard",
    # Results View
    "render_results_review_tab",
    "render_image_verdict_card",
    "render_session_summary",
]
