"""
Streamlit UI for Vision Inspection System.
Professional frontend with enhanced UX and state management.
"""

import streamlit as st
import time
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import plotly.express as px
import pandas as pd
import base64

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config, UPLOAD_DIR, REPORT_DIR
from src.orchestration import run_inspection, run_multi_image_inspection
from src.orchestration.session_aggregation import aggregate_session_results
from src.database import InspectionRepository
from src.chat_memory import get_memory_manager, get_session_history
from utils.logger import setup_logger
from src.reporting import generate_report

# Import modular components
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
from app.components.inspection_progress import render_live_inspection_tab
from app.components.results_view import render_results_review_tab
from app.services.file_handler import save_uploaded_file, validate_image
from app.services.session_manager import init_session_state, get_state, set_state

# Configure page
st.set_page_config(
    page_title=config.app_title,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    # No fallback needed - external CSS file contains all styles
    pass


# Setup logger
logger = setup_logger(__name__, level=config.log_level, component="UI")



# ============================================================================
# SESSION STATE - Using imported session_manager module
# ============================================================================
# init_session_state is imported from app.services.session_manager


# ============================================================================
# UTILITY FUNCTIONS - Using imported components
# ============================================================================
# save_uploaded_file is imported from app.services.file_handler
# display_* functions are imported from app.components.*

# Function aliases for backward compatibility with rest of codebase
# These redirect to the new modular components:
display_verdict_banner = render_verdict_banner
display_confidence_bar = render_confidence_bar  
display_all_gate_results = render_gate_results
display_decision_support = render_decision_support
# Note: chat_widget is imported directly and used


def display_inspection_results(results: Dict[str, Any]):
    """Display inspection results in enhanced UI with confidence bars and gate display."""
    verdict = results.get("safety_verdict", {})
    consensus = results.get("consensus", {})
    
    defects = consensus.get("combined_defects", [])
    defect_count = len(defects)
    
    # Get gate results for display
    gate_results = verdict.get("defect_summary", {}).get("all_gate_results", [])
    
    # Verdict banner with defect count for all-clear display
    display_verdict_banner(verdict.get("verdict", "UNKNOWN"), defect_count, gate_results)
    
    # Confidence Metrics Section - Using component
    inspector_result = results.get("inspector_result", {})
    auditor_result = results.get("auditor_result", {})
    agreement_score = consensus.get("agreement_score", 0.5)
    processing_time = results.get('processing_time') or 0
    
    render_confidence_metrics(
        inspector_confidence=inspector_result.get("overall_confidence", "medium"),
        auditor_confidence=auditor_result.get("overall_confidence", "medium"),
        agreement_score=agreement_score,
        processing_time=processing_time
    )
    
    # Summary metrics row - Using component
    st.divider()
    render_defect_summary(defects)
    
    # Decision Support Section
    display_decision_support(results)

    # Safety Gate Evaluation (show all gates)
    if gate_results:
        st.divider()
        display_all_gate_results(gate_results)
    
    # Show criticality upgrade notification if applicable
    context = results.get("context", {})
    if context.get("criticality_upgraded"):
        original = context.get("original_criticality", "unknown")
        upgraded = context.get("criticality", "unknown")
        reason = context.get("upgrade_reason", "Based on detected object type")
        st.warning(
            f"⚠️ **Criticality Upgraded**: Agent upgraded criticality from "
            f"**{original.upper()}** → **{upgraded.upper()}**\n\n"
            f"*Reason: {reason}*"
        )
    
    # Show inferred criticality from Inspector
    inspector_result = results.get("inspector_result", {})
    if inspector_result.get("inferred_criticality"):
        inferred = inspector_result.get("inferred_criticality")
        inferred_reason = inspector_result.get("inferred_criticality_reasoning", "")
        
        criticality_colors = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = criticality_colors.get(inferred, "⚪")
        
        with st.expander(f"🤖 AI-Inferred Criticality: {emoji} {inferred.upper()}"):
            st.markdown(f"**Reasoning:** {inferred_reason}")
            st.caption("The AI agent analyzed the image and automatically determined the appropriate criticality level.")
    
    st.divider()
    
    # Explanation
    st.subheader("📝 Analysis Summary")
    explanation = results.get("explanation")
    if explanation:
        st.info(explanation)
    else:
        st.info("⚠️ Analysis summary pending system completion.")
    
    # Visual Evidence Section - 3-Panel Layout
    st.subheader("🖼️ Visual Evidence (3-Panel View)")
    image_path = results.get("image_path")
    
    # Try alternate location if main path fails (sometimes relative/absolute mix-up)
    if not image_path and results.get("inspector_result", {}).get("image_path"):
         image_path = results.get("inspector_result", {}).get("image_path")
         
    if image_path and Path(image_path).exists():
        try:
            from utils.image_utils import create_heatmap_overlay, draw_bounding_boxes
            from utils.config import REPORT_DIR
            
            image_path = Path(image_path)
            
            # Create heatmap overlay
            heatmap_path = REPORT_DIR / f"heatmap_{image_path.stem}.jpg"
            if not heatmap_path.exists():
                create_heatmap_overlay(image_path, defects, heatmap_path)
            
            # Create annotated image with numbered markers
            annotated_path = REPORT_DIR / f"annotated_{image_path.stem}.jpg"
            if defects and not annotated_path.exists():
                boxes = []
                for i, defect in enumerate(defects, 1):
                    bbox = defect.get("bbox", {})
                    boxes.append({
                        "x": bbox.get("x", 50 + i*30),
                        "y": bbox.get("y", 50 + i*30),
                        "width": bbox.get("width", 50),
                        "height": bbox.get("height", 50),
                        "label": f"#{i}",
                        "severity": defect.get("safety_impact", "MODERATE")
                    })
                draw_bounding_boxes(image_path, boxes, annotated_path)
            
            # 3-Panel Display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**1. Original Image**")
                st.image(str(image_path), width="stretch")
            
            with col2:
                st.markdown("**2. Defect Heatmap**")
                if heatmap_path.exists():
                    st.image(str(heatmap_path), width="stretch")
                else:
                    st.image(str(image_path), width="stretch")
            
            with col3:
                st.markdown("**3. Numbered Markers**")
                if annotated_path.exists():
                    st.image(str(annotated_path), width="stretch")
                else:
                    st.image(str(image_path), width="stretch")
            
            # Legend for markers
            if defects:
                legend_text = "**Legend:** "
                for i, defect in enumerate(defects, 1):
                    severity = defect.get("safety_impact", "UNKNOWN")
                    defect_type = defect.get("type", "unknown")
                    severity_emoji = {"CRITICAL": "🔴", "MODERATE": "🟡", "COSMETIC": "🔵"}.get(severity, "⚪")
                    legend_text += f"**#{i}** = {defect_type.title()} ({severity_emoji}) &nbsp;&nbsp; "
                st.markdown(legend_text, unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Failed to display 3-panel images: {e}")
            st.image(str(image_path), caption="📷 Uploaded Image")
    else:
        st.warning(f"Original image not available for comparison ({image_path or 'unknown path'})")
    
    st.divider()
    
    # Safety Gates Evaluation Section (ALL gates)
    gate_results = verdict.get("defect_summary", {}).get("all_gate_results", [])
    if gate_results:
        with st.expander("🔒 Safety Gates Evaluation (All Gates)", expanded=True):
            for gate in gate_results:
                passed = gate.get("passed", True)
                gate_name = gate.get("display_name", gate.get("gate_id", "Unknown"))
                message = gate.get("message", "")
                
                if passed:
                    st.markdown(f"✅ **{gate_name}**: {message}")
                else:
                    st.markdown(f"❌ **{gate_name}**: {message}")
    
    # Disagreement Analysis (if models disagree)
    if not consensus.get("models_agree", True):
        st.warning("⚠️ **Model Disagreement Detected**")
        
        inspector_defects = len(results.get("inspector_result", {}).get("defects", []))
        auditor_defects = len(results.get("auditor_result", {}).get("defects", []))
        
        disagree_col1, disagree_col2 = st.columns(2)
        with disagree_col1:
            st.metric("Inspector Found", f"{inspector_defects} defect(s)")
        with disagree_col2:
            st.metric("Auditor Found", f"{auditor_defects} defect(s)")
        
        st.info(f"""
        **Disagreement Details:**
        - Agreement Score: {consensus.get('agreement_score', 0):.0%}
        - {consensus.get('disagreement_details', 'Defect count mismatch between models')}
        - Resolution: Combined {defect_count} unique defects from both models
        """)
    
    st.divider()
    
    # Defect details
    if defects:
        st.subheader(f"🔍 Defect Details ({len(defects)} found)")
        
        for i, defect in enumerate(defects, 1):
            severity = defect.get("safety_impact", "UNKNOWN")
            severity_emoji = {
                "CRITICAL": "🔴",
                "MODERATE": "🟡",
                "COSMETIC": "🔵"
            }.get(severity, "⚪")
            
            with st.expander(
                f"{severity_emoji} **{i}. {defect.get('type', 'Unknown').upper()}** "
                f"- {severity}"
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**📍 Location:**")
                    st.write(defect.get("location", "Not specified"))
                    
                    st.markdown("**🎯 Confidence:**")
                    confidence = defect.get("confidence", "unknown")
                    conf_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                        confidence, "⚪"
                    )
                    st.write(f"{conf_emoji} {confidence.title()}")
                
                with col2:
                    st.markdown("**💭 Reasoning:**")
                    st.write(defect.get("reasoning", "Not provided"))
                    
                    st.markdown("**📋 Recommended Action:**")
                    st.warning(defect.get("recommended_action", "Not provided"))
    
    # Model comparison
    with st.expander("📊 Model Analysis Comparison"):
        inspector = results.get("inspector_result", {})
        auditor = results.get("auditor_result", {})
        
        comparison_df = pd.DataFrame({
            "Metric": [
                "Object Identified",
                "Overall Condition",
                "Defects Found",
                "Confidence Level"
            ],
            "Inspector (Qwen2.5-VL)": [
                str(inspector.get("object_identified", "N/A")),
                str(inspector.get("overall_condition", "N/A")),
                str(len(inspector.get("defects", []))),
                str(inspector.get("overall_confidence", "N/A"))
            ],
            "Auditor (Llama 4 Maverick)": [
                str(auditor.get("object_identified", "N/A")),
                str(auditor.get("overall_condition", "N/A")),
                str(len(auditor.get("defects", []))),
                str(auditor.get("overall_confidence", "N/A"))
            ]
        })
        
        st.dataframe(comparison_df, hide_index=True)
    
    # PDF Report Section with Preview
    if results.get("report_path"):
        st.divider()
        st.subheader("📄 Inspection Report")
        
        report_path = Path(results["report_path"])
        
        # Action buttons (removed duplicate column definition)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with open(report_path, "rb") as f:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=f,
                    file_name=report_path.name,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
        
        with col2:
            st.caption(f"📁 {report_path.name}")
            st.caption(f"📍 {report_path.absolute()}")
            
        # Embedded PDF Viewer
        st.divider()
        st.subheader("👁️ Live PDF Preview")
        
        with open(report_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)


# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

def analytics_dashboard():
    """Display analytics dashboard with charts."""
    st.header("📊 Analytics Dashboard")
    
    try:
        repo = InspectionRepository()
        stats = repo.get_defect_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Inspections",
                stats.get("total_inspections", 0)
            )
        with col2:
            st.metric(
                "Agreement Rate",
                f"{stats.get('agreement_rate', 0):.1%}"
            )
        with col3:
            st.metric(
                "Avg Processing Time",
                f"{stats.get('avg_processing_time', 0):.2f}s"
            )
        with col4:
            st.metric(
                "Total Defects",
                sum(stats.get("defect_counts", {}).values())
            )
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Defects by Type")
            defect_counts = stats.get("defect_counts", {})
            if defect_counts:
                fig = px.pie(
                    names=list(defect_counts.keys()),
                    values=list(defect_counts.values()),
                    title="Distribution of Defect Types",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(showlegend=True, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No defect data available yet.")
        
        with col2:
            st.subheader("Verdicts Distribution")
            verdict_counts = stats.get("verdict_counts", {})
            if verdict_counts:
                colors = {
                    "SAFE": "#10b981",
                    "UNSAFE": "#ef4444",
                    "REQUIRES_HUMAN_REVIEW": "#f59e0b"
                }
                fig = px.bar(
                    x=list(verdict_counts.keys()),
                    y=list(verdict_counts.values()),
                    title="Inspection Verdicts",
                    labels={"x": "Verdict", "y": "Count"},
                    color=list(verdict_counts.keys()),
                    color_discrete_map=colors
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No verdict data available yet.")
        
        # Recent inspections
        st.subheader("📋 Recent Inspections")
        recent = repo.list_inspections(limit=10)
        
        if recent:
            df = pd.DataFrame([{
                "ID": r.inspection_id[:8],
                "Image": r.image_filename,
                "Verdict": r.overall_verdict,
                "Defects": r.defect_count,
                "Critical": r.critical_defect_count,
                "Criticality": r.criticality.upper(),
                "Date": r.created_at.strftime("%Y-%m-%d %H:%M")
            } for r in recent])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Verdict": st.column_config.TextColumn(
                        "Verdict",
                        help="Safety verdict"
                    ),
                    "Critical": st.column_config.NumberColumn(
                        "Critical",
                        help="Number of critical defects"
                    )
                }
            )
        else:
            st.info(
                "📋 No inspection history yet. "
                "Run your first inspection to see data here."
            )
    
    except Exception as e:
        st.error(f"❌ Failed to load analytics: {e}")
        logger.error(f"Analytics dashboard error: {e}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application with tabbed layout."""
    init_session_state()
    
    # Enhanced Sidebar
    with st.sidebar:
        page = render_sidebar()
    
    # Main content area
    if page == "🏠 Inspection Session":
        inspection_session_page()
    elif page == "📊 Analytics":
        analytics_dashboard()
    elif page == "📋 Inspection History":
        inspection_history_page()
    else:  # Settings
        settings_page()


def inspection_session_page():
    """Inspection session page with tabbed workflow."""
    st.title("🔍 Visual Inspection System")
    st.caption("AI-powered damage detection and safety analysis")
    
    # Create 4 tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Configure",
        "🔄 Live Inspection",
        "📋 Results & Review",
        "💬 Chat & Analysis"
    ])
    
    with tab1:
        render_upload_configure_tab()
    
    with tab2:
        render_live_inspection_tab()
    
    with tab3:
        render_results_review_tab()
    
    with tab4:
        render_chat_analysis_tab()


def render_upload_configure_tab():
    """TAB 1: Upload & Configure - Multi-image upload and session configuration."""
    st.header("📤 Upload & Configure")
    
    # Multi-image upload zone
    uploaded_files = render_multi_image_upload_zone()
    
    # Batch configuration form
    metadata = render_batch_config_form()
    
    # Process uploaded files button
    if uploaded_files:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📥 Process Uploaded Files", type="primary", use_container_width=True):
                process_uploaded_files(uploaded_files, metadata)
                st.success(f"✅ Processed {len(uploaded_files)} file(s)")
                st.rerun()
    
    # Image preview gallery
    st.divider()
    render_image_preview_gallery()
    
    # Start Inspection button (if images uploaded)
    uploaded_images = get_state("uploaded_images", [])
    if uploaded_images:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            start_inspection = st.button("🚀 Start Inspection", type="primary", use_container_width=True)
            
            if start_inspection:
                # Get session metadata
                session_metadata = get_state("session_metadata", {})
                criticality = session_metadata.get("criticality", "medium")
                domain = session_metadata.get("domain")
                user_notes = session_metadata.get("user_notes")
                session_id = get_state("current_session_id")
                
                # Update image statuses to processing
                for img in uploaded_images:
                    img["status"] = "processing"
                set_state("uploaded_images", uploaded_images)
                set_state("session_status", "processing")
                
                # Prepare image mappings (path -> image_id from uploaded_images)
                image_path_to_id = {img["filepath"]: img["image_id"] for img in uploaded_images}
                image_paths = list(image_path_to_id.keys())
                
                # Run multi-image inspection
                with st.spinner(f"🔄 Processing {len(image_paths)} image(s)..."):
                    try:
                        # Update all to processing
                        for img in uploaded_images:
                            img["status"] = "processing"
                        set_state("uploaded_images", uploaded_images)
                        
                        # Pass image_id_map to preserve original IDs
                        results = run_multi_image_inspection(
                            image_paths=image_paths,
                            criticality=criticality,
                            domain=domain,
                            user_notes=user_notes,
                            session_id=session_id,
                            image_id_map=image_path_to_id
                        )
                        
                        # Map results back to uploaded_images
                        image_results_dict = results.get("image_results", {})
                        
                        # Update uploaded_images with results
                        for img in uploaded_images:
                            img_id = img["image_id"]
                            if img_id in image_results_dict:
                                result = image_results_dict[img_id]
                                img["inspection_result"] = result
                                img["status"] = "complete" if result.get("completed") else "failed"
                                if result.get("error"):
                                    img["error"] = result.get("error")
                        
                        set_state("uploaded_images", uploaded_images)
                        set_state("image_results", image_results_dict)
                        set_state("session_results", results.get("session_results", {}))
                        
                        set_state("uploaded_images", uploaded_images)
                        set_state("session_status", "complete")
                        
                        st.success(f"✅ Inspection complete! Processed {results['session_results']['completed_images']}/{len(image_paths)} images")
                        
                        # Switch to results tab
                        set_state("active_tab", "results")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Multi-image inspection failed: {e}", exc_info=True)
                        st.error(f"❌ Inspection failed: {e}")
                        
                        # Mark all as failed
                        for img in uploaded_images:
                            img["status"] = "failed"
                            img["error"] = str(e)
                        
                        set_state("uploaded_images", uploaded_images)
                        set_state("session_status", "error")


def render_chat_analysis_tab():
    """TAB 4: Chat & Analysis - Context-aware chat widget."""
    st.header("💬 Chat & Analysis")
    
    # Get session results for chat context
    uploaded_images = get_state("uploaded_images", [])
    image_results = get_state("image_results", {})
    session_results = get_state("session_results", {})
    
    # Combine results for chat context
    # For now, use the most recent result or create aggregate context
    chat_results = None
    
    if image_results:
        # Use the first available result as context
        first_image_id = list(image_results.keys())[0]
        chat_results = image_results.get(first_image_id)
    elif session_results:
        # Create a synthetic result from session results
        chat_results = {
            "safety_verdict": {"verdict": session_results.get("aggregate_verdict", "UNKNOWN")},
            "consensus": {"combined_defects": []},
            "session_summary": session_results
        }
    
    # Fallback to legacy inspection_results for backward compatibility
    if not chat_results:
        chat_results = get_state("inspection_results")
    
    if chat_results and config.enable_chat_memory:
        chat_widget(chat_results)
    else:
        st.info("💬 Start an inspection to enable chat. Results will appear here for Q&A about findings.")


def inspection_history_page():
    """Inspection History page - shows past inspection sessions."""
    st.header("📋 Inspection History")
    
    try:
        from src.database import InspectionRepository
        repo = InspectionRepository()
        recent = repo.list_inspections(limit=20)
        
        if recent:
            import pandas as pd
            
            df = pd.DataFrame([{
                "ID": r.inspection_id[:8],
                "Image": r.image_filename,
                "Verdict": r.overall_verdict,
                "Defects": r.defect_count,
                "Critical": r.critical_defect_count,
                "Criticality": r.criticality.upper(),
                "Date": r.created_at.strftime("%Y-%m-%d %H:%M")
            } for r in recent])
            
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Verdict": st.column_config.TextColumn("Verdict", help="Safety verdict"),
                    "Critical": st.column_config.NumberColumn("Critical", help="Critical defects")
                }
            )
        else:
            st.info("📋 No inspection history yet. Run your first inspection to see data here.")
    
    except Exception as e:
        st.error(f"❌ Failed to load inspection history: {e}")
        logger.error(f"Inspection history error: {e}")


def settings_page():
    """Settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("🤖 Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Inspector:** {config.vlm_inspector_model}")
        st.info(f"**Auditor:** {config.vlm_auditor_model}")
    with col2:
        st.info(f"**Explainer:** {config.explainer_model}")
    
    st.subheader("🛡️ Safety Settings")
    settings_data = {
        "Confidence Threshold": config.confidence_threshold,
        "Max Auto Defects": config.max_defects_auto,
        "VLM Agreement Required": config.vlm_agreement_required,
        "High Criticality Review": config.high_criticality_requires_review
    }
    for key, value in settings_data.items():
        st.write(f"**{key}:** {value}")
    
    st.subheader("💻 System Information")
    st.write(f"**Environment:** {config.environment.upper()}")
    st.write(f"**Database:** {config.database_path}")
    st.write(f"**LangSmith:** {'Enabled' if config.langsmith_enabled else 'Disabled'}")
    
    st.divider()
    
    if st.button("🗑️ Clear Session Data", type="secondary"):
        st.session_state.clear()  # Streamlit built-in clear - keep as is
        init_session_state()
        st.success("✅ Session data cleared!")
        st.rerun()


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
