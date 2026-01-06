"""
Streamlit UI for Vision Inspection System.
Professional frontend with state management, file handling, and analytics.
"""

import streamlit as st
import time
from pathlib import Path
from typing import Optional, Dict, Any
import uuid
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from config import config, UPLOAD_DIR, REPORT_DIR
from workflow import run_inspection
from database import InspectionRepository
from chat_memory import get_memory_manager, get_session_history, rewrite_query_with_history
from logger import setup_logger, set_request_id
from models import get_explainer
from reports import generate_report

# Configure page
st.set_page_config(
    page_title=config.app_title,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logger
logger = setup_logger(__name__, level=config.log_level, component="UI")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.inspection_results = None
        st.session_state.chat_session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.processing = False
        st.session_state.current_image_path = None
        st.session_state.show_analytics = False
        
        logger.info(f"Session initialized: {st.session_state.chat_session_id}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_uploaded_file(uploaded_file) -> Optional[Path]:
    """
    Save uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        Path to saved file, or None if error
    """
    try:
        # Validate file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            st.error(f"File too large: {file_size_mb:.1f}MB (max: {config.max_file_size_mb}MB)")
            return None
        
        # Validate extension
        file_ext = Path(uploaded_file.name).suffix[1:].lower()
        if file_ext not in config.allowed_extensions_list:
            st.error(f"Invalid file type: .{file_ext} (allowed: {', '.join(config.allowed_extensions_list)})")
            return None
        
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{uploaded_file.name}"
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File uploaded: {filename} ({file_size_mb:.2f}MB)")
        
        return file_path
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        st.error(f"Failed to upload file: {e}")
        return None


def display_inspection_results(results: Dict[str, Any]):
    """Display inspection results in UI."""
    verdict = results.get("safety_verdict", {})
    consensus = results.get("consensus", {})
    
    # Verdict banner
    verdict_text = verdict.get("verdict", "UNKNOWN")
    
    if verdict_text == "SAFE":
        st.success(f"### ✅ SAFE - No Critical Defects Detected")
    elif verdict_text == "UNSAFE":
        st.error(f"### 🚫 UNSAFE - Critical Defect Detected")
    else:
        st.warning(f"### ⚠️ REQUIRES HUMAN REVIEW")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    defects = consensus.get("combined_defects", [])
    critical_count = sum(1 for d in defects if d.get("safety_impact") == "CRITICAL")
    
    col1.metric("Total Defects", len(defects))
    col2.metric("Critical Defects", critical_count)
    col3.metric(
        "Models Agree",
        "Yes" if consensus.get("models_agree") else "No",
        delta="High confidence" if consensus.get("models_agree") else "Review needed"
    )
    col4.metric(
        "Processing Time",
        f"{results.get('processing_time', 0):.2f}s"
    )
    
    st.divider()
    
    # Explanation
    st.subheader("Analysis Summary")
    explanation = results.get("explanation", "No explanation available.")
    st.write(explanation)
    
    # Defect details
    if defects:
        st.subheader(f"Defect Details ({len(defects)} found)")
        
        for i, defect in enumerate(defects, 1):
            with st.expander(f"**{i}. {defect.get('type', 'Unknown').upper()}** - {defect.get('safety_impact')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write("**Location:**")
                    st.write(defect.get("location", "Not specified"))
                    
                    st.write("**Confidence:**")
                    st.write(defect.get("confidence", "unknown").title())
                
                with col2:
                    st.write("**Reasoning:**")
                    st.write(defect.get("reasoning", "Not provided"))
                    
                    st.write("**Recommended Action:**")
                    st.info(defect.get("recommended_action", "Not provided"))
    
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
            "Inspector (Qwen2-VL)": [
                inspector.get("object_identified", "N/A"),
                inspector.get("overall_condition", "N/A"),
                len(inspector.get("defects", [])),
                inspector.get("overall_confidence", "N/A")
            ],
            "Auditor (Llama 3.2)": [
                auditor.get("object_identified", "N/A"),
                auditor.get("overall_condition", "N/A"),
                len(auditor.get("defects", [])),
                auditor.get("overall_confidence", "N/A")
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Download report
    if results.get("report_path"):
        with open(results["report_path"], "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name=Path(results["report_path"]).name,
                mime="application/pdf",
                use_container_width=True
            )


# ============================================================================
# CHAT INTERFACE
# ============================================================================

def chat_interface(results: Dict[str, Any]):
    """Interactive chat interface for follow-up questions."""
    st.subheader("💬 Ask Questions About This Inspection")
    
    # Get chat history
    session_id = st.session_state.chat_session_id
    history = get_session_history(session_id)
    
    # Display chat history
    for msg in history.messages:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the inspection results..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to history
        from langchain_core.messages import HumanMessage, AIMessage
        history.add_message(HumanMessage(content=prompt))
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create context from results
                    context = {
                        "verdict": results.get("safety_verdict"),
                        "defects": results.get("consensus", {}).get("combined_defects", []),
                        "explanation": results.get("explanation")
                    }
                    
                    # Simple response generation (in production, use LLM)
                    response = f"Based on the inspection results: {results.get('explanation', 'No information available.')}"
                    
                    st.write(response)
                    
                    # Add to history
                    history.add_message(AIMessage(content=response))
                
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
                    logger.error(f"Chat response generation failed: {e}")


# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

def analytics_dashboard():
    """Display analytics dashboard."""
    st.header("📊 Analytics Dashboard")
    
    try:
        repo = InspectionRepository()
        stats = repo.get_defect_statistics()
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Total Inspections",
            stats.get("total_inspections", 0)
        )
        col2.metric(
            "Agreement Rate",
            f"{stats.get('agreement_rate', 0):.1%}"
        )
        col3.metric(
            "Avg Processing Time",
            f"{stats.get('avg_processing_time', 0):.2f}s"
        )
        col4.metric(
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
                    title="Distribution of Defect Types"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No defect data available yet.")
        
        with col2:
            st.subheader("Verdicts Distribution")
            verdict_counts = stats.get("verdict_counts", {})
            if verdict_counts:
                fig = px.bar(
                    x=list(verdict_counts.keys()),
                    y=list(verdict_counts.values()),
                    title="Inspection Verdicts",
                    labels={"x": "Verdict", "y": "Count"},
                    color=list(verdict_counts.keys()),
                    color_discrete_map={
                        "SAFE": "green",
                        "UNSAFE": "red",
                        "REQUIRES_HUMAN_REVIEW": "orange"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No verdict data available yet.")
        
        # Recent inspections
        st.subheader("Recent Inspections")
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
            
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No inspection history yet. Run your first inspection to see data here.")
    
    except Exception as e:
        st.error(f"Failed to load analytics: {e}")
        logger.error(f"Analytics dashboard error: {e}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application."""
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🔍 Vision Inspection")
        st.caption(f"v1.0.0 | {config.environment.upper()}")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Inspection", "📊 Analytics", "⚙️ Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # System info
        with st.expander("System Information"):
            st.write(f"**Inspector:** {config.vlm_inspector_model.split('/')[-1]}")
            st.write(f"**Auditor:** {config.vlm_auditor_model.split('/')[-1]}")
            st.write(f"**LangSmith:** {'✅ Enabled' if config.langsmith_enabled else '❌ Disabled'}")
        
        # Session info
        st.caption(f"Session: {st.session_state.chat_session_id[:8]}")
    
    # Main content
    if page == "🏠 Inspection":
        inspection_page()
    elif page == "📊 Analytics":
        analytics_dashboard()
    else:
        settings_page()


def inspection_page():
    """Inspection page."""
    st.title("🔍 Visual Inspection System")
    st.caption("AI-powered damage detection and safety analysis")
    
    # File upload
    st.subheader("Upload Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=config.allowed_extensions_list,
            help=f"Maximum file size: {config.max_file_size_mb}MB"
        )
    
    with col2:
        criticality = st.selectbox(
            "Criticality Level",
            options=["low", "medium", "high"],
            index=1,
            help="Safety criticality of the component being inspected"
        )
        
        domain = st.text_input(
            "Domain (optional)",
            placeholder="e.g., mechanical_fasteners, medical, automotive",
            help="Provide context about the inspection domain"
        )
    
    user_notes = st.text_area(
        "Additional Notes (optional)",
        placeholder="Any specific concerns or context about this inspection..."
    )
    
    # Display uploaded image
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Analyze button
    if st.button("🔍 Analyze Image", type="primary", disabled=uploaded_file is None or st.session_state.processing):
        if uploaded_file:
            # Save file
            image_path = save_uploaded_file(uploaded_file)
            
            if image_path:
                st.session_state.processing = True
                st.session_state.current_image_path = str(image_path)
                
                # Progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Run inspection
                    status_text.text("🔄 Initializing inspection...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    status_text.text("🔍 Inspector analyzing image...")
                    progress_bar.progress(30)
                    
                    # Run workflow
                    results = run_inspection(
                        image_path=str(image_path),
                        criticality=criticality,
                        domain=domain or None,
                        user_notes=user_notes or None
                    )
                    
                    progress_bar.progress(70)
                    status_text.text("✅ Generating report...")
                    
                    # Generate PDF report
                    report_path = generate_report(results)
                    results["report_path"] = str(report_path)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Inspection complete!")
                    time.sleep(1)
                    
                    # Clear progress
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Store results
                    st.session_state.inspection_results = results
                    
                    # Success message
                    st.success("Inspection completed successfully!")
                    
                except Exception as e:
                    logger.error(f"Inspection failed: {e}", exc_info=True)
                    st.error(f"Inspection failed: {e}")
                    
                    if config.verbose_errors:
                        st.exception(e)
                
                finally:
                    st.session_state.processing = False
    
    # Display results
    if st.session_state.inspection_results:
        st.divider()
        st.header("Inspection Results")
        
        display_inspection_results(st.session_state.inspection_results)
        
        # Chat interface
        if config.enable_chat_memory:
            st.divider()
            chat_interface(st.session_state.inspection_results)


def settings_page():
    """Settings page."""
    st.title("⚙️ Settings")
    
    st.subheader("Model Configuration")
    st.info(f"**Inspector:** {config.vlm_inspector_model}")
    st.info(f"**Auditor:** {config.vlm_auditor_model}")
    st.info(f"**Explainer:** {config.explainer_model}")
    
    st.subheader("Safety Settings")
    st.write(f"**Confidence Threshold:** {config.confidence_threshold}")
    st.write(f"**Max Auto Defects:** {config.max_defects_auto}")
    st.write(f"**VLM Agreement Required:** {config.vlm_agreement_required}")
    
    st.subheader("System Information")
    st.write(f"**Environment:** {config.environment.upper()}")
    st.write(f"**Database:** {config.database_path}")
    st.write(f"**LangSmith:** {'Enabled' if config.langsmith_enabled else 'Disabled'}")
    
    if st.button("Clear Session Data"):
        st.session_state.clear()
        init_session_state()
        st.success("Session data cleared!")
        st.rerun()


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()