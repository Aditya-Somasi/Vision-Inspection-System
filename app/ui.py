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

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import config, UPLOAD_DIR, REPORT_DIR
from src.orchestration import run_inspection
from src.database import InspectionRepository
from src.chat_memory import get_memory_manager, get_session_history
from utils.logger import setup_logger
from src.reporting import generate_report

# Configure page
st.set_page_config(
    page_title=config.app_title,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Verdict banners */
    .verdict-safe {
        background: linear-gradient(135deg, #10b981, #059669);
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        color: white;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    .verdict-unsafe {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        color: white;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
    }
    .verdict-review {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        color: white;
        font-size: 1.25rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Defect severity badges */
    .severity-critical {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .severity-moderate {
        background: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .severity-cosmetic {
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Improved spacing */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Setup logger
logger = setup_logger(__name__, level=config.log_level, component="UI")


# ============================================================================
# SESSION STATE
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
    """Save uploaded file to disk with validation."""
    try:
        # Validate file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > config.max_file_size_mb:
            st.error(
                f"❌ File too large: {file_size_mb:.1f}MB "
                f"(maximum: {config.max_file_size_mb}MB)"
            )
            return None
        
        # Validate extension
        file_ext = Path(uploaded_file.name).suffix[1:].lower()
        if file_ext not in config.allowed_extensions_list:
            st.error(
                f"❌ Invalid file type: .{file_ext} "
                f"(allowed: {', '.join(config.allowed_extensions_list)})"
            )
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
        st.error(f"❌ Failed to upload file: {e}")
        return None


def display_verdict_banner(verdict: str):
    """Display styled verdict banner."""
    if verdict == "SAFE":
        st.markdown(
            '<div class="verdict-safe">✅ SAFE - No Critical Defects Detected</div>',
            unsafe_allow_html=True
        )
    elif verdict == "UNSAFE":
        st.markdown(
            '<div class="verdict-unsafe">🚫 UNSAFE - Critical Defect Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="verdict-review">⚠️ REQUIRES HUMAN REVIEW</div>',
            unsafe_allow_html=True
        )


def display_inspection_results(results: Dict[str, Any]):
    """Display inspection results in enhanced UI."""
    verdict = results.get("safety_verdict", {})
    consensus = results.get("consensus", {})
    
    # Verdict banner
    display_verdict_banner(verdict.get("verdict", "UNKNOWN"))
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    defects = consensus.get("combined_defects", [])
    critical_count = sum(
        1 for d in defects if d.get("safety_impact") == "CRITICAL"
    )
    
    with col1:
        st.metric("Total Defects", len(defects))
    with col2:
        st.metric("Critical Defects", critical_count)
    with col3:
        st.metric(
            "Models Agree",
            "Yes ✓" if consensus.get("models_agree") else "No ⚠",
            delta="High confidence" if consensus.get("models_agree") else None,
            delta_color="off"
        )
    with col4:
        st.metric(
            "Processing Time",
            f"{results.get('processing_time', 0):.2f}s"
        )
    
    st.divider()
    
    # Explanation
    st.subheader("📝 Analysis Summary")
    explanation = results.get("explanation", "No explanation available.")
    st.info(explanation)
    
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
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with open(results["report_path"], "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=Path(results["report_path"]).name,
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
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
                    verdict = results.get("safety_verdict", {})
                    defects = results.get("consensus", {}).get("combined_defects", [])
                    
                    # Build contextual response
                    response = (
                        f"Based on the inspection:\n\n"
                        f"**Verdict:** {verdict.get('verdict', 'UNKNOWN')}\n\n"
                        f"**Defects Found:** {len(defects)}\n\n"
                        f"{results.get('explanation', 'No details available.')}"
                    )
                    
                    st.markdown(response)
                    
                    # Add to history
                    history.add_message(AIMessage(content=response))
                
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
                    logger.error(f"Chat response generation failed: {e}")


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
        with st.expander("ℹ️ System Information"):
            st.write(f"**Inspector:** {config.vlm_inspector_model.split('/')[-1]}")
            st.write(f"**Auditor:** {config.vlm_auditor_model.split('/')[-1]}")
            langsmith_status = "✅ Enabled" if config.langsmith_enabled else "❌ Disabled"
            st.write(f"**LangSmith:** {langsmith_status}")
        
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
    """Inspection page with enhanced UX."""
    st.title("🔍 Visual Inspection System")
    st.caption("AI-powered damage detection and safety analysis")
    
    # File upload section
    st.subheader("📤 Upload Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=config.allowed_extensions_list,
            help=f"Maximum file size: {config.max_file_size_mb}MB",
            label_visibility="collapsed"
        )
    
    with col2:
        criticality = st.selectbox(
            "🎚️ Criticality Level",
            options=["low", "medium", "high"],
            index=1,
            help="Safety criticality of the component being inspected"
        )
        
        domain = st.text_input(
            "🏷️ Domain (optional)",
            placeholder="e.g., mechanical_fasteners",
            help="Provide context about the inspection domain"
        )
    
    user_notes = st.text_area(
        "📝 Additional Notes (optional)",
        placeholder="Any specific concerns or context about this inspection...",
        height=80
    )
    
    # Preview uploaded image
    if uploaded_file:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button(
            "🔍 Analyze Image",
            type="primary",
            disabled=uploaded_file is None or st.session_state.processing,
            use_container_width=True
        )
    
    if analyze_clicked and uploaded_file:
        # Save file
        image_path = save_uploaded_file(uploaded_file)
        
        if image_path:
            st.session_state.processing = True
            st.session_state.current_image_path = str(image_path)
            
            # Progress
            progress_container = st.empty()
            
            with progress_container.container():
                progress_bar = st.progress(0)
                status_text = st.status("Initializing inspection...", expanded=True)
                
                try:
                    with status_text:
                        st.write("🔄 Starting inspection workflow...")
                        progress_bar.progress(10)
                        time.sleep(0.3)
                        
                        st.write("🔍 Inspector analyzing image...")
                        progress_bar.progress(30)
                        
                        # Run workflow
                        results = run_inspection(
                            image_path=str(image_path),
                            criticality=criticality,
                            domain=domain or None,
                            user_notes=user_notes or None
                        )
                        
                        progress_bar.progress(70)
                        st.write("📄 Generating report...")
                        
                        # Generate PDF report
                        report_path = generate_report(results)
                        results["report_path"] = str(report_path)
                        
                        progress_bar.progress(100)
                        st.write("✅ Inspection complete!")
                        time.sleep(0.5)
                    
                    # Clear progress
                    progress_container.empty()
                    
                    # Store results
                    st.session_state.inspection_results = results
                    
                    # Success toast
                    st.toast("✅ Inspection completed successfully!", icon="✅")
                    
                except Exception as e:
                    logger.error(f"Inspection failed: {e}", exc_info=True)
                    st.error(f"❌ Inspection failed: {e}")
                    
                    if config.verbose_errors:
                        st.exception(e)
                
                finally:
                    st.session_state.processing = False
    
    # Display results
    if st.session_state.inspection_results:
        st.divider()
        st.header("📋 Inspection Results")
        
        display_inspection_results(st.session_state.inspection_results)
        
        # Chat interface
        if config.enable_chat_memory:
            st.divider()
            chat_interface(st.session_state.inspection_results)


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
        st.session_state.clear()
        init_session_state()
        st.success("✅ Session data cleared!")
        st.rerun()


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
