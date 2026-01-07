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
    
    /* System status badges */
    .status-online {
        background: #dcfce7;
        color: #166534;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-offline {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Agent stream container */
    .agent-stream {
        background: #1f2937;
        color: #e5e7eb;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.875rem;
        line-height: 1.5;
        max-height: 300px;
        overflow-y: auto;
    }
    .agent-stream .thinking {
        color: #60a5fa;
    }
    .agent-stream .success {
        color: #34d399;
    }
    .agent-stream .warning {
        color: #fbbf24;
    }
    .agent-stream .error {
        color: #f87171;
    }
    
    /* Stepper */
    .stepper {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .stepper-step {
        flex: 1;
        text-align: center;
        position: relative;
    }
    .stepper-step::after {
        content: '';
        position: absolute;
        top: 15px;
        left: 50%;
        width: 100%;
        height: 2px;
        background: #e5e7eb;
    }
    .stepper-step:last-child::after {
        display: none;
    }
    .stepper-icon {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #e5e7eb;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.875rem;
        position: relative;
        z-index: 1;
    }
    .stepper-icon.active {
        background: #3b82f6;
        color: white;
    }
    .stepper-icon.complete {
        background: #10b981;
        color: white;
    }
    .stepper-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
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
        st.session_state.pending_review = None  # For human-in-loop
        st.session_state.pending_thread_id = None
        
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


def display_human_review_form(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Display human review form for interrupted inspections.
    
    Returns:
        Dict with decision and notes if submitted, None otherwise
    """
    from src.orchestration import resume_inspection
    
    st.warning("🔔 **Human Review Required**")
    st.markdown("The AI agents have flagged this inspection for human verification.")
    
    # Show context
    safety_verdict = results.get("safety_verdict", {})
    consensus = results.get("consensus", {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**AI Verdict:** {safety_verdict.get('verdict', 'UNKNOWN')}")
        st.caption(f"Triggered gates: {', '.join(safety_verdict.get('triggered_gates', []))}")
    with col2:
        agree_status = "✅ Yes" if consensus.get("models_agree") else "⚠️ No"
        st.info(f"**Models Agree:** {agree_status}")
        st.caption(f"Agreement score: {consensus.get('agreement_score', 0):.0%}")
    
    st.divider()
    
    # Decision form
    st.subheader("📋 Your Decision")
    
    with st.form("human_review_form"):
        decision = st.radio(
            "Select your decision:",
            options=["APPROVE", "REJECT", "MODIFY"],
            horizontal=True,
            help="APPROVE: Mark as safe. REJECT: Mark as unsafe. MODIFY: Keep AI verdict but add notes."
        )
        
        notes = st.text_area(
            "Reviewer Notes (optional):",
            placeholder="Add any observations or justification for your decision...",
            height=100
        )
        
        submitted = st.form_submit_button("✅ Submit Decision", type="primary", use_container_width=True)
        
        if submitted:
            thread_id = results.get("_thread_id")
            if thread_id:
                try:
                    with st.spinner("Resuming inspection workflow..."):
                        final_results = resume_inspection(thread_id, decision, notes)
                        st.session_state.inspection_results = final_results
                        st.session_state.pending_review = None
                        st.session_state.pending_thread_id = None
                        st.success(f"✅ Decision recorded: {decision}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to resume workflow: {e}")
                    logger.error(f"Resume failed: {e}", exc_info=True)
            else:
                # No thread_id means we need to just update the results in session
                st.session_state.inspection_results["human_decision"] = decision
                st.session_state.inspection_results["human_notes"] = notes
                if decision == "APPROVE":
                    st.session_state.inspection_results["safety_verdict"]["verdict"] = "SAFE"
                elif decision == "REJECT":
                    st.session_state.inspection_results["safety_verdict"]["verdict"] = "UNSAFE"
                st.session_state.pending_review = None
                st.success(f"✅ Decision recorded: {decision}")
                st.rerun()
    
    return None


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
        processing_time = results.get('processing_time') or 0
        st.metric(
            "Processing Time",
            f"{processing_time:.2f}s"
        )
    
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
    
    # Visual Evidence Section - Side by Side Comparison
    st.subheader("🖼️ Visual Evidence")
    image_path = results.get("image_path")
    
    # Try alternate location if main path fails (sometimes relative/absolute mix-up)
    if not image_path and results.get("inspector_result", {}).get("image_path"):
         image_path = results.get("inspector_result", {}).get("image_path")
         
    if image_path and Path(image_path).exists():
        try:
            from utils.image_utils import create_heatmap_overlay, create_side_by_side_comparison
            from utils.config import REPORT_DIR
            
            image_path = Path(image_path)
            
            # Create heatmap overlay
            heatmap_path = REPORT_DIR / f"heatmap_{image_path.stem}.jpg"
            if not heatmap_path.exists():
                create_heatmap_overlay(image_path, defects, heatmap_path)
            
            # Create comparison
            comparison_path = REPORT_DIR / f"comparison_{image_path.stem}.jpg"
            if not comparison_path.exists():
                create_side_by_side_comparison(
                    image_path, 
                    heatmap_path, 
                    comparison_path,
                    labels=("Original Input", "AI Analysis Layer")
                )
            
            # Display comparison
            if comparison_path.exists():
                st.image(
                    str(comparison_path), 
                    caption="📊 Original vs AI Analysis with Defect Heatmap",
                    use_container_width=True
                )
            else:
                # Fallback: show original and heatmap separately
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Input**")
                    st.image(str(image_path))
                with col2:
                    st.markdown("**AI Analysis Layer**")
                    if heatmap_path.exists():
                        st.image(str(heatmap_path))
                    else:
                        st.image(str(image_path))
        except Exception as e:
            logger.error(f"Failed to display comparison images: {e}")
            st.image(str(image_path), caption="📷 Uploaded Image")
    else:
        st.warning(f"Original image not available for comparison ({image_path or 'unknown path'})")
    
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
            "Inspector (Qwen2-VL)": [
                str(inspector.get("object_identified", "N/A")),
                str(inspector.get("overall_condition", "N/A")),
                str(len(inspector.get("defects", []))),
                str(inspector.get("overall_confidence", "N/A"))
            ],
            "Auditor (Qwen2-VL)": [
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
        
        # Action buttons
        col1, col2, col3 = st.columns([2, 2, 1])
        
        # Action buttons
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
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        history.add_message(HumanMessage(content=prompt))
        
        # Generate response using LLM
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build context from inspection results
                    verdict = results.get("safety_verdict", {})
                    consensus = results.get("consensus", {})
                    defects = consensus.get("combined_defects", [])
                    inspector = results.get("inspector_result", {})
                    auditor = results.get("auditor_result", {})
                    explanation = results.get("explanation", "No explanation available.")
                    
                    # Format defects for context
                    defect_summary = ""
                    if defects:
                        for i, d in enumerate(defects, 1):
                            defect_summary += f"\n{i}. {d.get('type', 'Unknown')} at {d.get('location', 'unknown location')}"
                            defect_summary += f" - Severity: {d.get('safety_impact', 'UNKNOWN')}"
                            defect_summary += f" - Reasoning: {d.get('reasoning', 'N/A')}"
                    else:
                        defect_summary = "No defects were found."
                    
                    # Build system context
                    system_context = f"""You are a helpful assistant for a vision inspection system. 
You have access to the following inspection results and should answer questions based ONLY on this data.

INSPECTION RESULTS:
- Final Verdict: {verdict.get('verdict', 'UNKNOWN')}
- Reason: {verdict.get('reason', 'N/A')}
- Requires Human Review: {verdict.get('requires_human', False)}
- Object Inspected: {inspector.get('object_identified', 'Unknown')}
- Overall Condition: {inspector.get('overall_condition', 'Unknown')}
- Models Agreement: {consensus.get('models_agree', 'Unknown')} (Score: {consensus.get('agreement_score', 0):.1%})

DEFECTS FOUND:
{defect_summary}

DETAILED ANALYSIS:
{explanation}

GUIDELINES:
- Answer questions based on the inspection data above
- Be specific and reference actual findings
- If asked about something not in the data, say you don't have that information
- For safety-critical questions, recommend consulting a qualified professional
- Be concise but helpful"""

                    # Use Groq for fast responses
                    from langchain_groq import ChatGroq
                    
                    chat_llm = ChatGroq(
                        api_key=config.groq_api_key,
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=500
                    )
                    
                    # Build messages for the LLM
                    messages = [
                        SystemMessage(content=system_context),
                        HumanMessage(content=prompt)
                    ]
                    
                    # Get response from LLM
                    response = chat_llm.invoke(messages)
                    response_text = response.content
                    
                    st.markdown(response_text)
                    
                    # Add to history
                    history.add_message(AIMessage(content=response_text))
                
                except Exception as e:
                    error_msg = f"I encountered an error: {e}. Please try rephrasing your question."
                    st.error(error_msg)
                    logger.error(f"Chat response generation failed: {e}", exc_info=True)


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
        # System info
        with st.expander("ℹ️ System Information", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<span class="status-online">● Inspector</span>', unsafe_allow_html=True)
                st.caption(config.vlm_inspector_model.split('/')[-1][:15] + "...")
            with col2:
                st.markdown('<span class="status-online">● Auditor</span>', unsafe_allow_html=True)
                st.caption(config.vlm_auditor_model.split('/')[-1][:15] + "...")
            
            st.divider()
            
            langsmith_status = "status-online" if config.langsmith_enabled else "status-offline"
            langsmith_text = "Tracking Active" if config.langsmith_enabled else "Tracking Inactive"
            st.markdown(f'<span class="{langsmith_status}">● LangSmith</span> {langsmith_text}', unsafe_allow_html=True)
            st.markdown(f'<span class="status-online">● Database</span> Connected', unsafe_allow_html=True)
        
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
        st.image(uploaded_file, caption="📷 Uploaded Image")
    
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
                        
                        # Create a link that opens the PDF - browser will handle viewing
                        try:
                            # Resolve to absolute path first to avoid relative/absolute mixing errors
                            abs_report_path = report_path.resolve()
                            rel_path = abs_report_path.relative_to(Path.cwd())
                            pdf_url = f"/{rel_path}"
                            
                            st.markdown(
                                f'<a href="file:///{abs_report_path}" target="_blank" style="text-decoration:none;">'
                                f'<button style="width:100%; padding:0.5rem 1rem; background-color:#0066cc; color:white; '
                                f'border:none; border-radius:0.25rem; cursor:pointer; font-size:1rem;">'
                                f'👁️ Open PDF Location</button></a>',
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            logger.warning(f"Could not create relative path for PDF link: {e}")
                            st.info("Click the path below to open file location")
                        
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
        results = st.session_state.inspection_results
        st.divider()
        st.header("📋 Inspection Results")
        
        # Show informational banner if human review was recommended (but don't block)
        if results.get("requires_human_review"):
            verdict = results.get("safety_verdict", {})
            st.warning(
                f"🔔 **Human Review Recommended** - AI Verdict: {verdict.get('verdict', 'UNKNOWN')}\n\n"
                f"Triggered gates: {', '.join(verdict.get('triggered_gates', ['None']))}"
            )
        
        # Display full inspection results
        display_inspection_results(results)
        
        # Chat interface
        if config.enable_chat_memory:
            st.divider()
            chat_interface(results)


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
