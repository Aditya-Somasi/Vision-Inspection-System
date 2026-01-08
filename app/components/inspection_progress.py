"""
Inspection progress components for per-image progress tracking.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from app.services.session_manager import get_state, set_state


def render_per_image_progress_card(image_id: str, image_info: Dict[str, Any]):
    """
    Render progress card for a single image during inspection.
    
    Args:
        image_id: Unique image identifier
        image_info: Image metadata dictionary (may contain inspection_result)
    """
    status = image_info.get("status", "uploaded")
    filename = image_info.get("filename", "unknown")
    result = image_info.get("inspection_result") or {}
    
    with st.expander(f"📷 {filename} - {status.title()}", expanded=status == "processing"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if status == "processing":
                st.info("🔄 Processing in progress...")
                progress_bar = st.progress(0.5)
                st.caption("Inspector → Auditor → Consensus → Safety Evaluation")
            elif status == "complete":
                if result:
                    verdict = result.get("safety_verdict", {}).get("verdict", "UNKNOWN")
                    verdict_emoji = {"SAFE": "✅", "UNSAFE": "🚫", "REQUIRES_HUMAN_REVIEW": "⚠️"}.get(verdict, "❓")
                    st.success(f"{verdict_emoji} {verdict}")
                    
                    defect_count = len(result.get("consensus", {}).get("combined_defects", []))
                    st.caption(f"Defects found: {defect_count}")
                else:
                    st.success("✅ Complete")
            elif status == "failed":
                st.error("❌ Inspection failed")
                error_msg = image_info.get("error", "Unknown error")
                st.caption(f"Error: {error_msg}")
            else:
                st.info("⏳ Waiting to start...")
        
        with col2:
            if status == "processing":
                if st.button("⏸️ Cancel", key=f"cancel_{image_id}"):
                    set_state("session_status", "idle")
                    # Update image status
                    uploaded_images = get_state("uploaded_images", [])
                    for img in uploaded_images:
                        if img.get("image_id") == image_id:
                            img["status"] = "uploaded"
                            break
                    set_state("uploaded_images", uploaded_images)
                    st.rerun()


def render_session_progress_dashboard():
    """Render aggregate progress dashboard for the entire session."""
    uploaded_images = get_state("uploaded_images", [])
    
    if not uploaded_images:
        return
    
    st.subheader("📊 Session Progress")
    
    total = len(uploaded_images)
    completed = sum(1 for img in uploaded_images if img.get("status") == "complete")
    processing = sum(1 for img in uploaded_images if img.get("status") == "processing")
    failed = sum(1 for img in uploaded_images if img.get("status") == "failed")
    pending = total - completed - processing - failed
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("Completed", completed, delta=None)
    with col3:
        st.metric("Processing", processing, delta=None)
    with col4:
        st.metric("Failed", failed, delta=None)
    with col5:
        st.metric("Pending", pending, delta=None)
    
    # Overall progress bar
    if total > 0:
        progress = (completed / total) if total > 0 else 0
        st.progress(progress)
        st.caption(f"Overall progress: {int(progress * 100)}% ({completed}/{total} images)")


def render_streaming_status():
    """Render real-time status updates section."""
    session_status = get_state("session_status", "idle")
    
    if session_status == "processing":
        st.info("🔄 Inspection session in progress. Check individual image cards below for detailed status.")
    elif session_status == "complete":
        st.success("✅ All inspections completed!")
    elif session_status == "error":
        st.error("❌ Session error occurred. Check individual images for details.")
    elif session_status == "idle":
        st.info("ℹ️ Ready to start inspection. Click 'Start Inspection' in the Upload tab.")


def render_live_inspection_tab():
    """
    Main function to render the entire Live Inspection tab.
    Shows per-image progress cards and aggregate dashboard.
    """
    st.header("🔄 Live Inspection")
    
    # Session-level status
    render_streaming_status()
    
    st.divider()
    
    # Aggregate dashboard
    render_session_progress_dashboard()
    
    st.divider()
    
    # Per-image progress cards
    uploaded_images = get_state("uploaded_images", [])
    
    if uploaded_images:
        st.subheader("📷 Per-Image Status")
        
        for img_info in uploaded_images:
            render_per_image_progress_card(
                img_info.get("image_id"),
                img_info
            )
    else:
        st.info("📷 No images in session. Upload images in the 'Upload & Configure' tab.")
