"""
Sidebar components for Vision Inspection System.
Enhanced sidebar with system status, active session info, and quick actions.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from app.services.session_manager import get_state, set_state
from utils.config import config


def render_system_status():
    """Render system status indicators in sidebar."""
    st.markdown("#### ┌─ SYSTEM STATUS ─────────────┐")
    
    # Inspector status
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<span class="status-online">●</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Inspector**")
        st.caption(config.vlm_inspector_model.split('/')[-1][:20] + "...")
    
    # Auditor status
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<span class="status-online">●</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Auditor**")
        st.caption(config.vlm_auditor_model.split('/')[-1][:20] + "...")
    
    # Database status
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown('<span class="status-online">●</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Database**")
        st.caption("Connected")
    
    # LangSmith status
    col1, col2 = st.columns([1, 3])
    with col1:
        langsmith_status = "status-online" if config.langsmith_enabled else "status-offline"
        st.markdown(f'<span class="{langsmith_status}">●</span>', unsafe_allow_html=True)
    with col2:
        st.markdown("**LangSmith**")
        langsmith_text = "Active" if config.langsmith_enabled else "Inactive"
        st.caption(langsmith_text)
    
    st.markdown("#### └──────────────────────────────┘")


def render_active_session():
    """Render active session information."""
    session_id = get_state("current_session_id") or get_state("session_id")
    session_status = get_state("session_status", "idle")
    uploaded_images = get_state("uploaded_images", [])
    session_start_time = get_state("session_start_time")
    
    if session_id or uploaded_images:
        st.markdown("#### ┌─ ACTIVE SESSION ─────────────┐")
        
        if session_id:
            st.markdown(f"**Session ID:** `{str(session_id)[:8]}`")
        
        if uploaded_images:
            total_images = len(uploaded_images)
            completed = sum(1 for img in uploaded_images if img.get("status") == "complete")
            processing = sum(1 for img in uploaded_images if img.get("status") == "processing")
            
            st.markdown(f"**Images:** {completed + processing}/{total_images}")
            st.markdown(f"**Status:** {session_status.title()}")
            
            if session_start_time:
                if isinstance(session_start_time, str):
                    from datetime import datetime as dt
                    session_start_time = dt.fromisoformat(session_start_time)
                elapsed = datetime.now() - session_start_time
                elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
                st.markdown(f"**Elapsed:** {elapsed_str}")
        else:
            st.markdown("**Status:** No active session")
        
        st.markdown("#### └──────────────────────────────┘")


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("#### ┌─ QUICK ACTIONS ─────────────┐")
    
    results = get_state("inspection_results")
    session_results = get_state("session_results")
    
    # Export Report button
    if results or session_results:
        if st.button("📥 Export Report", use_container_width=True):
            st.info("Export functionality - to be implemented")
    
    # Review Queue button (if needed)
    pending_review = get_state("pending_review")
    if pending_review:
        if st.button("📋 Review Queue", use_container_width=True):
            st.info("Review queue - to be implemented")
    
    # New Session button
    if st.button("🔄 New Session", use_container_width=True):
        set_state("current_session_id", None)
        set_state("uploaded_images", [])
        set_state("session_status", "idle")
        set_state("session_results", {})
        set_state("image_results", {})
        set_state("inspection_results", None)
        st.rerun()
    
    st.markdown("#### └──────────────────────────────┘")


def render_sidebar():
    """Render complete enhanced sidebar."""
    st.title("🔍 Vision Inspection")
    st.caption(f"v1.0.0 | {config.environment.upper()}")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("#### ┌─ NAVIGATION ─────────────────┐")
    page = st.radio(
        "Navigation",
        ["🏠 Inspection Session", "📊 Analytics", "📋 Inspection History", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    st.markdown("#### └──────────────────────────────┘")
    
    st.markdown("---")
    
    # System Status
    render_system_status()
    
    st.markdown("---")
    
    # Active Session
    render_active_session()
    
    st.markdown("---")
    
    # Quick Actions
    render_quick_actions()
    
    st.markdown("---")
    
    # Session ID footer
    session_id = get_state("current_session_id") or get_state("session_id") or get_state("chat_session_id") or "unknown"
    st.caption(f"Session: {str(session_id)[:8]}")
    
    return page
