"""
Session state management for Vision Inspection System.
Centralizes all session state initialization and management.
"""

import streamlit as st
from datetime import datetime
import uuid
from typing import Any, Dict, Optional


def init_session_state():
    """Initialize all required session state variables."""
    defaults = {
        # Core state
        "session_id": str(uuid.uuid4()),
        "initialized": True,
        "inspection_results": None,
        "pending_review": None,
        "pending_thread_id": None,
        "current_image_path": None,
        "processing": False,  # For tracking if inspection is in progress
        
        # Chat state
        "chat_messages": [],
        "chat_session_id": str(uuid.uuid4()),
        "chat_history": [],  # Legacy format for compatibility
        
        # UI state
        "sidebar_expanded": True,
        "active_tab": "inspection",
        "theme": "light",
        "show_analytics": False,
        
        # Analytics
        "inspection_count": 0,
        "last_inspection_time": None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """Get a session state value safely."""
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """Set a session state value."""
    st.session_state[key] = value


def update_state(updates: Dict[str, Any]):
    """Update multiple session state values at once."""
    for key, value in updates.items():
        st.session_state[key] = value


def clear_inspection_state():
    """Clear inspection-related state for new inspection."""
    st.session_state.inspection_results = None
    st.session_state.pending_review = None
    st.session_state.pending_thread_id = None
    st.session_state.current_image_path = None


def reset_chat_state():
    """Reset chat state for new conversation."""
    st.session_state.chat_messages = []
    st.session_state.chat_session_id = str(uuid.uuid4())


def record_inspection(results: Optional[dict] = None):
    """Record that an inspection was completed."""
    st.session_state.inspection_count += 1
    st.session_state.last_inspection_time = datetime.now()
    if results:
        st.session_state.inspection_results = results


def get_session_summary() -> dict:
    """Get summary of current session for debugging."""
    return {
        "session_id": get_state("session_id"),
        "inspection_count": get_state("inspection_count", 0),
        "has_results": get_state("inspection_results") is not None,
        "chat_messages_count": len(get_state("chat_messages", [])),
        "last_inspection": get_state("last_inspection_time"),
    }
