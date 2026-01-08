"""
Verdict Display Components for Vision Inspection System.
Renders verdict banners, confidence bars, and safety gate results.
"""

import streamlit as st
from typing import Dict, Any, Optional, List


def render_verdict_banner(verdict: str, defect_count: int = 0, gate_results: list = None):
    """
    Display styled verdict banner with enhanced all-clear for safe images.
    
    Args:
        verdict: Safety verdict (SAFE, UNSAFE, REQUIRES_HUMAN_REVIEW)
        defect_count: Number of defects found
        gate_results: List of gate evaluation results
    """
    verdict_upper = verdict.upper()
    
    # Special all-clear banner for safe items with no defects
    if verdict_upper == "SAFE" and defect_count == 0:
        st.markdown("""
        <div class="all-clear-banner">
            <h1 style="margin:0; font-size: 2.5rem;">✅ ALL CLEAR</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                No defects detected • All safety gates passed • Item is safe for use
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Standard verdict banner
    if verdict_upper == "SAFE":
        css_class = "verdict-safe"
        icon = "✅"
        message = "SAFE FOR USE"
    elif verdict_upper == "UNSAFE":
        css_class = "verdict-unsafe"
        icon = "🚫"
        message = "SAFETY CONCERNS DETECTED"
    else:
        css_class = "verdict-review"
        icon = "⚠️"
        message = "REQUIRES HUMAN REVIEW"
    
    st.markdown(f"""
    <div class="{css_class}">
        <span style="font-size: 2rem;">{icon}</span><br/>
        {message}
    </div>
    """, unsafe_allow_html=True)


def render_confidence_bar(label: str, confidence: str, numeric_value: float = None):
    """
    Display a confidence progress bar with percentage.
    
    Args:
        label: Label for the confidence metric
        confidence: Confidence level (high, medium, low) or percentage string
        numeric_value: Optional numeric value (0-1) for precise display
    """
    # Determine percentage
    if numeric_value is not None:
        percentage = int(numeric_value * 100)
    elif isinstance(confidence, str):
        confidence_map = {"high": 85, "medium": 60, "low": 35}
        percentage = confidence_map.get(confidence.lower(), 50)
    else:
        percentage = 50
    
    # Determine color class
    if percentage >= 70:
        color_class = "confidence-high"
    elif percentage >= 40:
        color_class = "confidence-medium"
    else:
        color_class = "confidence-low"
    
    st.markdown(f"""
    <div style="margin-bottom: 0.75rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-weight: 500;">{label}</span>
            <span style="font-weight: bold;">{percentage}%</span>
        </div>
        <div class="confidence-bar-container">
            <div class="confidence-bar {color_class}" style="width: {percentage}%;">
                {confidence.title() if isinstance(confidence, str) else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_gate_results(gate_results: list):
    """
    Display all safety gate evaluation results.
    
    Args:
        gate_results: List of gate result dictionaries
    """
    if not gate_results:
        return
    
    st.subheader("🔒 Safety Gate Evaluation")
    
    for gate in gate_results:
        passed = gate.get("passed", True)
        gate_name = gate.get("display_name", gate.get("gate_id", "Unknown Gate"))
        message = gate.get("message", "")
        
        emoji = "✅" if passed else "❌"
        badge_class = "gate-passed" if passed else "gate-failed"
        status_text = "PASSED" if passed else "FAILED"
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem; padding: 0.5rem; background: #f8fafc; border-radius: 0.5rem;">
            <span style="font-size: 1.25rem; margin-right: 0.75rem;">{emoji}</span>
            <div style="flex: 1;">
                <span style="font-weight: 600;">{gate_name}</span>
                <span style="color: #6b7280; margin-left: 0.5rem; font-size: 0.875rem;">{message}</span>
            </div>
            <span class="{badge_class}">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)


def render_confidence_metrics(inspector_confidence: str, auditor_confidence: str, 
                             agreement_score: float, processing_time: float = 0):
    """
    Render the confidence metrics section with inspector/auditor bars.
    
    Args:
        inspector_confidence: Inspector confidence level
        auditor_confidence: Auditor confidence level  
        agreement_score: Model agreement score (0-1)
        processing_time: Processing time in seconds
    """
    st.subheader("📊 Confidence Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_confidence_bar("Inspector Confidence", inspector_confidence)
        render_confidence_bar("Auditor Confidence", auditor_confidence)
    
    with col2:
        agreement_level = "high" if agreement_score >= 0.8 else "medium" if agreement_score >= 0.5 else "low"
        render_confidence_bar("Model Agreement", agreement_level, agreement_score)
        
        st.markdown(f"""
        <div style="margin-bottom: 0.75rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-weight: 500;">Processing Time</span>
                <span style="font-weight: bold;">{processing_time:.2f}s</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_defect_summary(defects: list):
    """
    Render defect count summary with severity breakdown.
    
    Args:
        defects: List of defect dictionaries
    """
    defect_count = len(defects)
    critical_count = sum(1 for d in defects if d.get("safety_impact") == "CRITICAL")
    moderate_count = sum(1 for d in defects if d.get("safety_impact") == "MODERATE")
    cosmetic_count = sum(1 for d in defects if d.get("safety_impact") == "COSMETIC")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Defects", defect_count)
    with col2:
        st.metric("🔴 Critical", critical_count)
    with col3:
        st.metric("🟡 Moderate", moderate_count)
    with col4:
        st.metric("🔵 Cosmetic", cosmetic_count)
