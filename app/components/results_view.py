"""
Results view components for per-image verdict cards and session aggregation.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.components.verdict_display import (
    render_verdict_banner,
    render_confidence_metrics,
    render_defect_summary,
    render_gate_results
)
from app.components.decision_support import render_decision_support
from app.services.session_manager import get_state


def render_image_verdict_card(image_id: str, image_info: Dict[str, Any], results: Optional[Dict[str, Any]] = None):
    """
    Render a verdict card for a single image result.
    
    Args:
        image_id: Unique image identifier
        image_info: Image metadata (may contain inspection_result)
        results: Optional inspection results dict (if None, use image_info["inspection_result"])
    """
    # Get results from parameter or from image_info
    if results is None:
        results = image_info.get("inspection_result", {})
    
    if not results:
        # No results yet
        st.info(f"⏳ {image_info.get('filename', 'unknown')} - No results yet")
        return
    
    verdict = results.get("safety_verdict", {})
    consensus = results.get("consensus", {})
    defects = consensus.get("combined_defects", [])
    defect_count = len(defects)
    
    filename = image_info.get("filename", "unknown")
    verdict_value = verdict.get("verdict", "UNKNOWN")
    
    with st.expander(
        f"📷 {filename} - {verdict_value}",
        expanded=True
    ):
        # Verdict banner
        gate_results = verdict.get("defect_summary", {}).get("all_gate_results", [])
        render_verdict_banner(verdict_value, defect_count, gate_results)
        
        st.divider()
        
        # Confidence metrics
        inspector_result = results.get("inspector_result", {})
        auditor_result = results.get("auditor_result", {})
        agreement_score = consensus.get("agreement_score", 0.5)
        processing_time = results.get("processing_time", 0)
        
        render_confidence_metrics(
            inspector_confidence=inspector_result.get("overall_confidence", "medium"),
            auditor_confidence=auditor_result.get("overall_confidence", "medium"),
            agreement_score=agreement_score,
            processing_time=processing_time
        )
        
        st.divider()
        
        # Defect summary
        render_defect_summary(defects)
        
        # Gate results (if available)
        if gate_results:
            st.divider()
            render_gate_results(gate_results)
        
        # Decision support
        render_decision_support(results)


def render_session_summary():
    """Render aggregated session-level verdict summary."""
    session_results = get_state("session_results", {})
    uploaded_images = get_state("uploaded_images", [])
    
    if not session_results and not uploaded_images:
        return
    
    st.header("📊 Session Summary")
    
    # Aggregate metrics
    total_images = session_results.get("total_images", len(uploaded_images))
    completed_images = session_results.get("completed_images", 0)
    failed_images = session_results.get("failed_images", 0)
    aggregate_verdict = session_results.get("aggregate_verdict", "UNKNOWN")
    total_defects = session_results.get("total_defects", 0)
    critical_defects = session_results.get("critical_defects", 0)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Completed", completed_images)
    with col3:
        st.metric("Total Defects", total_defects)
    with col4:
        st.metric("Critical Defects", critical_defects)
    
    # Aggregate verdict banner
    verdict_emoji = {
        "SAFE": "✅",
        "UNSAFE": "🚫",
        "REQUIRES_HUMAN_REVIEW": "⚠️",
        "MIXED": "🔀"
    }.get(aggregate_verdict, "❓")
    
    st.markdown(f"""
    <div style="background-color: #f0f9ff; border: 2px solid #3b82f6; padding: 1rem; border-radius: 0.5rem; text-align: center; margin: 1rem 0;">
        <h2 style="margin:0; color: #1f2937;">
            {verdict_emoji} Aggregate Verdict: {aggregate_verdict}
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()


def render_image_comparison_grid():
    """Render side-by-side comparison view for multiple images."""
    uploaded_images = get_state("uploaded_images", [])
    image_results = get_state("image_results", {})
    
    # Filter to completed images only
    completed_images = [
        img for img in uploaded_images
        if img.get("status") == "complete" and img.get("image_id") in image_results
    ]
    
    if not completed_images:
        return
    
    st.subheader("🖼️ Image Comparison View")
    
    # Display in grid (2 columns for side-by-side)
    for idx in range(0, len(completed_images), 2):
        cols = st.columns(2)
        
        for i, col in enumerate(cols):
            if idx + i < len(completed_images):
                with col:
                    img_info = completed_images[idx + i]
                    image_id = img_info.get("image_id")
                    results = image_results.get(image_id, {})
                    
                    verdict = results.get("safety_verdict", {}).get("verdict", "UNKNOWN")
                    
                    # Display image thumbnail
                    filepath = Path(img_info.get("filepath"))
                    if filepath.exists():
                        try:
                            from PIL import Image
                            img = Image.open(filepath)
                            img.thumbnail((400, 400))
                            st.image(img, caption=f"{img_info.get('filename')} - {verdict}")
                        except Exception:
                            st.info(f"📷 {img_info.get('filename')}")


def render_gates_dashboard():
    """Render enhanced safety gates visualization dashboard."""
    uploaded_images = get_state("uploaded_images", [])
    image_results = get_state("image_results", {})
    
    completed_images = [
        img for img in uploaded_images
        if img.get("status") == "complete" and img.get("image_id") in image_results
    ]
    
    if not completed_images:
        return
    
    st.subheader("🔒 Safety Gates Dashboard")
    
    # Aggregate gate statistics
    gate_stats = {}
    
    for img_info in completed_images:
        image_id = img_info.get("image_id")
        results = image_results.get(image_id, {})
        verdict = results.get("safety_verdict", {})
        gate_results = verdict.get("defect_summary", {}).get("all_gate_results", [])
        
        for gate in gate_results:
            gate_id = gate.get("gate_id")
            passed = gate.get("passed", True)
            
            if gate_id not in gate_stats:
                gate_stats[gate_id] = {"passed": 0, "failed": 0, "name": gate.get("display_name", gate_id)}
            
            if passed:
                gate_stats[gate_id]["passed"] += 1
            else:
                gate_stats[gate_id]["failed"] += 1
    
    # Display gate statistics
    if gate_stats:
        for gate_id, stats in gate_stats.items():
            total = stats["passed"] + stats["failed"]
            passed_pct = (stats["passed"] / total * 100) if total > 0 else 0
            
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.write(f"**{stats['name']}**")
            with col2:
                st.metric("Pass Rate", f"{passed_pct:.0f}%")
            with col3:
                st.caption(f"✅ {stats['passed']} passed, ❌ {stats['failed']} failed")


def render_results_review_tab():
    """
    Main function to render the entire Results & Review tab.
    Shows per-image verdict cards, session summary, and comparison views.
    """
    st.header("📋 Results & Review")
    
    uploaded_images = get_state("uploaded_images", [])
    image_results = get_state("image_results", {})
    
    if not uploaded_images:
        st.info("📷 No images in session. Upload and inspect images first.")
        return
    
    # Sync image_results with uploaded_images (results may be keyed by image_id)
    # Update uploaded_images with results for easier access
    for img in uploaded_images:
        img_id = img.get("image_id")
        if img_id and img_id in image_results:
            img["inspection_result"] = image_results[img_id]
        elif not img.get("inspection_result"):
            # Try to find by filepath as fallback
            img_path = img.get("filepath")
            for result_id, result in image_results.items():
                if result.get("image_path") == img_path:
                    img["image_id"] = result_id
                    img["inspection_result"] = result
                    break
    
    # Session summary
    render_session_summary()
    
    st.divider()
    
    # Per-image verdict cards
    st.subheader("📷 Per-Image Results")
    
    completed_images = [
        img for img in uploaded_images
        if img.get("status") == "complete" and img.get("image_id") in image_results
    ]
    
    if completed_images:
        for img_info in completed_images:
            image_id = img_info.get("image_id", "unknown")
            # Results may be in image_info["inspection_result"] or image_results dict
            results = img_info.get("inspection_result") or image_results.get(image_id, {})
            render_image_verdict_card(image_id, img_info, results)
            st.divider()
    else:
        st.info("⏳ No completed inspections yet. Check the 'Live Inspection' tab for progress.")
    
    # Comparison view
    if len(completed_images) > 1:
        st.divider()
        render_image_comparison_grid()
    
    # Gates dashboard
    if completed_images:
        st.divider()
        render_gates_dashboard()
    
    # PDF report download (if available)
    session_results = get_state("session_results", {})
    session_report_path = session_results.get("session_report_path")
    
    if session_report_path and Path(session_report_path).exists():
        st.divider()
        st.subheader("📄 Session Report")
        
        with open(session_report_path, "rb") as f:
            st.download_button(
                label="📥 Download Combined PDF Report",
                data=f,
                file_name=Path(session_report_path).name,
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
