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
from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, level=config.log_level, component="RESULTS_VIEW")



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
        
        # Ensure processing_time is not None - handle both None and missing key
        processing_time = results.get("processing_time")
        if processing_time is None:
            processing_time = 0
        
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
        
        # Analysis Summary
        st.divider()
        st.subheader("📝 Analysis Summary")
        explanation = results.get("explanation")
        if explanation:
            # Parse explanation into sections for better display
            from src.reporting.pdf_generator import parse_explanation_sections
            sections = parse_explanation_sections(explanation)
            
            # Display sections in a structured way
            if sections:
                # Show SUMMARY first if available
                if "SUMMARY" in sections and sections["SUMMARY"].strip():
                    st.markdown("#### 📋 Summary")
                    st.markdown(sections["SUMMARY"])
                    st.divider()
                
                # Show other key sections
                key_sections = ["KEY TAKEAWAYS", "RECOMMENDATIONS", "FINAL RECOMMENDATION", 
                               "REASONING CHAINS", "INSPECTOR ANALYSIS", "AUDITOR VERIFICATION", 
                               "COUNTERFACTUAL"]
                
                for section_name in key_sections:
                    if section_name in sections and sections[section_name].strip():
                        display_name = section_name.replace("_", " ").title()
                        st.markdown(f"#### {display_name}")
                        st.markdown(sections[section_name])
                        st.divider()
                
                # Show any remaining sections
                shown_sections = set(["SUMMARY"] + key_sections)
                remaining = set(sections.keys()) - shown_sections
                for section_name in remaining:
                    if sections[section_name].strip():
                        display_name = section_name.replace("_", " ").title()
                        st.markdown(f"#### {display_name}")
                        st.markdown(sections[section_name])
                        st.divider()
            else:
                # Fallback: display raw explanation
                st.markdown(explanation)
        else:
            st.warning("⚠️ Analysis summary pending system completion.")
        
        # 3-Panel Image Comparison
        st.divider()
        st.subheader("🖼️ Visual Evidence (3-Panel View)")
        
        image_path = results.get("image_path") or image_info.get("filepath")
        if not image_path:
            image_path = results.get("inspector_result", {}).get("image_path")
        
        if image_path and Path(image_path).exists():
            try:
                from utils.image_utils import create_heatmap_overlay, draw_bounding_boxes
                from utils.config import REPORT_DIR
                
                image_path = Path(image_path)
                
                # Create heatmap overlay
                heatmap_path = REPORT_DIR / f"heatmap_{image_path.stem}.jpg"
                logger.info(f"DEBUG: [Results View] Heatmap path: {heatmap_path}, exists: {heatmap_path.exists()}")
                if not heatmap_path.exists():
                    logger.info(f"DEBUG: [Results View] Creating heatmap for {len(defects)} defects")
                    create_heatmap_overlay(image_path, defects, heatmap_path)
                    logger.info(f"DEBUG: [Results View] Heatmap created, exists now: {heatmap_path.exists()}")
                else:
                    logger.info(f"DEBUG: [Results View] Using existing heatmap file")
                
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
                    st.image(str(image_path), width='stretch')
                
                with col2:
                    st.markdown("**2. Defect Heatmap**")
                    if heatmap_path.exists():
                        st.image(str(heatmap_path), width='stretch')
                    else:
                        st.image(str(image_path), width='stretch')
                
                with col3:
                    st.markdown("**3. Numbered Markers**")
                    if annotated_path.exists():
                        st.image(str(annotated_path), width='stretch')
                    else:
                        st.image(str(image_path), width='stretch')
                
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
                st.warning(f"Could not generate comparison images: {e}")
                if image_path:
                    st.image(str(image_path), caption="📷 Original Image", width='stretch')
        else:
            st.warning(f"Original image not available for comparison ({image_path or 'unknown path'})")
        
        # PDF Report Section
        report_path = results.get("report_path")
        logger.info(f"DEBUG: [Results View] Image {image_id} - report_path: {report_path}")
        if report_path and Path(report_path).exists():
            logger.info(f"DEBUG: [Results View] PDF exists at: {Path(report_path).absolute()}")
            st.divider()
            st.subheader("📄 Inspection Report")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name=Path(report_path).name,
                        mime="application/pdf",
                        width='stretch',
                        type="primary"
                    )
            
            with col2:
                # Button to open PDF in new tab - use data URI (works reliably in Chrome)
                import base64
                with open(report_path, "rb") as f:
                    pdf_bytes = f.read()
                    pdf_base64 = base64.b64encode(pdf_bytes).decode()
                
                # Use data URI which works reliably in Chrome
                pdf_data_uri = f"data:application/pdf;base64,{pdf_base64}"
                
                st.markdown(
                    f'''
                    <a href="{pdf_data_uri}" target="_blank" style="text-decoration: none; display: block;">
                        <button style="background-color: #0ea5e9; color: white; padding: 0.5rem 1rem; 
                        border: none; border-radius: 0.25rem; cursor: pointer; width: 100%;">
                        🔗 Open PDF in New Tab</button>
                    </a>
                    ''',
                    unsafe_allow_html=True
                )
            
            st.caption(f"📁 {Path(report_path).name}")
            st.caption(f"📍 {Path(report_path).absolute()}")
            
            # PDF Preview - Use download button for best cross-browser experience
            st.divider()
            st.info("💡 **Tip**: Use the '📥 Download PDF Report' button above to view the full report. PDF preview in browsers can be limited by security restrictions.")
            
            # Alternative: Try simple embed (works in some browsers)
            st.subheader("👁️ Quick PDF Preview")
            
            try:
                with open(report_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                
                # Simple embed tag (works better than iframe for some browsers)
                pdf_embed = f'<embed src="data:application/pdf;base64,{base64_pdf}" type="application/pdf" width="100%" height="600px" />'
                st.markdown(pdf_embed, unsafe_allow_html=True)
                st.caption("If preview doesn't display, please use the Download button above.")
            except Exception as e:
                st.warning(f"PDF preview unavailable. Please download the PDF to view it. ({str(e)[:100]})")


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
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with open(session_report_path, "rb") as f:
                st.download_button(
                    label="📥 Download Combined PDF Report",
                    data=f,
                    file_name=Path(session_report_path).name,
                    mime="application/pdf",
                    width='stretch',
                    type="primary"
                )
        
        with col2:
            # Button to open PDF in new tab
            report_url = f"file:///{Path(session_report_path).absolute()}"
            st.markdown(
                f'<a href="{report_url}" target="_blank">'
                '<button style="background-color: #0ea5e9; color: white; padding: 0.5rem 1rem; '
                'border: none; border-radius: 0.25rem; cursor: pointer; width: 100%;">'
                '🔗 Open PDF in New Tab</button></a>',
                unsafe_allow_html=True
            )
