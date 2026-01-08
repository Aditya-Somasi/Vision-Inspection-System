"""
Decision Support Component for Vision Inspection System.
Displays repair vs replace cost analysis and recommendations.
"""

import streamlit as st
from typing import Dict, Any


def render_decision_support(results: Dict[str, Any]):
    """
    Display Decision Support (Repair vs Replace) section.
    
    Args:
        results: Full inspection results containing decision_support key
    """
    if "decision_support" not in results:
        return

    decision = results.get("decision_support", {})
    if not decision or decision.get("recommendation", "Review") == "No Action Required":
        return

    st.divider()
    st.subheader("💰 Decision Support")
    
    # Recommendation Banner
    rec = decision.get("recommendation", "REVIEW").upper()
    
    if rec == "REPLACE":
        bg_color = "#fee2e2"  # Red-100
        border_color = "#ef4444"
        icon = "🛑"
    elif rec == "REPAIR":
        bg_color = "#fef3c7"  # Amber-100
        border_color = "#f59e0b" 
        icon = "🔧"
    else:
        bg_color = "#e0f2fe"  # Blue-100
        border_color = "#3b82f6"
        icon = "ℹ️"

    st.markdown(f"""
    <div style="background-color: {bg_color}; border: 2px solid {border_color}; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-bottom: 1rem;">
        <h3 style="margin:0; color: #1f2937;">{icon} RECOMMENDATION: {rec}</h3>
        <p style="margin:0.5rem 0 0 0; color: #4b5563;">{decision.get('reasoning', '')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cost Comparison Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0 0 0.5rem 0; color: #374151;">🔧 Repair Option</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Estimated Cost", decision.get("repair_cost", "N/A"))
        st.caption(f"⏱️ Time: {decision.get('repair_time', 'N/A')}")
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin:0 0 0.5rem 0; color: #374151;">📦 Replace Option</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Estimated Cost", decision.get("replace_cost", "N/A"))
        st.caption(f"📅 Lead Time: {decision.get('replace_time', 'N/A')}")


def render_cost_comparison_table(decision: Dict[str, Any]):
    """
    Render a detailed cost comparison table.
    
    Args:
        decision: Decision support data dictionary
    """
    import pandas as pd
    
    data = {
        "Metric": ["Estimated Cost", "Time Required", "Recommendation"],
        "Repair": [
            decision.get("repair_cost", "N/A"),
            decision.get("repair_time", "N/A"),
            "✓" if decision.get("recommendation") == "REPAIR" else ""
        ],
        "Replace": [
            decision.get("replace_cost", "N/A"),
            decision.get("replace_time", "N/A"),
            "✓" if decision.get("recommendation") == "REPLACE" else ""
        ]
    }
    
    df = pd.DataFrame(data)
    st.table(df)
