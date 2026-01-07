"""
Professional PDF report generation with annotated images.
Creates detailed inspection reports with logo, status stamps, and visual overlays.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import os

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, red, green, orange, Color
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether, Flowable
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing, Rect, String
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import config, REPORT_DIR
from utils.image_utils import create_heatmap_overlay, create_side_by_side_comparison

logger = setup_logger(__name__, level=config.log_level, component="REPORTS")


# ============================================================================
# COLORS
# ============================================================================

BRAND_PRIMARY = HexColor("#1e40af")   # Deep blue
BRAND_SUCCESS = HexColor("#059669")   # Green
BRAND_WARNING = HexColor("#d97706")   # Orange
BRAND_DANGER = HexColor("#dc2626")    # Red
BRAND_GRAY = HexColor("#6b7280")      # Gray
BRAND_LIGHT = HexColor("#f3f4f6")     # Light gray
BRAND_DARK = HexColor("#1f2937")      # Dark


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_explanation_sections(explanation: str) -> Dict[str, str]:
    """
    Parse explainer output into sections.
    
    Returns dict with keys: EXECUTIVE SUMMARY, INSPECTION DETAILS, 
    DEFECT ANALYSIS, FINAL RECOMMENDATION
    """
    # Handle None or empty explanation
    if not explanation:
        return {"INTRODUCTION": "Explanation not available - workflow may have been interrupted."}
    
    sections = {}
    section_names = [
        "EXECUTIVE SUMMARY",
        "INSPECTION DETAILS", 
        "DEFECT ANALYSIS",
        "FINAL RECOMMENDATION"
    ]
    
    # Clean markdown formatting
    text = explanation.replace("**", "").replace("##", "").replace("#", "")
    text = text.strip()
    
    # Split by section headers
    current_section = "INTRODUCTION"
    current_content = []
    
    for line in text.split("\n"):
        line_upper = line.strip().upper()
        
        # Check if line is a section header
        matched_section = None
        for section in section_names:
            if section in line_upper or line_upper.startswith(section):
                matched_section = section
                break
        
        if matched_section:
            # Save previous section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()
            current_section = matched_section
            current_content = []
        else:
            if line.strip():
                current_content.append(line.strip())
    
    # Save last section
    if current_content:
        sections[current_section] = "\n".join(current_content).strip()
    
    return sections


def format_agreement_score(score: float) -> str:
    """Format agreement score as percentage, avoiding floating point issues."""
    if score >= 0.9999:
        return "100.0%"
    elif score <= 0.0001:
        return "0.0%"
    else:
        return f"{score * 100:.1f}%"


def get_short_model_name(model_id: str) -> str:
    """Get short display name from full model ID."""
    # Extract just the model name, not the full repo path
    short_name = model_id.split("/")[-1]
    # Further shorten if needed
    short_name = short_name.replace("-Instruct", "").replace("-instruct", "")
    return short_name


# ============================================================================
# CUSTOM FLOWABLES
# ============================================================================

class StatusStamp(Flowable):
    """A large status stamp (PASSED/REJECTED/REVIEW)."""
    
    def __init__(self, verdict: str, width=200, height=60):
        Flowable.__init__(self)
        self.verdict = verdict
        self.width = width
        self.height = height
        
    def draw(self):
        # Determine colors and text
        if self.verdict == "SAFE":
            text = "✓ PASSED"
            fill_color = BRAND_SUCCESS
        elif self.verdict == "UNSAFE":
            text = "✗ REJECTED"
            fill_color = BRAND_DANGER
        else:
            text = "⚠ REVIEW REQUIRED"
            fill_color = BRAND_WARNING
        
        # Draw rounded rectangle
        self.canv.setFillColor(fill_color)
        self.canv.setStrokeColor(fill_color)
        self.canv.roundRect(0, 0, self.width, self.height, 10, fill=1, stroke=1)
        
        # Draw text
        self.canv.setFillColor(white)
        self.canv.setFont("Helvetica-Bold", 18)
        text_width = self.canv.stringWidth(text, "Helvetica-Bold", 18)
        self.canv.drawString((self.width - text_width) / 2, (self.height - 18) / 2 + 5, text)
    
    def wrap(self, availWidth, availHeight):
        return (self.width, self.height)


# ============================================================================
# PDF HEADER/FOOTER WITH LOGO
# ============================================================================

class BrandedCanvas(canvas.Canvas):
    """Canvas with branded header, footer, and page numbers."""
    
    def __init__(self, *args, logo_path=None, report_id=None, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
        self.logo_path = logo_path
        self.report_id = report_id or "N/A"
        self.inspector_model = config.vlm_inspector_model.split("/")[-1]
        self.auditor_model = config.vlm_auditor_model.split("/")[-1]
    
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
    
    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header()
            self._draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    
    def _draw_header(self):
        """Draw branded header on every page."""
        self.saveState()
        
        page_width = letter[0]
        header_height = 50
        header_y = letter[1] - header_height - 20
        
        # Header background
        self.setFillColor(BRAND_DARK)
        self.rect(0.5 * inch, header_y, page_width - 1 * inch, header_height, fill=1, stroke=0)
        
        # Logo (left side)
        if self.logo_path and Path(self.logo_path).exists():
            try:
                logo_width = 100
                logo_height = 40
                self.drawImage(
                    str(self.logo_path),
                    0.6 * inch,
                    header_y + 5,
                    width=logo_width,
                    height=logo_height,
                    preserveAspectRatio=True,
                    mask='auto'
                )
            except Exception as e:
                logger.warning(f"Failed to draw logo: {e}")
                self.setFillColor(white)
                self.setFont("Helvetica-Bold", 14)
                self.drawString(0.6 * inch, header_y + 20, "VISION INSPECTION")
        else:
            self.setFillColor(white)
            self.setFont("Helvetica-Bold", 14)
            self.drawString(0.6 * inch, header_y + 20, "VISION INSPECTION")
        
        # Report ID (right side)
        self.setFillColor(white)
        self.setFont("Helvetica-Bold", 11)
        report_text = f"INSPECTION ID: #{self.report_id.upper()}"
        self.drawRightString(page_width - 0.6 * inch, header_y + 28, report_text)
        
        # Date (right side, below ID)
        self.setFont("Helvetica", 9)
        date_text = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.drawRightString(page_width - 0.6 * inch, header_y + 12, date_text)
        
        self.restoreState()
    
    def _draw_footer(self, page_count):
        """Draw footer with page numbers and model info."""
        self.saveState()
        
        page_width = letter[0]
        footer_y = 0.4 * inch
        
        # Footer line
        self.setStrokeColor(BRAND_GRAY)
        self.line(0.5 * inch, footer_y + 15, page_width - 0.5 * inch, footer_y + 15)
        
        # Left: Generated by
        self.setFont("Helvetica", 8)
        self.setFillColor(BRAND_GRAY)
        self.drawString(
            0.5 * inch,
            footer_y,
            f"Generated by Agentic AI Framework • Auditor: {self.auditor_model}"
        )
        
        # Right: Page number
        self.drawRightString(
            page_width - 0.5 * inch,
            footer_y,
            f"Page {self._pageNumber} of {page_count}"
        )
        
        self.restoreState()


# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class InspectionReport:
    """Professional inspection report generator with branding."""
    
    def __init__(self, logo_path: Optional[Path] = None):
        self.logger = logger
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Look for logo
        if logo_path:
            self.logo_path = Path(logo_path)
        else:
            # Try to find Mouri.jpg in project root
            possible_paths = [
                Path("Mouri.jpg"),
                Path("d:/Vision Inspection System/Mouri.jpg"),
                Path(__file__).parent.parent.parent / "Mouri.jpg"
            ]
            self.logo_path = None
            for p in possible_paths:
                if p.exists():
                    self.logo_path = p
                    break
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        
        def safe_add(style):
            """Safely add style if it doesn't exist."""
            if style.name not in self.styles:
                self.styles.add(style)
        
        # Title style
        safe_add(ParagraphStyle(
            name="CustomTitle",
            parent=self.styles["Title"],
            fontSize=28,
            textColor=BRAND_PRIMARY,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold"
        ))
        
        # Section header
        safe_add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading1"],
            fontSize=16,
            textColor=BRAND_PRIMARY,
            spaceBefore=25,
            spaceAfter=12,
            fontName="Helvetica-Bold",
            borderWidth=0,
            borderColor=BRAND_PRIMARY,
            borderPadding=5,
            leftIndent=0
        ))
        
        # Subsection header
        safe_add(ParagraphStyle(
            name="SubHeader",
            parent=self.styles["Heading2"],
            fontSize=12,
            textColor=BRAND_DARK,
            spaceBefore=15,
            spaceAfter=8,
            fontName="Helvetica-Bold"
        ))
        
        # Body text with better wrapping
        safe_add(ParagraphStyle(
            name="CustomBodyText",
            parent=self.styles["Normal"],
            fontSize=10,
            textColor=BRAND_DARK,
            spaceAfter=10,
            alignment=TA_JUSTIFY,
            leading=14
        ))
        
        # Verdict styles
        for verdict, color in [
            ("Safe", BRAND_SUCCESS),
            ("Unsafe", BRAND_DANGER),
            ("Review", BRAND_WARNING)
        ]:
            safe_add(ParagraphStyle(
                name=f"Verdict{verdict}",
                parent=self.styles["Normal"],
                fontSize=20,
                textColor=color,
                alignment=TA_CENTER,
                spaceBefore=10,
                spaceAfter=10,
                fontName="Helvetica-Bold"
            ))
    
    def generate(
        self,
        state: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate PDF report from inspection state.
        
        Args:
            state: Inspection workflow state
            output_path: Optional output path
        
        Returns:
            Path to generated PDF
        """
        self.logger.info("Generating PDF report...")
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_id = state.get("request_id", "unknown")
            filename = f"inspection_{request_id}_{timestamp}.pdf"
            output_path = REPORT_DIR / filename
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        request_id = state.get("request_id", "unknown")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1.25 * inch,  # More space for header
            bottomMargin=0.75 * inch
        )
        
        # Build story (content elements)
        story = []
        
        # Title section
        story.extend(self._build_title_section(state))
        
        # Status stamp
        story.extend(self._build_status_stamp(state))
        
        # Executive summary
        story.extend(self._build_executive_summary(state))
        
        # Evidence section (side-by-side images)
        story.extend(self._build_evidence_section(state))
        
        # Defect details
        story.extend(self._build_defect_details(state))
        
        # Model comparison
        story.extend(self._build_model_comparison(state))
        
        # Audit trail
        story.extend(self._build_audit_trail(state))
        
        # Build PDF with branded canvas
        def make_canvas(*args, **kwargs):
            return BrandedCanvas(
                *args,
                logo_path=str(self.logo_path) if self.logo_path else None,
                report_id=request_id,
                **kwargs
            )
        
        doc.build(story, canvasmaker=make_canvas)
        
        self.logger.info(f"PDF report generated: {output_path}")
        
        return output_path
    
    def _build_title_section(self, state: Dict[str, Any]) -> List:
        """Build title section."""
        elements = []
        
        # Title
        title = Paragraph(
            "INSPECTION REPORT",
            self.styles["CustomTitle"]
        )
        elements.append(title)
        elements.append(Spacer(1, 0.1 * inch))
        
        # Subtitle
        subtitle = Paragraph(
            "AI-Powered Damage Detection & Safety Analysis",
            ParagraphStyle(
                name="Subtitle",
                parent=self.styles["Normal"],
                fontSize=12,
                textColor=BRAND_GRAY,
                alignment=TA_CENTER
            )
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _build_status_stamp(self, state: Dict[str, Any]) -> List:
        """Build large status stamp."""
        elements = []
        
        verdict = state.get("safety_verdict", {}).get("verdict", "UNKNOWN")
        
        # Center the stamp
        stamp_table = Table(
            [[StatusStamp(verdict, width=250, height=70)]],
            colWidths=[7 * inch]
        )
        stamp_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        
        elements.append(stamp_table)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _build_executive_summary(self, state: Dict[str, Any]) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        
        # Get data
        context = state.get("context", {})
        verdict = state.get("safety_verdict", {})
        consensus = state.get("consensus", {})
        explanation = state.get("explanation", "No explanation available.")
        
        # Key metrics table
        defect_count = len(consensus.get("combined_defects", []))
        critical_count = sum(
            1 for d in consensus.get("combined_defects", [])
            if d.get("safety_impact") == "CRITICAL"
        )
        
        summary_data = [
            ["Metric", "Value"],
            ["Image Inspected", Path(state.get("image_path", "")).name],
            ["Criticality Level", context.get("criticality", "unknown").upper()],
            ["Domain", context.get("domain") or "General"],
            ["Total Defects Found", str(defect_count)],
            ["Critical Defects", str(critical_count)],
            ["Models Agreement", "Yes" if consensus.get("models_agree") else "No"],
            ["Confidence Level", verdict.get("confidence_level", "unknown").title()],
            ["Human Review", "Required" if verdict.get("requires_human") else "Not Required"]
        ]
        
        table = Table(summary_data, colWidths=[2.5 * inch, 4 * inch])
        table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            # Data rows
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("BACKGROUND", (0, 1), (0, -1), BRAND_LIGHT),
            # Alignment
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Borders
            ("BOX", (0, 0), (-1, -1), 1, BRAND_GRAY),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Analysis text - Parse into sections with proper formatting
        elements.append(Paragraph("<b>Analysis Summary:</b>", self.styles["SubHeader"]))
        
        # Parse explanation into sections
        sections = parse_explanation_sections(explanation)
        
        # Section header style
        section_header_style = ParagraphStyle(
            name="_SectionHeader",
            parent=self.styles["Normal"],
            fontSize=12,
            textColor=BRAND_PRIMARY,
            fontName="Helvetica-Bold",
            spaceBefore=12,
            spaceAfter=6
        )
        
        # Display each section with proper formatting
        section_order = [
            "EXECUTIVE SUMMARY",
            "INSPECTION DETAILS",
            "DEFECT ANALYSIS",
            "FINAL RECOMMENDATION"
        ]
        
        for section_name in section_order:
            if section_name in sections:
                elements.append(Paragraph(section_name, section_header_style))
                elements.append(Paragraph(sections[section_name], self.styles["CustomBodyText"]))
                elements.append(Spacer(1, 0.1 * inch))
        
        # If no sections were parsed, just display the raw explanation
        if not any(s in sections for s in section_order):
            display_text = explanation if explanation else "Explanation not available."
            elements.append(Paragraph(display_text, self.styles["CustomBodyText"]))
        
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_evidence_section(self, state: Dict[str, Any]) -> List:
        """Build evidence section with side-by-side images."""
        elements = []
        
        elements.append(Paragraph("Visual Evidence", self.styles["SectionHeader"]))
        
        image_path = Path(state.get("image_path", ""))
        if not image_path.exists():
            elements.append(Paragraph("Image not available", self.styles["CustomBodyText"]))
            return elements
        
        consensus = state.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        
        try:
            # Create heatmap overlay
            heatmap_path = REPORT_DIR / f"heatmap_{image_path.stem}.jpg"
            create_heatmap_overlay(image_path, defects, heatmap_path)
            
            # Create side-by-side comparison
            comparison_path = REPORT_DIR / f"comparison_{image_path.stem}.jpg"
            create_side_by_side_comparison(
                image_path, 
                heatmap_path, 
                comparison_path,
                labels=("Original Input", "AI Analysis Layer")
            )
            
            # Add comparison image to PDF
            if comparison_path.exists():
                img = RLImage(
                    str(comparison_path), 
                    width=6.5 * inch, 
                    height=3 * inch,
                    kind="proportional"
                )
                elements.append(img)
                elements.append(Spacer(1, 0.1 * inch))
                
                # Caption
                caption = Paragraph(
                    "<i>Left: Original uploaded image. Right: AI analysis with defect regions highlighted.</i>",
                    ParagraphStyle(
                        name="Caption",
                        parent=self.styles["Normal"],
                        fontSize=9,
                        textColor=BRAND_GRAY,
                        alignment=TA_CENTER
                    )
                )
                elements.append(caption)
        except Exception as e:
            self.logger.error(f"Failed to create evidence images: {e}")
            # Fallback to original image
            try:
                img = RLImage(str(image_path), width=5 * inch, height=4 * inch, kind="proportional")
                elements.append(img)
            except Exception as e2:
                elements.append(Paragraph(f"Failed to load image: {e2}", self.styles["CustomBodyText"]))
        
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_defect_details(self, state: Dict[str, Any]) -> List:
        """Build detailed defect listing."""
        elements = []
        
        consensus = state.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        
        elements.append(Paragraph(f"Defect Analysis ({len(defects)} found)", self.styles["SectionHeader"]))
        
        if not defects:
            elements.append(Paragraph(
                "✓ No defects detected. The inspected item appears to be in good condition.",
                self.styles["CustomBodyText"]
            ))
            return elements
        
        for i, defect in enumerate(defects, 1):
            # Defect header with severity badge
            severity = defect.get("safety_impact", "UNKNOWN")
            if severity == "CRITICAL":
                severity_color = BRAND_DANGER
            elif severity == "MODERATE":
                severity_color = BRAND_WARNING
            else:
                severity_color = BRAND_SUCCESS
            
            defect_title = f"<b>{i}. {defect.get('type', 'Unknown').upper()}</b>"
            elements.append(Paragraph(defect_title, self.styles["SubHeader"]))
            
            # Defect details table
            details = [
                ["Severity", severity],
                ["Location", defect.get("location", "Not specified")],
                ["Confidence", defect.get("confidence", "unknown").title()],
                ["Reasoning", defect.get("reasoning", "Not provided")],
                ["Recommendation", defect.get("recommended_action", "Further inspection recommended")]
            ]
            
            table = Table(details, colWidths=[1.5 * inch, 5 * inch])
            table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (0, -1), BRAND_LIGHT),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 1, BRAND_GRAY),
                ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                # Color severity row
                ("TEXTCOLOR", (1, 0), (1, 0), severity_color),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.15 * inch))
        
        return elements
    
    def _build_model_comparison(self, state: Dict[str, Any]) -> List:
        """Build model comparison section."""
        elements = []
        
        elements.append(Paragraph("Model Analysis Comparison", self.styles["SectionHeader"]))
        
        inspector = state.get("inspector_result", {})
        auditor = state.get("auditor_result", {})
        consensus = state.get("consensus", {})
        
        # Get model names (short versions)
        inspector_name = config.vlm_inspector_model.split("/")[-1]
        auditor_name = config.vlm_auditor_model.split("/")[-1]
        
        # Comparison table
        comparison_data = [
            ["Metric", f"Inspector\n({inspector_name})", f"Auditor\n({auditor_name})"],
            ["Object Identified", 
             str(inspector.get("object_identified", "N/A")), 
             str(auditor.get("object_identified", "N/A"))],
            ["Overall Condition", 
             str(inspector.get("overall_condition", "N/A")), 
             str(auditor.get("overall_condition", "N/A"))],
            ["Defects Found", 
             str(len(inspector.get("defects", []))), 
             str(len(auditor.get("defects", [])))],
            ["Confidence", 
             str(inspector.get("overall_confidence", "N/A")).title(), 
             str(auditor.get("overall_confidence", "N/A")).title()],
            ["Agreement Score", 
             f"{consensus.get('agreement_score', 0):.0%}", 
             "Yes" if consensus.get("models_agree") else "No"]
        ]
        
        table = Table(comparison_data, colWidths=[2 * inch, 2.25 * inch, 2.25 * inch])
        table.setStyle(TableStyle([
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            # Data
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (0, -1), BRAND_LIGHT),
            # Alignment
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Borders
            ("BOX", (0, 0), (-1, -1), 1, BRAND_GRAY),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_audit_trail(self, state: Dict[str, Any]) -> List:
        """Build audit trail section."""
        elements = []
        
        elements.append(Paragraph("Audit Trail", self.styles["SectionHeader"]))
        
        verdict = state.get("safety_verdict", {})
        
        # Triggered gates
        gates = verdict.get("triggered_gates", [])
        if gates:
            elements.append(Paragraph("<b>Safety Gates Triggered:</b>", self.styles["SubHeader"]))
            for gate in gates:
                elements.append(Paragraph(f"• {gate}", self.styles["CustomBodyText"]))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Processing info
        processing_time = state.get('processing_time') or 0
        processing_data = [
            ["Parameter", "Value"],
            ["Processing Time", f"{processing_time:.2f} seconds"],
            ["Inspector Model", config.vlm_inspector_model],
            ["Auditor Model", config.vlm_auditor_model],
            ["Explainer Model", config.explainer_model],
            ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["System Version", "1.0.0"]
        ]
        
        table = Table(processing_data, colWidths=[2 * inch, 4.5 * inch])
        table.setStyle(TableStyle([
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            # Data
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (0, -1), BRAND_LIGHT),
            # Alignment
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            # Borders
            ("BOX", (0, 0), (-1, -1), 1, BRAND_GRAY),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            # Padding
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        
        return elements


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_report(state: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
    """Generate inspection report."""
    reporter = InspectionReport()
    return reporter.generate(state, output_path)