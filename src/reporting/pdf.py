"""
Professional PDF report generation with annotated images.
Creates detailed inspection reports with timestamps, audit trails, and visual overlays.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import io

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, red, green, orange
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.pdfgen import canvas
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from utils.logger import setup_logger
from utils.config import config, REPORT_DIR

logger = setup_logger(__name__, level=config.log_level, component="REPORTS")


# ============================================================================
# COLORS
# ============================================================================

BRAND_PRIMARY = HexColor("#2563eb")  # Blue
BRAND_SUCCESS = HexColor("#10b981")  # Green
BRAND_WARNING = HexColor("#f59e0b")  # Orange
BRAND_DANGER = HexColor("#ef4444")   # Red
BRAND_GRAY = HexColor("#6b7280")     # Gray


# ============================================================================
# IMAGE ANNOTATION
# ============================================================================

def annotate_image(
    image_path: Path,
    defects: List[Dict[str, Any]],
    output_path: Path
) -> Path:
    """
    Annotate image with bounding boxes for defects.
    
    Args:
        image_path: Path to original image
        defects: List of defect dictionaries with bbox info
        output_path: Path to save annotated image
    
    Returns:
        Path to annotated image
    """
    logger.debug(f"Annotating image: {image_path}")
    
    try:
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return image_path
        
        height, width = img.shape[:2]
        
        # Draw bounding boxes
        for i, defect in enumerate(defects):
            bbox = defect.get("bbox")
            if not bbox or bbox.get("x") is None:
                continue
            
            # Extract coordinates
            x = int(bbox["x"])
            y = int(bbox["y"])
            w = int(bbox["width"])
            h = int(bbox["height"])
            
            # Choose color based on severity
            safety_impact = defect.get("safety_impact", "MODERATE")
            if safety_impact == "CRITICAL":
                color = (0, 0, 255)  # Red
                thickness = 3
            elif safety_impact == "MODERATE":
                color = (0, 165, 255)  # Orange
                thickness = 2
            else:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            
            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Add label
            label = f"{i + 1}. {defect.get('type', 'defect').upper()}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Label background
            cv2.rectangle(
                img,
                (x, y - label_size[1] - 10),
                (x + label_size[0] + 10, y),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                img,
                label,
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Save annotated image
        cv2.imwrite(str(output_path), img)
        logger.info(f"Annotated image saved: {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Image annotation failed: {e}")
        return image_path


# ============================================================================
# PDF HEADER/FOOTER
# ============================================================================

class NumberedCanvas(canvas.Canvas):
    """Canvas with page numbers and headers."""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []
    
    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
    
    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
    
    def draw_page_number(self, page_count):
        """Draw page number at bottom."""
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColorRGB(0.4, 0.4, 0.4)
        
        # Page number
        page_num = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(
            letter[0] - 0.5 * inch,
            0.5 * inch,
            page_num
        )
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.drawString(
            0.5 * inch,
            0.5 * inch,
            f"Generated: {timestamp}"
        )
        
        self.restoreState()


# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class InspectionReport:
    """Professional inspection report generator."""
    
    def __init__(self):
        self.logger = logger
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name="CustomTitle",
            parent=self.styles["Title"],
            fontSize=24,
            textColor=BRAND_PRIMARY,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold"
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self.styles["Heading1"],
            fontSize=16,
            textColor=BRAND_PRIMARY,
            spaceBefore=20,
            spaceAfter=12,
            fontName="Helvetica-Bold",
            borderWidth=0,
            borderColor=BRAND_PRIMARY,
            borderPadding=5
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name="SubHeader",
            parent=self.styles["Heading2"],
            fontSize=13,
            textColor=BRAND_GRAY,
            spaceBefore=12,
            spaceAfter=8,
            fontName="Helvetica-Bold"
        ))
        
        # Verdict styles
        for verdict, color in [
            ("Safe", BRAND_SUCCESS),
            ("Unsafe", BRAND_DANGER),
            ("Review", BRAND_WARNING)
        ]:
            self.styles.add(ParagraphStyle(
                name=f"Verdict{verdict}",
                parent=self.styles["Normal"],
                fontSize=18,
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
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1 * inch,
            bottomMargin=1 * inch
        )
        
        # Build story (content elements)
        story = []
        
        # Title page
        story.extend(self._build_title_page(state))
        
        # Executive summary
        story.extend(self._build_executive_summary(state))
        
        # Annotated image
        story.extend(self._build_image_section(state))
        
        # Defect details
        story.extend(self._build_defect_details(state))
        
        # Model analysis
        story.extend(self._build_model_analysis(state))
        
        # Audit trail
        story.extend(self._build_audit_trail(state))
        
        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        
        self.logger.info(f"PDF report generated: {output_path}")
        
        return output_path
    
    def _build_title_page(self, state: Dict[str, Any]) -> List:
        """Build title page elements."""
        elements = []
        
        # Logo/Title
        title = Paragraph(
            "VISION INSPECTION SYSTEM",
            self.styles["CustomTitle"]
        )
        elements.append(title)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Subtitle
        subtitle = Paragraph(
            "AI-Powered Damage Detection & Safety Analysis",
            self.styles["Normal"]
        )
        subtitle.alignment = TA_CENTER
        elements.append(subtitle)
        elements.append(Spacer(1, 0.5 * inch))
        
        # Report info table
        context = state.get("context", {})
        verdict = state.get("safety_verdict", {})
        
        report_data = [
            ["Report ID:", state.get("request_id", "N/A")],
            ["Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Image:", Path(state.get("image_path", "")).name],
            ["Criticality:", context.get("criticality", "unknown").upper()],
            ["Domain:", context.get("domain") or "General"],
            ["", ""],
            ["Final Verdict:", verdict.get("verdict", "UNKNOWN")]
        ]
        
        # Determine verdict style
        verdict_text = verdict.get("verdict", "UNKNOWN")
        if verdict_text == "SAFE":
            verdict_color = BRAND_SUCCESS
        elif verdict_text == "UNSAFE":
            verdict_color = BRAND_DANGER
        else:
            verdict_color = BRAND_WARNING
        
        table = Table(report_data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 11),
            ("FONT", (0, -1), (0, -1), "Helvetica-Bold", 11),
            ("FONT", (1, -1), (1, -1), "Helvetica-Bold", 14),
            ("TEXTCOLOR", (1, -1), (1, -1), verdict_color),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -2), 1, BRAND_GRAY),
            ("BOX", (0, -1), (-1, -1), 2, verdict_color),
        ]))
        
        elements.append(table)
        elements.append(PageBreak())
        
        return elements
    
    def _build_executive_summary(self, state: Dict[str, Any]) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Get data
        verdict = state.get("safety_verdict", {})
        consensus = state.get("consensus", {})
        explanation = state.get("explanation", "No explanation available.")
        
        # Verdict box
        verdict_text = verdict.get("verdict", "UNKNOWN")
        verdict_para = Paragraph(
            f"<b>VERDICT: {verdict_text}</b>",
            self.styles[f"Verdict{verdict_text.title()}" if verdict_text in ["SAFE", "UNSAFE"] else "VerdictReview"]
        )
        elements.append(verdict_para)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Summary stats
        defect_count = len(consensus.get("combined_defects", []))
        critical_count = sum(
            1 for d in consensus.get("combined_defects", [])
            if d.get("safety_impact") == "CRITICAL"
        )
        
        summary_data = [
            ["Defects Found:", str(defect_count)],
            ["Critical Defects:", str(critical_count)],
            ["Models Agree:", "Yes" if consensus.get("models_agree") else "No"],
            ["Confidence:", verdict.get("confidence_level", "unknown").title()],
            ["Human Review:", "Required" if verdict.get("requires_human") else "Not Required"]
        ]
        
        table = Table(summary_data, colWidths=[2.5 * inch, 3 * inch])
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f3f4f6")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3 * inch))
        
        # Explanation text
        elements.append(Paragraph("Analysis:", self.styles["SubHeader"]))
        elements.append(Paragraph(explanation, self.styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements
    
    def _build_image_section(self, state: Dict[str, Any]) -> List:
        """Build image section with annotations."""
        elements = []
        
        elements.append(Paragraph("Inspected Image", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Annotate image
        image_path = Path(state.get("image_path", ""))
        if not image_path.exists():
            elements.append(Paragraph("Image not available", self.styles["Normal"]))
            return elements
        
        consensus = state.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        
        # Create annotated image
        annotated_path = REPORT_DIR / f"annotated_{image_path.stem}.jpg"
        annotate_image(image_path, defects, annotated_path)
        
        # Add to PDF
        try:
            img = RLImage(str(annotated_path), width=6 * inch, height=4 * inch, kind="proportional")
            elements.append(img)
        except Exception as e:
            self.logger.error(f"Failed to add image to PDF: {e}")
            elements.append(Paragraph(f"Failed to load image: {e}", self.styles["Normal"]))
        
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_defect_details(self, state: Dict[str, Any]) -> List:
        """Build detailed defect listing."""
        elements = []
        
        consensus = state.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        
        if not defects:
            elements.append(Paragraph("Defect Analysis", self.styles["SectionHeader"]))
            elements.append(Paragraph("No defects detected.", self.styles["Normal"]))
            return elements
        
        elements.append(Paragraph(f"Defect Analysis ({len(defects)} found)", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        for i, defect in enumerate(defects, 1):
            # Defect header
            defect_title = Paragraph(
                f"<b>{i}. {defect.get('type', 'Unknown').upper()} - {defect.get('safety_impact', 'UNKNOWN')}</b>",
                self.styles["SubHeader"]
            )
            elements.append(defect_title)
            
            # Defect details
            details = [
                ["Location:", defect.get("location", "Not specified")],
                ["Confidence:", defect.get("confidence", "unknown").title()],
                ["Reasoning:", defect.get("reasoning", "Not provided")],
                ["Recommendation:", defect.get("recommended_action", "Not provided")]
            ]
            
            table = Table(details, colWidths=[1.5 * inch, 4.5 * inch])
            table.setStyle(TableStyle([
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
                ("BACKGROUND", (0, 0), (0, -1), HexColor("#f9fafb")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.15 * inch))
        
        return elements
    
    def _build_model_analysis(self, state: Dict[str, Any]) -> List:
        """Build model analysis section."""
        elements = []
        
        elements.append(Paragraph("Model Analysis", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        inspector = state.get("inspector_result", {})
        auditor = state.get("auditor_result", {})
        consensus = state.get("consensus", {})
        
        # Comparison table
        comparison_data = [
            ["Metric", "Inspector (Qwen2-VL)", "Auditor (Llama 3.2)"],
            ["Object Identified", inspector.get("object_identified", "N/A"), auditor.get("object_identified", "N/A")],
            ["Overall Condition", inspector.get("overall_condition", "N/A"), auditor.get("overall_condition", "N/A")],
            ["Defects Found", str(len(inspector.get("defects", []))), str(len(auditor.get("defects", [])))],
            ["Confidence", inspector.get("overall_confidence", "N/A").title(), auditor.get("overall_confidence", "N/A").title()],
            ["Agreement", "Yes" if consensus.get("models_agree") else "No", f"{consensus.get('agreement_score', 0):.1%}"]
        ]
        
        table = Table(comparison_data, colWidths=[2 * inch, 2 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
            ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 1, BRAND_GRAY),
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
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
        elements.append(Spacer(1, 0.1 * inch))
        
        verdict = state.get("safety_verdict", {})
        
        # Triggered gates
        gates = verdict.get("triggered_gates", [])
        if gates:
            elements.append(Paragraph("<b>Safety Gates Triggered:</b>", self.styles["Normal"]))
            for gate in gates:
                elements.append(Paragraph(f"• {gate}", self.styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Processing info
        processing_data = [
            ["Start Time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Processing Time:", f"{state.get('processing_time', 0):.2f} seconds"],
            ["Inspector Model:", config.vlm_inspector_model],
            ["Auditor Model:", config.vlm_auditor_model],
            ["System Version:", "1.0.0"]
        ]
        
        table = Table(processing_data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#f9fafb")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
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