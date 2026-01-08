"""
Professional PDF report generation with annotated images.
Creates detailed inspection reports with logo, status stamps, and visual overlays.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import io
import os
import re

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, red, green, orange, Color
from reportlab.lib import colors
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
from utils.image_utils import create_heatmap_overlay, create_side_by_side_comparison, draw_bounding_boxes

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
    Parse explainer output into sections for structured PDF display.
    
    Handles various markdown formats: **Bold Headers**, ## Markdown Headers.
    Preserves the full content of each section including multi-paragraph text.
    
    Expected format from explainer:
    [Main explanation text]
    
    ---
    
    ## REASONING CHAINS
    
    [Reasoning chains content]
    
    ---
    
    ## COUNTERFACTUAL ANALYSIS
    
    [Counterfactual content]
    """
    # Handle None or empty explanation
    if not explanation:
        return {"SUMMARY": "Explanation not available - workflow may have been interrupted."}
    
    sections = {}
    logger.debug(f"Parsing explanation sections. Length: {len(explanation)}")
    
    # Map of patterns to normalized section names
    # Ordered by specificity (longer patterns first to avoid false matches)
    section_patterns = [
        ("REASONING CHAINS", ["reasoning chains", "reasoning chain"]),
        ("INSPECTOR ANALYSIS", ["inspector analysis", "inspector:"]),
        ("AUDITOR VERIFICATION", ["auditor verification", "auditor:"]),
        ("COUNTERFACTUAL", ["counterfactual analysis", "counterfactual"]),
        ("KEY TAKEAWAYS", ["key takeaways", "key findings", "highlights"]),
        ("RECOMMENDATIONS", ["recommendations", "recommended actions", "next steps", "action items"]),
        ("SUMMARY", ["summary", "inspection findings", "verdict", "overview"]),
    ]
    
    text = explanation.strip()
    
    # Strategy 1: Try to split by explicit section markers (---) followed by ## headers
    # This handles the explainer format: "content\n\n---\n\n## SECTION_NAME\n\ncontent"
    section_marker_pattern = r'(?:^|\n+)---+\n+##\s*([A-Z\s]+)\n+'
    
    # Find all explicit sections with markers
    explicit_sections = list(re.finditer(section_marker_pattern, text, re.MULTILINE))
    
    if explicit_sections:
        logger.debug(f"Found {len(explicit_sections)} explicit section markers")
        
        # Extract SUMMARY (everything before first marker)
        first_marker_start = explicit_sections[0].start()
        summary_text = text[:first_marker_start].strip()
        if summary_text:
            # Clean up markdown
            summary_text = summary_text.replace("**", "").replace("##", "").replace("#", "")
            sections["SUMMARY"] = summary_text
        
        # Extract each named section
        for i, match in enumerate(explicit_sections):
            section_header = match.group(1).strip()
            section_start = match.end()
            
            # Find where this section ends (either at next marker or end of text)
            if i + 1 < len(explicit_sections):
                section_end = explicit_sections[i + 1].start()
            else:
                section_end = len(text)
            
            section_content = text[section_start:section_end].strip()
            
            # Normalize section name
            normalized_name = section_header
            for sname, patterns in section_patterns:
                for pattern in patterns:
                    if pattern in section_header.lower():
                        normalized_name = sname
                        break
                if normalized_name != section_header:
                    break
            
            # Clean markdown
            section_content = section_content.replace("**", "").replace("##", "").replace("#", "")
            
            if section_content:
                sections[normalized_name] = section_content
                logger.debug(f"Extracted section: {normalized_name} ({len(section_content)} chars)")
    else:
        logger.debug("No explicit section markers found, using pattern matching on lines")
        
        # Strategy 2: Fall back to line-by-line pattern matching (no explicit markers)
        current_section = "SUMMARY"
        current_content = []
        
        for line in text.split("\n"):
            line_stripped = line.strip()
            if not line_stripped:
                # Preserve blank lines within sections
                if current_content:
                    current_content.append("")
                continue
            
            line_clean = line_stripped.replace("**", "").replace("##", "").replace("#", "").replace(":", "").strip()
            line_lower = line_clean.lower()
            
            # Check if line is a section header (must be at line start and reasonably short)
            matched_section = None
            for section_name, patterns in section_patterns:
                for pattern in patterns:
                    # Match if pattern is at the start of the line and line is short (header-like)
                    if line_lower.startswith(pattern) and len(line_clean) < 80:
                        matched_section = section_name
                        break
                if matched_section:
                    break
            
            if matched_section:
                # Save previous section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    # Clean up markdown formatting for PDF
                    content_text = content_text.replace("**", "").replace("##", "").replace("#", "")
                    if content_text:
                        sections[current_section] = content_text
                current_section = matched_section
                current_content = []
            else:
                # Add content line
                if line_stripped:
                    clean_line = line_stripped.replace("**", "").replace("##", "").replace("#", "")
                    current_content.append(clean_line)
                elif current_content:
                    # Preserve paragraph breaks
                    current_content.append("")
        
        # Save last section
        if current_content:
            content_text = "\n".join(current_content).strip()
            content_text = content_text.replace("**", "").replace("##", "").replace("#", "")
            if content_text:
                sections[current_section] = content_text
    
    # Strategy 3: Enhanced keyword-based extraction (ALWAYS RUN - more robust, works even if formatting varies)
    # This is more aggressive and will extract sections even without explicit markers
    logger.debug("Performing enhanced keyword-based section extraction")
    
    # Enhanced keywords that indicate section starts (more comprehensive)
    section_keywords = {
        "EXECUTIVE SUMMARY": ["executive summary", "summary", "overview", "inspection summary"],
        "KEY TAKEAWAYS": ["key takeaways", "key findings", "highlights", "main points", "takeaways"],
        "RECOMMENDATIONS": ["recommendations", "recommended actions", "next steps", "action items", "action required", "suggested actions"],
        "FINAL RECOMMENDATION": ["final recommendation", "recommendation", "verdict", "conclusion", "final verdict"],
        "REASONING CHAINS": ["reasoning chains", "reasoning", "analysis reasoning", "thought process"],
        "INSPECTOR ANALYSIS": ["inspector analysis", "inspector:", "inspector findings", "primary analysis"],
        "AUDITOR VERIFICATION": ["auditor verification", "auditor:", "auditor findings", "verification"],
        "COUNTERFACTUAL": ["counterfactual", "what if", "alternative scenario", "counterfactual analysis"]
    }
    
    # Always run keyword extraction to find additional sections
    # This supplements Strategy 1 and 2, doesn't replace them
    lines = text.split("\n")
    keyword_sections = {}
    current_section = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if line matches any section keyword
        matched_keyword = None
        for section_name, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in line_lower and len(line_lower) < 100:  # Reasonable header length
                    matched_keyword = section_name
                    break
            if matched_keyword:
                break
        
        if matched_keyword:
            # Save previous section
            if current_section and current_content:
                content = "\n".join(current_content).strip()
                if content:
                    keyword_sections[current_section] = content.replace("**", "").replace("##", "")
            
            current_section = matched_keyword
            current_content = []
        elif current_section:
            # Add to current section
            if line.strip():
                current_content.append(line.strip().replace("**", "").replace("##", ""))
        elif not current_section:
            # Before any section identified, treat as SUMMARY
            if line.strip() and not line_lower.startswith(("---", "##")):
                current_section = "SUMMARY"
                current_content.append(line.strip().replace("**", "").replace("##", ""))
    
    # Save last section
    if current_section and current_content:
        content = "\n".join(current_content).strip()
        if content:
            keyword_sections[current_section] = content.replace("**", "").replace("##", "")
    
    # Merge keyword sections with existing sections (keyword sections take precedence if they have more content)
    for section_name, content in keyword_sections.items():
        if section_name not in sections or len(content) > len(sections.get(section_name, "")):
            sections[section_name] = content
    
    # Ensure we have at least a SUMMARY - if not, use first meaningful content
    if not sections or "SUMMARY" not in sections:
        logger.warning("No SUMMARY section found, generating fallback from raw text")
        # Take first 3-5 sentences as summary
        sentences = explanation.split('.')
        summary_text = '. '.join(sentences[:5]).strip()
        if not summary_text:
            summary_text = explanation[:500]  # First 500 chars as fallback
        sections["SUMMARY"] = summary_text.replace("**", "").replace("##", "").replace("#", "")
    
    logger.debug(f"Parsed {len(sections)} sections: {list(sections.keys())}")
    
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
            # Try to find Mouri.jpg in project root (use relative/dynamic paths only)
            possible_paths = [
                Path("Mouri.jpg"),
                Path(__file__).parent.parent.parent / "Mouri.jpg",
                Path.cwd() / "Mouri.jpg",
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
        
        # Evidence section (3-panel visual layout)
        story.extend(self._build_visual_evidence(state))
        
        # Defect details
        story.extend(self._build_defect_details(state))
        
        # Decision Support (Cost & Time)
        story.extend(self._build_decision_support(state))
        
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
    
    def _build_decision_support(self, state: Dict[str, Any]) -> List:
        """Build decision support section with Repiar vs Replace analysis."""
        elements = []
        
        # Check if decision support data exists
        decision = state.get("decision_support", {})
        if not decision or decision.get("recommendation", "Review") == "No Action Required":
            return elements
            
        elements.append(Paragraph("Decision Support Analysis", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 0.1 * inch))
        
        # Main Recommendation Banner
        rec = decision.get("recommendation", "REVIEW").upper()
        rec_color = BRAND_DANGER if rec == "REPLACE" else BRAND_WARNING if rec == "REPAIR" else BRAND_PRIMARY
        
        # Create a banner table
        banner_data = [[f"RECOMMENDED ACTION: {rec}"]]
        banner = Table(banner_data, colWidths=[6.5 * inch])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), rec_color),
            ("TEXTCOLOR", (0, 0), (-1, -1), white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 14),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("ROUNDEDCORNERS", [10, 10, 10, 10]), 
        ]))
        elements.append(banner)
        elements.append(Spacer(1, 0.2 * inch))
        
        # Reasoning text
        reasoning = decision.get("reasoning", "")
        if reasoning:
            elements.append(Paragraph(f"<b>Basis:</b> {reasoning}", self.styles["CustomBodyText"]))
            elements.append(Spacer(1, 0.2 * inch))
        
        # Comparison Table
        # Metrics: Cost (INR), Time, Feasibility
        
        data = [
            ["Metric", "Repair Option", "Replace Option"],
            ["Estimated Cost", decision.get("repair_cost", "N/A"), decision.get("replace_cost", "N/A")],
            ["Time Required", decision.get("repair_time", "N/A"), decision.get("replace_time", "N/A")],
        ]
        
        t = Table(data, colWidths=[2.0 * inch, 2.25 * inch, 2.25 * inch])
        t.setStyle(TableStyle([
            # Header
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_GRAY),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            
            # Data
            ("ALIGN", (0, 1), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"), # First col bold
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 0.3 * inch))
        
        return elements

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
        
        # Get additional details for comprehensive summary
        inspector = state.get("inspector_result", {})
        auditor = state.get("auditor_result", {})
        object_name = inspector.get("object_identified") or auditor.get("object_identified", "Unknown")
        agreement_score = consensus.get("agreement_score", 0)
        inspector_conf = inspector.get("overall_confidence", "unknown").title()
        auditor_conf = auditor.get("overall_confidence", "unknown").title()
        moderate_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "MODERATE")
        cosmetic_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "COSMETIC")
        
        summary_data = [
            ["Metric", "Value"],
            ["Image Inspected", Path(state.get("image_path", "")).name],
            ["Object Identified", object_name],
            ["Criticality Level", context.get("criticality", "unknown").upper()],
            ["Domain", context.get("domain") or "General"],
            ["Total Defects Found", str(defect_count)],
            ["Critical Defects", str(critical_count)],
            ["Moderate Defects", str(moderate_count)],
            ["Cosmetic Defects", str(cosmetic_count)],
            ["Final Verdict", verdict.get("verdict", "UNKNOWN")],
            ["Models Agreement", f"{'Yes' if consensus.get('models_agree') else 'No'} ({agreement_score:.0%})"],
            ["Inspector Confidence", inspector_conf],
            ["Auditor Confidence", auditor_conf],
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
        elements.append(Paragraph("<b>Detailed Analysis Summary:</b>", self.styles["SubHeader"]))
        
        # Parse explanation into sections
        sections = parse_explanation_sections(explanation)
        logger.info(f"Parsed {len(sections)} sections for PDF: {list(sections.keys())}")
        
        # Add completeness indicator
        required_sections = ["SUMMARY", "FINAL RECOMMENDATION"]
        missing_sections = [s for s in required_sections if s not in sections or not sections.get(s, "").strip()]
        
        if missing_sections:
            completeness_text = f"<b>Analysis Completeness:</b> ⚠ Some sections may be missing or truncated. "
            completeness_text += f"Missing: {', '.join(missing_sections)}. "
            completeness_text += "Fallback summary generated from structured data."
            elements.append(Paragraph(completeness_text, ParagraphStyle(
                name="CompletenessWarning",
                parent=self.styles["Normal"],
                fontSize=9,
                textColor=BRAND_WARNING,
                fontName="Helvetica-Oblique"
            )))
            elements.append(Spacer(1, 0.1 * inch))
        else:
            completeness_text = "<b>Analysis Completeness:</b> ✓ All required sections present."
            elements.append(Paragraph(completeness_text, ParagraphStyle(
                name="CompletenessOK",
                parent=self.styles["Normal"],
                fontSize=9,
                textColor=BRAND_SUCCESS,
                fontName="Helvetica-Oblique"
            )))
            elements.append(Spacer(1, 0.1 * inch))
        
        # Validate that SUMMARY section exists (critical) - generate comprehensive fallback if missing
        if "SUMMARY" not in sections or not sections.get("SUMMARY", "").strip():
            logger.warning("SUMMARY section missing or empty - generating comprehensive fallback")
            inspector = state.get("inspector_result", {})
            auditor = state.get("auditor_result", {})
            object_name = inspector.get("object_identified", "component") or auditor.get("object_identified", "component")
            verdict_str = verdict.get("verdict", "UNKNOWN")
            agreement_score = consensus.get("agreement_score", 0)
            models_agree = consensus.get("models_agree", False)
            
            # Build comprehensive summary from all available data
            fallback_summary = (
                f"EXECUTIVE SUMMARY\n\n"
                f"Inspection of {object_name} identified {defect_count} defect(s). "
                f"Final verdict: {verdict_str}. "
                f"Both Inspector ({config.vlm_inspector_model.split('/')[-1]}) and Auditor ({config.vlm_auditor_model.split('/')[-1]}) models analyzed the image independently.\n\n"
            )
            
            if defect_count > 0:
                if critical_count > 0:
                    fallback_summary += (
                        f"CRITICAL FINDINGS: {critical_count} critical defect(s) were detected, requiring immediate attention. "
                    )
                moderate_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "MODERATE")
                cosmetic_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "COSMETIC")
                
                if moderate_count > 0:
                    fallback_summary += f"{moderate_count} moderate defect(s) and "
                if cosmetic_count > 0:
                    fallback_summary += f"{cosmetic_count} cosmetic defect(s) were also identified. "
                
                # Add agreement information
                if models_agree:
                    fallback_summary += f"Both models agreed on the findings (agreement score: {agreement_score:.0%}). "
                else:
                    fallback_summary += f"Models showed some disagreement (agreement score: {agreement_score:.0%}). "
            else:
                fallback_summary += "No defects were detected. "
                if models_agree:
                    fallback_summary += f"Both models confirmed the item is defect-free (agreement score: {agreement_score:.0%}). "
            
            # Add verdict implications
            if verdict_str == "UNSAFE":
                fallback_summary += "\n\nFINAL RECOMMENDATION\n\n"
                fallback_summary += f"Verdict: {verdict_str}\n"
                fallback_summary += "Action Required: Further inspection or remediation recommended. "
                fallback_summary += "The detected defects pose safety concerns that require attention before use."
            elif verdict_str == "SAFE":
                fallback_summary += "\n\nFINAL RECOMMENDATION\n\n"
                fallback_summary += f"Verdict: {verdict_str}\n"
                fallback_summary += "Action Required: No immediate action required. "
                fallback_summary += "The item appears safe for use based on the analysis."
            else:
                fallback_summary += "\n\nFINAL RECOMMENDATION\n\n"
                fallback_summary += f"Verdict: {verdict_str}\n"
                fallback_summary += "Action Required: Human review recommended to make final determination."
            
            sections["SUMMARY"] = fallback_summary
            logger.info("Generated comprehensive fallback SUMMARY from structured data")
        
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
        # Include FINAL RECOMMENDATION if present (may be parsed separately)
        section_order = [
            "SUMMARY",
            "EXECUTIVE SUMMARY",  # Also check for this variant
            "KEY TAKEAWAYS",
            "RECOMMENDATIONS",
            "FINAL RECOMMENDATION",  # Ensure this is included
            "REASONING CHAINS",
            "INSPECTOR ANALYSIS",
            "AUDITOR VERIFICATION",
            "COUNTERFACTUAL"
        ]
        
        # ALWAYS enhance SUMMARY with structured data for comprehensive details
        # Even if summary exists, add structured details to make it more comprehensive
        inspector = state.get("inspector_result", {})
        auditor = state.get("auditor_result", {})
        object_name = inspector.get("object_identified") or auditor.get("object_identified", "component")
        agreement_score = consensus.get("agreement_score", 0)
        models_agree = consensus.get("models_agree", False)
        inspector_conf = inspector.get("overall_confidence", "unknown").title()
        auditor_conf = auditor.get("overall_confidence", "unknown").title()
        moderate_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "MODERATE")
        cosmetic_count = sum(1 for d in consensus.get("combined_defects", []) if d.get("safety_impact") == "COSMETIC")
        
        # Get model names
        from utils.config import Config
        config = Config()
        inspector_model = config.vlm_inspector_model.split('/')[-1] if config.vlm_inspector_model else "Unknown"
        auditor_model = config.vlm_auditor_model.split('/')[-1] if config.vlm_auditor_model else "Unknown"
        
        # Get defect details for comprehensive summary
        defects = consensus.get("combined_defects", [])
        defect_types = {}
        for defect in defects:
            defect_type = defect.get("type", "Unknown")
            severity = defect.get("safety_impact", "UNKNOWN")
            if defect_type not in defect_types:
                defect_types[defect_type] = {"CRITICAL": 0, "MODERATE": 0, "COSMETIC": 0}
            defect_types[defect_type][severity] = defect_types[defect_type].get(severity, 0) + 1
        
        # Get processing time
        processing_time = state.get("processing_time", 0)
        
        if "SUMMARY" in sections and sections["SUMMARY"].strip():
            summary_text = sections["SUMMARY"].strip()
            # Enhance with structured data for comprehensive details
            enhanced_summary = (
                f"{summary_text}\n\n"
                f"═══════════════════════════════════════════════════════════════\n"
                f"DETAILED INSPECTION FINDINGS\n"
                f"═══════════════════════════════════════════════════════════════\n\n"
                f"OBJECT ANALYSIS:\n"
                f"• Object Identified: {object_name}\n"
                f"• Criticality Level: {context.get('criticality', 'unknown').upper()}\n"
                f"• Domain: {context.get('domain') or 'General'}\n"
                f"• Processing Time: {processing_time:.2f} seconds\n\n"
                f"DEFECT ANALYSIS:\n"
                f"• Total Defects Found: {defect_count}\n"
                f"  - Critical Defects: {critical_count} (require immediate attention)\n"
                f"  - Moderate Defects: {moderate_count} (need monitoring)\n"
                f"  - Cosmetic Defects: {cosmetic_count} (documentation only)\n\n"
            )
            
            # Add defect type breakdown
            if defect_types:
                enhanced_summary += f"DEFECT TYPE BREAKDOWN:\n"
                for defect_type, counts in defect_types.items():
                    total = sum(counts.values())
                    enhanced_summary += f"• {defect_type}: {total} total"
                    if counts["CRITICAL"] > 0:
                        enhanced_summary += f" ({counts['CRITICAL']} critical"
                        if counts["MODERATE"] > 0 or counts["COSMETIC"] > 0:
                            enhanced_summary += f", {counts['MODERATE']} moderate, {counts['COSMETIC']} cosmetic"
                        enhanced_summary += ")"
                    elif counts["MODERATE"] > 0:
                        enhanced_summary += f" ({counts['MODERATE']} moderate"
                        if counts["COSMETIC"] > 0:
                            enhanced_summary += f", {counts['COSMETIC']} cosmetic"
                        enhanced_summary += ")"
                    elif counts["COSMETIC"] > 0:
                        enhanced_summary += f" ({counts['COSMETIC']} cosmetic)"
                    enhanced_summary += "\n"
                enhanced_summary += "\n"
            
            enhanced_summary += (
                f"MODEL CONSENSUS ANALYSIS:\n"
                f"• Inspector Model: {inspector_model}\n"
                f"• Auditor Model: {auditor_model}\n"
                f"• Models Agreement: {'✓ Agreed' if models_agree else '⚠ Disagreed'}\n"
                f"• Agreement Score: {agreement_score:.1%}\n"
                f"• Inspector Confidence: {inspector_conf}\n"
                f"• Auditor Confidence: {auditor_conf}\n\n"
                f"FINAL VERDICT:\n"
                f"• Safety Verdict: {verdict.get('verdict', 'UNKNOWN')}\n"
                f"• Human Review Required: {'Yes' if verdict.get('requires_human') else 'No'}\n"
            )
            
            if verdict.get('verdict') == "UNSAFE":
                enhanced_summary += (
                    f"• Action Required: Further inspection or remediation recommended. "
                    f"The detected defects pose safety concerns that require attention before use.\n"
                )
            elif verdict.get('verdict') == "SAFE":
                enhanced_summary += (
                    f"• Action Required: No immediate action required. "
                    f"The item appears safe for use based on the comprehensive analysis.\n"
                )
            else:
                enhanced_summary += (
                    f"• Action Required: Human review recommended to make final determination.\n"
                )
            
            sections["SUMMARY"] = enhanced_summary
            logger.info("Enhanced SUMMARY with comprehensive structured data")
        else:
            # If no SUMMARY, create comprehensive one from structured data
            comprehensive_summary = (
                f"═══════════════════════════════════════════════════════════════\n"
                f"EXECUTIVE SUMMARY\n"
                f"═══════════════════════════════════════════════════════════════\n\n"
                f"OBJECT ANALYSIS:\n"
                f"• Object Identified: {object_name}\n"
                f"• Criticality Level: {context.get('criticality', 'unknown').upper()}\n"
                f"• Domain: {context.get('domain') or 'General'}\n"
                f"• Processing Time: {processing_time:.2f} seconds\n\n"
                f"DEFECT ANALYSIS:\n"
                f"• Total Defects Found: {defect_count}\n"
                f"  - Critical Defects: {critical_count} (require immediate attention)\n"
                f"  - Moderate Defects: {moderate_count} (need monitoring)\n"
                f"  - Cosmetic Defects: {cosmetic_count} (documentation only)\n\n"
            )
            
            # Add defect type breakdown
            if defect_types:
                comprehensive_summary += f"DEFECT TYPE BREAKDOWN:\n"
                for defect_type, counts in defect_types.items():
                    total = sum(counts.values())
                    comprehensive_summary += f"• {defect_type}: {total} total"
                    if counts["CRITICAL"] > 0:
                        comprehensive_summary += f" ({counts['CRITICAL']} critical"
                        if counts["MODERATE"] > 0 or counts["COSMETIC"] > 0:
                            comprehensive_summary += f", {counts['MODERATE']} moderate, {counts['COSMETIC']} cosmetic"
                        comprehensive_summary += ")"
                    elif counts["MODERATE"] > 0:
                        comprehensive_summary += f" ({counts['MODERATE']} moderate"
                        if counts["COSMETIC"] > 0:
                            comprehensive_summary += f", {counts['COSMETIC']} cosmetic"
                        comprehensive_summary += ")"
                    elif counts["COSMETIC"] > 0:
                        comprehensive_summary += f" ({counts['COSMETIC']} cosmetic)"
                    comprehensive_summary += "\n"
                comprehensive_summary += "\n"
            
            comprehensive_summary += (
                f"MODEL CONSENSUS ANALYSIS:\n"
                f"• Inspector Model: {inspector_model}\n"
                f"• Auditor Model: {auditor_model}\n"
                f"• Models Agreement: {'✓ Agreed' if models_agree else '⚠ Disagreed'}\n"
                f"• Agreement Score: {agreement_score:.1%}\n"
                f"• Inspector Confidence: {inspector_conf}\n"
                f"• Auditor Confidence: {auditor_conf}\n\n"
                f"═══════════════════════════════════════════════════════════════\n"
                f"FINAL RECOMMENDATION\n"
                f"═══════════════════════════════════════════════════════════════\n\n"
                f"Verdict: {verdict.get('verdict', 'UNKNOWN')}\n"
                f"Human Review Required: {'Yes' if verdict.get('requires_human') else 'No'}\n\n"
            )
            
            if verdict.get('verdict') == "UNSAFE":
                comprehensive_summary += (
                    "Action Required: Further inspection or remediation recommended. "
                    "The detected defects pose safety concerns that require attention before use. "
                    "Immediate action is recommended to address critical defects identified in this inspection."
                )
            elif verdict.get('verdict') == "SAFE":
                comprehensive_summary += (
                    "Action Required: No immediate action required. "
                    "The item appears safe for use based on the comprehensive analysis performed by both "
                    "Inspector and Auditor models. All safety gates passed successfully."
                )
            else:
                comprehensive_summary += (
                    "Action Required: Human review recommended to make final determination. "
                    "The automated analysis requires expert evaluation to confirm the safety verdict."
                )
            
            sections["SUMMARY"] = comprehensive_summary
            logger.info("Created comprehensive SUMMARY from structured data")
        
        # Track which sections we've displayed
        displayed_sections = set()
        
        for section_name in section_order:
            if section_name in sections:
                section_content = sections[section_name].strip()
                if not section_content:
                    continue
                
                displayed_sections.add(section_name)
                    
                # Use user-friendly display names
                display_name = section_name.replace("_", " ").title()
                elements.append(Paragraph(f"<b>{display_name}</b>", section_header_style))
                
                # Special formatting for specific sections
                if section_name in ["RECOMMENDATIONS", "KEY TAKEAWAYS"]:
                    # Format bullet points - split by common bullet markers
                    # Split by bullet patterns: "* ", "• ", "- ", "1. ", etc.
                    lines = re.split(r'\n(?:\*|•|-|[0-9]+\.)\s+', section_content)
                    if lines and lines[0].strip():
                        # First line without bullet
                        elements.append(Paragraph(lines[0], self.styles["CustomBodyText"]))
                    # Remaining lines as bullets
                    for line in lines[1:]:
                        line = line.strip()
                        if line:
                            # Use paragraph with bullet list style
                            bullet_text = f"• {line}"
                            elements.append(Paragraph(bullet_text, self.styles["CustomBodyText"]))
                elif section_name in ["REASONING CHAINS", "INSPECTOR ANALYSIS", "AUDITOR VERIFICATION"]:
                    # For analysis sections, preserve structure with monospace-style formatting
                    lines = section_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Preserve numbered lists and indentation
                            if re.match(r'^\d+\.', line):
                                elements.append(Paragraph(f"<b>{line}</b>", self.styles["CustomBodyText"]))
                            else:
                                elements.append(Paragraph(line, self.styles["CustomBodyText"]))
                else:
                    # Default: display as paragraph with preserved structure
                    # For SUMMARY section, preserve line breaks for better readability
                    if section_name == "SUMMARY":
                        # Split by lines to preserve structure (especially for detailed summaries)
                        lines = section_content.split("\n")
                        current_para = []
                        for line in lines:
                            line = line.strip()
                            if not line:
                                # Empty line - end current paragraph
                                if current_para:
                                    para_text = " ".join(current_para)
                                    elements.append(Paragraph(para_text, self.styles["CustomBodyText"]))
                                    current_para = []
                                elements.append(Spacer(1, 0.1 * inch))
                            elif line.startswith("•") or line.startswith("-") or line.startswith("*"):
                                # Bullet point - end current para and add bullet
                                if current_para:
                                    para_text = " ".join(current_para)
                                    elements.append(Paragraph(para_text, self.styles["CustomBodyText"]))
                                    current_para = []
                                elements.append(Paragraph(line, self.styles["CustomBodyText"]))
                            elif line.startswith("═") or line.startswith("="):
                                # Separator line - skip or make it bold
                                if current_para:
                                    para_text = " ".join(current_para)
                                    elements.append(Paragraph(para_text, self.styles["CustomBodyText"]))
                                    current_para = []
                                # Skip separator lines or make them subtle
                            else:
                                # Regular line - add to current paragraph
                                current_para.append(line)
                        # End last paragraph
                        if current_para:
                            para_text = " ".join(current_para)
                            elements.append(Paragraph(para_text, self.styles["CustomBodyText"]))
                    else:
                        # Split into logical chunks (paragraphs separated by blank lines)
                        paragraphs = section_content.split("\n\n")
                        for para in paragraphs:
                            para = para.strip().replace("\n", " ")  # Normalize whitespace
                            if para:
                                elements.append(Paragraph(para, self.styles["CustomBodyText"]))
                
                elements.append(Spacer(1, 0.15 * inch))
        
        # Display any additional sections that weren't in the predefined order
        additional_sections = set(sections.keys()) - set(displayed_sections)
        if additional_sections:
            logger.info(f"Displaying additional sections: {additional_sections}")
            for section_name in additional_sections:
                section_content = sections[section_name].strip()
                if not section_content:
                    continue
                
                # Use user-friendly display names
                display_name = section_name.replace("_", " ").title()
                elements.append(Paragraph(f"<b>{display_name}</b>", section_header_style))
                
                # Default formatting for unexpected sections
                paragraphs = section_content.split("\n\n")
                for para in paragraphs:
                    para = para.strip().replace("\n", " ")  # Normalize whitespace
                    if para:
                        elements.append(Paragraph(para, self.styles["CustomBodyText"]))
                
                elements.append(Spacer(1, 0.15 * inch))
        
        # If no sections were parsed or explanation is empty, display full explanation text
        if not displayed_sections and not additional_sections:
            logger.warning("No sections were parsed from explanation. Displaying full explanation text.")
            if explanation and explanation.strip():
                # Display the full explanation text, preserving structure
                display_text = explanation.strip()
                # Clean markdown but preserve structure
                display_text = display_text.replace("**", "").replace("##", "").replace("#", "")
                # Split into paragraphs for better readability
                paragraphs = display_text.split("\n\n")
                for para in paragraphs:
                    para = para.strip()
                    if para:
                        # Preserve line breaks within paragraphs
                        para = para.replace("\n", " ")
                        elements.append(Paragraph(para, self.styles["CustomBodyText"]))
            else:
                elements.append(Paragraph("Explanation not available.", self.styles["CustomBodyText"]))
        
        elements.append(Spacer(1, 0.2 * inch))
        
        return elements
    
    def _build_visual_evidence(self, state: Dict[str, Any]) -> List:
        """Build evidence section with 3-panel visual layout."""
        elements = []
        
        elements.append(Paragraph("Visual Evidence", self.styles["SectionHeader"]))
        
        image_path = Path(state.get("image_path", ""))
        if not image_path.exists():
            elements.append(Paragraph("Image not available", self.styles["CustomBodyText"]))
            return elements
        
        consensus = state.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        
        try:
            # Panel 1: Original image
            # Panel 2: Heatmap with severity highlighting
            # Panel 3: Numbered annotation markers
            
            heatmap_path = REPORT_DIR / f"heatmap_{image_path.stem}.jpg"
            annotated_path = REPORT_DIR / f"annotated_{image_path.stem}.jpg"
            
            # Get context for confidence filtering
            context = state.get("context", {})
            criticality = context.get("criticality", "medium")
            actual_model_size = config.max_image_dimension  # From config (default 2048)
            
            # Create heatmap overlay (red hotspots on defects)
            # Pass actual_model_size and filtering parameters
            create_heatmap_overlay(
                image_path, 
                defects, 
                heatmap_path,
                actual_model_size=actual_model_size,
                confidence_threshold="low",  # Include all defects, but visual distinction by confidence
                criticality=criticality
            )
            
            # Create numbered annotation image
            if defects:
                boxes = []
                for i, defect in enumerate(defects, 1):
                    bbox = defect.get("bbox")
                    # Handle case where bbox is None or not a dict
                    if bbox is None or not isinstance(bbox, dict):
                        # Skip defects without valid bbox
                        self.logger.warning(f"Defect #{i} has no valid bbox, skipping annotation")
                        continue
                    boxes.append({
                        "x": bbox.get("x", 0),
                        "y": bbox.get("y", 0),
                        "width": bbox.get("width", 0),
                        "height": bbox.get("height", 0),
                        "label": f"#{i}",
                        "severity": defect.get("safety_impact", "MODERATE"),
                        "confidence": defect.get("confidence", "medium")  # Include confidence for filtering
                    })
                
                # Only draw boxes if we have valid ones
                if boxes:
                    # Pass criticality and confidence_threshold for filtering
                    draw_bounding_boxes(
                        image_path, 
                        boxes, 
                        annotated_path,
                        confidence_threshold="low",  # Include all, but visual distinction
                        criticality=criticality
                    )
                else:
                    # No valid boxes - just copy original
                    import shutil
                    shutil.copy(image_path, annotated_path)
            else:
                # No defects - just copy original
                import shutil
                shutil.copy(image_path, annotated_path)
            
            # Create 3-panel comparison using table
            panel_width = 2.1 * inch
            panel_height = 1.6 * inch
            
            panels = []
            
            # Load original
            if image_path.exists():
                panels.append(RLImage(str(image_path), width=panel_width, height=panel_height, kind="proportional"))
            else:
                panels.append(Paragraph("N/A", self.styles["Normal"]))
            
            # Load heatmap
            if heatmap_path.exists():
                panels.append(RLImage(str(heatmap_path), width=panel_width, height=panel_height, kind="proportional"))
            else:
                panels.append(Paragraph("N/A", self.styles["Normal"]))
            
            # Load annotated
            if annotated_path.exists():
                panels.append(RLImage(str(annotated_path), width=panel_width, height=panel_height, kind="proportional"))
            else:
                panels.append(Paragraph("N/A", self.styles["Normal"]))
            
            # Build the 3-panel table
            panel_table = Table([panels], colWidths=[panel_width + 0.1*inch] * 3)
            panel_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(panel_table)
            elements.append(Spacer(1, 0.05 * inch))
            
            # Captions for each panel
            captions = ["1. Original Image", "2. Defect Heatmap", "3. Numbered Markers"]
            caption_table = Table([captions], colWidths=[panel_width + 0.1*inch] * 3)
            caption_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, -1), BRAND_GRAY),
            ]))
            elements.append(caption_table)
            elements.append(Spacer(1, 0.1 * inch))
            
            # Legend
            if defects:
                legend_text = "<b>Legend:</b> "
                for i, defect in enumerate(defects, 1):
                    severity = defect.get("safety_impact", "UNKNOWN")
                    defect_type = defect.get("type", "unknown")
                    legend_text += f"<b>#{i}</b> = {defect_type.title()} ({severity}) &nbsp;&nbsp;"
                elements.append(Paragraph(legend_text, ParagraphStyle(
                    name="Legend",
                    parent=self.styles["Normal"],
                    fontSize=8,
                    textColor=BRAND_GRAY,
                )))
                
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
            # Wrap long text in Paragraphs to prevent truncation
            text_style = ParagraphStyle(
                name="TableText",
                parent=self.styles["Normal"],
                fontSize=9,
                leading=11,
            )
            
            # Format confidence as numeric percentage
            conf_str = defect.get("confidence", "unknown")
            conf_numeric = {"high": "90%", "medium": "60%", "low": "30%"}.get(conf_str.lower(), conf_str)
            
            details = [
                ["Severity", severity],
                ["Location", Paragraph(defect.get("location", "Not specified"), text_style)],
                ["Confidence", f"{conf_str.title()} ({conf_numeric})"],
                ["Reasoning", Paragraph(defect.get("reasoning", "Not provided"), text_style)],
                ["Recommendation", Paragraph(defect.get("recommended_action", "Further inspection recommended"), text_style)]
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
        """Build comprehensive audit trail with ALL safety gates evaluation."""
        elements = []
        
        elements.append(Paragraph("Safety Gates Evaluation", self.styles["SectionHeader"]))
        
        verdict = state.get("safety_verdict", {})
        consensus = state.get("consensus", {})
        defect_summary = verdict.get("defect_summary", {})
        
        # Get all gate results if available
        all_gate_results = defect_summary.get("all_gate_results", [])
        
        if all_gate_results:
            # Build table showing ALL gates with pass/fail
            gate_data = [["Gate", "Status", "Details"]]
            
            for gate in all_gate_results:
                gate_name = gate.get("display_name", gate.get("gate_id", "Unknown"))
                passed = gate.get("passed", True)
                message = gate.get("message", "")
                
                status = "✓ PASSED" if passed else "✗ FAILED"
                gate_data.append([gate_name, status, message])
            
            table = Table(gate_data, colWidths=[2.2 * inch, 1 * inch, 3.3 * inch])
            table.setStyle(TableStyle([
                # Header
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                # Data rows
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 1, BRAND_GRAY),
                ("GRID", (0, 0), (-1, -1), 0.5, BRAND_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]))
            
            # Color the status column based on pass/fail
            for i, gate in enumerate(all_gate_results, 1):
                if gate.get("passed", True):
                    table.setStyle(TableStyle([
                        ("TEXTCOLOR", (1, i), (1, i), BRAND_SUCCESS),
                        ("FONTNAME", (1, i), (1, i), "Helvetica-Bold"),
                    ]))
                else:
                    table.setStyle(TableStyle([
                        ("TEXTCOLOR", (1, i), (1, i), BRAND_DANGER),
                        ("FONTNAME", (1, i), (1, i), "Helvetica-Bold"),
                        ("BACKGROUND", (0, i), (-1, i), HexColor("#fff5f5")),
                    ]))
            
            elements.append(table)
            elements.append(Spacer(1, 0.15 * inch))
        else:
            # Fallback: Show triggered gates only
            gates = verdict.get("triggered_gates", [])
            if gates:
                elements.append(Paragraph("<b>Safety Gates Triggered:</b>", self.styles["SubHeader"]))
                for gate in gates:
                    elements.append(Paragraph(f"• {gate}", self.styles["CustomBodyText"]))
                elements.append(Spacer(1, 0.1 * inch))
        
        # Add disagreement analysis if models disagree
        if not consensus.get("models_agree", True):
            elements.append(Paragraph("<b>Disagreement Analysis:</b>", self.styles["SubHeader"]))
            
            inspector_defects = len(state.get("inspector_result", {}).get("defects", []))
            auditor_defects = len(state.get("auditor_result", {}).get("defects", []))
            
            disagreement_text = f"""
            <b>Point of Disagreement:</b> Defect Count<br/>
            • Inspector identified: {inspector_defects} defect(s)<br/>
            • Auditor identified: {auditor_defects} defect(s)<br/>
            • Agreement Score: {consensus.get('agreement_score', 0):.0%}<br/>
            • Resolution: Defer to more thorough analysis, combined {len(consensus.get('combined_defects', []))} unique defects<br/>
            """
            elements.append(Paragraph(disagreement_text, self.styles["CustomBodyText"]))
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