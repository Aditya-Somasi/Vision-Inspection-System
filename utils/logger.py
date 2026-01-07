"""
Enhanced logging with rich formatting and colorlog.
Provides Spring Boot-style logging with beautiful terminal output.
"""

import logging
import sys
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

import colorlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Global console for rich output
console = Console()

# Request ID context for correlation
_request_context = {}


def get_request_id() -> str:
    """Get or create request ID for current context."""
    if "request_id" not in _request_context:
        _request_context["request_id"] = str(uuid.uuid4())[:8]
    return _request_context["request_id"]


def set_request_id(request_id: str):
    """Set request ID for current context."""
    _request_context["request_id"] = request_id


def clear_request_id():
    """Clear request ID from context."""
    _request_context.clear()


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data like API keys in log messages."""
    
    # Patterns to mask (key prefix -> replacement)
    SENSITIVE_PATTERNS = [
        ("hf_", "hf_***MASKED***"),
        ("gsk_", "gsk_***MASKED***"),
        ("sk-", "sk-***MASKED***"),
        ("api_key=", "api_key=***MASKED***"),
        ("API_KEY=", "API_KEY=***MASKED***"),
        ("token=", "token=***MASKED***"),
    ]
    
    def filter(self, record):
        if hasattr(record, 'msg') and record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                if pattern in msg:
                    # Find and mask the sensitive value
                    import re
                    # Match pattern followed by alphanumeric characters
                    regex = rf"({re.escape(pattern)})([a-zA-Z0-9_-]+)"
                    msg = re.sub(regex, replacement, msg)
            record.msg = msg
        return True


class ContextFilter(logging.Filter):
    """Add request ID and component name to log records."""
    
    def __init__(self, component: str = "SYSTEM"):
        super().__init__()
        self.component = component
    
    def filter(self, record):
        record.request_id = get_request_id()
        record.component = self.component
        return True


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    component: str = None
) -> logging.Logger:
    """
    Setup logger with colorlog and rich formatting.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        component: Component name for contextualized logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Component name (use module name if not specified)
    comp = component or name.split(".")[-1].upper()
    
    # Console handler with colorlog
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Color formatter (Spring Boot style)
    console_formatter = colorlog.ColoredFormatter(
        fmt=(
            "%(log_color)s[%(asctime)s.%(msecs)03d] "
            "%(levelname)-8s "
            "%(white)s[%(request_id)s] "
            "%(cyan)s[%(component)s] "
            "%(message_log_color)s%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={
            "message": {
                "DEBUG": "white",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            }
        },
        reset=True,
        style="%"
    )
    
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ContextFilter(comp))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        
        # JSON-style formatter for file (easier parsing)
        file_formatter = logging.Formatter(
            fmt=(
                '{"timestamp":"%(asctime)s.%(msecs)03d",'
                '"level":"%(levelname)s",'
                '"request_id":"%(request_id)s",'
                '"component":"%(component)s",'
                '"logger":"%(name)s",'
                '"message":"%(message)s"}'
            ),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(ContextFilter(comp))
        logger.addHandler(file_handler)
    
    return logger


def print_banner():
    """Print application startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║   🔍  VISION INSPECTION SYSTEM  v1.0.0                  ║
║   AI-Powered Damage Detection & Safety Analysis         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    console.print(banner, style="bold cyan")


def print_health_check_table(checks: dict):
    """
    Print health check results in a formatted table.
    
    Args:
        checks: Dict of check_name -> (status: bool, details: str)
    """
    table = Table(title="🏥 System Health Checks", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=30)
    table.add_column("Status", width=12)
    table.add_column("Details", style="dim")
    
    for component, (status, details) in checks.items():
        status_icon = "✓ READY" if status else "✗ FAILED"
        status_style = "green" if status else "red"
        table.add_row(
            component,
            f"[{status_style}]{status_icon}[/{status_style}]",
            details
        )
    
    console.print(table)


def print_summary_panel(title: str, content: dict, style: str = "green"):
    """
    Print summary information in a panel.
    
    Args:
        title: Panel title
        content: Dict of key-value pairs to display
        style: Panel border style (green, yellow, red, cyan)
    """
    text = "\n".join([f"[bold]{k}:[/bold] {v}" for k, v in content.items()])
    panel = Panel(text, title=title, border_style=style, expand=False)
    console.print(panel)


def print_processing_status(
    image_id: str,
    step: str,
    progress: int,
    total: int,
    details: Optional[dict] = None
):
    """
    Print processing status update.
    
    Args:
        image_id: Image identifier
        step: Current processing step
        progress: Current progress (0-100)
        total: Total steps
        details: Optional additional details
    """
    progress_bar = "■" * (progress // 10) + "░" * (10 - progress // 10)
    
    status_text = f"[cyan]{step}[/cyan]\n"
    status_text += f"[{progress_bar}] {progress}%\n"
    
    if details:
        for k, v in details.items():
            status_text += f"• {k}: {v}\n"
    
    panel = Panel(
        status_text.strip(),
        title=f"Processing: {image_id}",
        border_style="cyan",
        expand=False
    )
    console.print(panel)


def print_inspection_result(
    verdict: str,
    defect_count: int,
    inspector_confidence: str,
    auditor_confidence: str,
    agreement: bool,
    processing_time: float,
    report_path: Optional[str] = None
):
    """
    Print final inspection result.
    
    Args:
        verdict: Safety verdict (SAFE, UNSAFE, REQUIRES_REVIEW)
        defect_count: Number of defects found
        inspector_confidence: Inspector confidence level
        auditor_confidence: Auditor confidence level
        agreement: Whether models agree
        processing_time: Total processing time in seconds
        report_path: Path to generated report
    """
    # Determine style based on verdict
    if verdict == "UNSAFE":
        style = "red"
        verdict_display = "🚫 UNSAFE - CRITICAL DEFECT DETECTED"
    elif verdict == "REQUIRES_REVIEW":
        style = "yellow"
        verdict_display = "⚠️  REQUIRES HUMAN REVIEW"
    else:
        style = "green"
        verdict_display = "✓ SAFE - NO CRITICAL DEFECTS"
    
    agreement_icon = "✓ MODELS AGREE" if agreement else "⚠ MODELS DISAGREE"
    agreement_style = "green" if agreement else "yellow"
    
    content = f"""[bold {style}]{verdict_display}[/bold {style}]

[bold]Defects Found:[/bold] {defect_count}
[bold]Inspector:[/bold] {inspector_confidence} confidence
[bold]Auditor:[/bold] {auditor_confidence} confidence
[bold]Agreement:[/bold] [{agreement_style}]{agreement_icon}[/{agreement_style}]

[bold]Processing Time:[/bold] {processing_time:.2f}s"""
    
    if report_path:
        content += f"\n[bold]Report:[/bold] {report_path}"
    
    panel = Panel(
        content,
        title="Inspection Complete",
        border_style=style,
        expand=False
    )
    console.print(panel)


def print_error(error_type: str, message: str, details: Optional[str] = None):
    """
    Print error message in formatted panel.
    
    Args:
        error_type: Type of error
        message: Error message
        details: Optional detailed error information
    """
    content = f"[bold red]{error_type}[/bold red]\n\n{message}"
    if details:
        content += f"\n\n[dim]{details}[/dim]"
    
    panel = Panel(
        content,
        title="❌ Error",
        border_style="red",
        expand=False
    )
    console.print(panel)


def create_progress_bar(description: str = "Processing"):
    """
    Create a rich progress bar for long operations.
    
    Args:
        description: Progress bar description
    
    Returns:
        Rich Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )


# Example usage logger
if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test", level="DEBUG", component="TEST")
    
    set_request_id("test123")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test rich outputs
    print_banner()
    
    print_health_check_table({
        "Qwen2-VL Inspector": (True, "342ms response time"),
        "Llama 3.2 Auditor": (True, "298ms response time"),
        "LangSmith": (True, "Connected to project: vision-qa"),
        "Database": (True, "1,247 inspection records"),
    })
    
    print_inspection_result(
        verdict="UNSAFE",
        defect_count=2,
        inspector_confidence="high",
        auditor_confidence="high",
        agreement=True,
        processing_time=3.42,
        report_path="reports/img_001_report.pdf"
    )