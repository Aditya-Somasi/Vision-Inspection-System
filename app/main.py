"""
Main entry point for Vision Inspection System.
Performs comprehensive startup health checks before launching UI.
"""

import sys
import subprocess
from pathlib import Path

from utils.logger import (
    setup_logger, print_banner, print_health_check_table,
    print_summary_panel
)
from utils.config import config
from src.agents import health_check_agents
from src.database import health_check_database, init_database

logger = setup_logger(__name__, level=config.log_level, component="MAIN")


def startup_health_checks() -> bool:
    """
    Perform comprehensive startup health checks.
    
    Returns:
        True if all checks pass, False otherwise
    """
    print_banner()
    
    logger.info("=" * 80)
    logger.info("VISION INSPECTION SYSTEM - STARTUP HEALTH CHECKS")
    logger.info("=" * 80)
    
    all_healthy = True
    health_results = {}
    
    # ====================================================================
    # 1. Configuration Validation
    # ====================================================================
    logger.info("1. Checking configuration...")
    try:
        logger.info(f"   ✓ Environment: {config.environment}")
        logger.info(f"   ✓ Inspector Model: {config.vlm_inspector_model}")
        logger.info(f"   ✓ Auditor Model: {config.vlm_auditor_model}")
        logger.info(f"   ✓ Explainer Model: {config.explainer_model}")
        logger.info(f"   ✓ Database Path: {config.database_path}")
        
        if config.langsmith_enabled:
            logger.info(f"   ✓ LangSmith Tracing: ENABLED")
            logger.info(f"   ✓ LangSmith Project: {config.langchain_project}")
        else:
            logger.info(f"   ✓ LangSmith Tracing: DISABLED")
        
        health_results["Configuration"] = (True, "All settings loaded")
    
    except Exception as e:
        logger.error(f"   ✗ Configuration validation failed: {e}")
        health_results["Configuration"] = (False, f"Error: {e}")
        all_healthy = False
    
    # ====================================================================
    # 2. File System Checks
    # ====================================================================
    logger.info("2. Checking file system...")
    try:
        # Create required directories
        config.get_upload_dir()
        config.get_report_dir()
        config.get_log_dir()
        
        logger.info("   ✓ Upload directory: writable")
        logger.info("   ✓ Report directory: writable")
        logger.info("   ✓ Log directory: writable")
        
        health_results["File System"] = (True, "All directories accessible")
    
    except Exception as e:
        logger.error(f"   ✗ File system check failed: {e}")
        health_results["File System"] = (False, f"Error: {e}")
        all_healthy = False
    
    # ====================================================================
    # 3. Database Health Check
    # ====================================================================
    logger.info("3. Checking database...")
    try:
        # Initialize database
        if init_database():
            # Test connection
            if health_check_database():
                from src.database import InspectionRepository
                repo = InspectionRepository()
                count = repo.get_inspection_count()
                
                logger.info(f"   ✓ Database initialized: {count} records")
                health_results["Database"] = (True, f"{count} inspection records")
            else:
                logger.error("   ✗ Database connection failed")
                health_results["Database"] = (False, "Connection failed")
                all_healthy = False
        else:
            logger.error("   ✗ Database initialization failed")
            health_results["Database"] = (False, "Initialization failed")
            all_healthy = False
    
    except Exception as e:
        logger.error(f"   ✗ Database check error: {e}")
        health_results["Database"] = (False, f"Error: {e}")
        all_healthy = False
    
    # ====================================================================
    # 4. VLM Agents Health Check
    # ====================================================================
    logger.info("4. Checking VLM agents...")
    try:
        agent_results = health_check_agents()
        
        for agent_name, (status, details) in agent_results.items():
            if status:
                logger.info(f"   ✓ {agent_name}: {details}")
            else:
                logger.error(f"   ✗ {agent_name}: {details}")
                all_healthy = False
        
        # Add to health results
        health_results.update(agent_results)
    
    except Exception as e:
        logger.error(f"   ✗ Agent health check error: {e}")
        health_results["VLM Agents"] = (False, f"Error: {e}")
        all_healthy = False
    
    # ====================================================================
    # 5. LangSmith Connection (Optional)
    # ====================================================================
    if config.langsmith_enabled:
        logger.info("5. Checking LangSmith connection...")
        try:
            logger.info(f"   ✓ LangSmith configured: {config.langchain_project}")
            health_results["LangSmith"] = (
                True,
                f"Project: {config.langchain_project}"
            )
        except Exception as e:
            logger.warning(f"   ⚠ LangSmith check skipped: {e}")
            health_results["LangSmith"] = (True, "Configured (will test on first use)")
    else:
        logger.info("5. LangSmith tracing: DISABLED")
        health_results["LangSmith"] = (True, "Disabled")
    
    # ====================================================================
    # Final Status
    # ====================================================================
    logger.info("=" * 80)
    
    if all_healthy:
        logger.info("✓ ALL HEALTH CHECKS PASSED - SYSTEM READY")
        print_health_check_table(health_results)
        
        # Print configuration summary
        print_summary_panel(
            "System Configuration",
            {
                "Environment": config.environment.upper(),
                "Inspector": config.vlm_inspector_model.split("/")[-1],
                "Auditor": config.vlm_auditor_model.split("/")[-1],
                "Database": config.database_path,
                "LangSmith": "Enabled" if config.langsmith_enabled else "Disabled"
            },
            style="green"
        )
    else:
        logger.error("✗ SOME HEALTH CHECKS FAILED - SYSTEM MAY NOT FUNCTION PROPERLY")
        print_health_check_table(health_results)
        
        print_summary_panel(
            "⚠️  Health Check Failures",
            {
                "Status": "FAILED",
                "Action": "Please fix the errors above before using the system"
            },
            style="red"
        )
    
    logger.info("=" * 80)
    
    return all_healthy


def main():
    """Main entry point."""
    logger.info("Starting Vision Inspection System...")
    
    # Skip health checks if configured
    if config.skip_health_checks:
        logger.warning("⚠️  Health checks SKIPPED (SKIP_HEALTH_CHECKS=true)")
    else:
        # Run health checks
        if not startup_health_checks():
            logger.error("❌ Startup health checks failed.")
            logger.error("   Cannot start with failed health checks.")
            logger.error("   Please fix the errors above and try again.")
            logger.error("")
            logger.error("   To bypass health checks (NOT RECOMMENDED), set:")
            logger.error("   SKIP_HEALTH_CHECKS=true in .env")
            sys.exit(1)
    
    # Launch Streamlit UI
    logger.info("🚀 Launching Streamlit UI...")
    
    # Get path to ui.py
    ui_path = Path(__file__).parent / "ui.py"
    
    if not ui_path.exists():
        logger.error(f"❌ UI file not found: {ui_path}")
        sys.exit(1)
    
    # Launch Streamlit in subprocess
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false"
    ]
    
    try:
        logger.info(f"   Command: {' '.join(cmd)}")
        logger.info("   Access the UI at: http://localhost:8501")
        logger.info("")
        logger.info("   Press Ctrl+C to stop the server")
        logger.info("")
        
        # Run Streamlit and forward exit code
        rc = subprocess.run(cmd).returncode
        sys.exit(rc)
    
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutting down gracefully...")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"❌ Failed to launch Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
