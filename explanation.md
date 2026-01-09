# Vision Inspection System: Detailed File Explanations

This document provides a comprehensive explanation of every file and folder in the repository. Each section describes the purpose, logic, and key implementation details, enabling you to answer questions about any part of the codebase.

---

## Root Directory

### main.py
- **Purpose:** Entry point or script for launching the application or running a demo/test. 
- **Logic:** May initialize the app, load configuration, or run a main loop. (Check for `if __name__ == "__main__":` to confirm.)

### requirements.txt / pyproject.toml / uv.lock
- **Purpose:** Manage Python dependencies. 
- **Logic:**
  - `requirements.txt`: List of required packages (legacy or pip-based installs).
  - `pyproject.toml`: Modern Python project configuration (dependencies, build system, metadata).
  - `uv.lock`: Lock file for reproducible installs (used by `uv`).

### README.md
- **Purpose:** Project overview, setup instructions, and usage guide.

---

## app/

### app/main.py
- **Purpose:** Likely the main application runner (e.g., starts the UI or API server).
- **Logic:** Sets up the app, loads UI, and connects services.

### app/ui.py
- **Purpose:** UI logic and orchestration.
- **Logic:** Handles layout, event handling, and UI state.

#### app/components/
- **Purpose:** Modular UI components.
- **Files:**
  - `chat_widget.py`: Chat interface for user/AI interaction.
  - `decision_support.py`: UI for repair/replace recommendations.
  - `image_upload.py`: Handles image upload and validation.
  - `inspection_progress.py`: Displays inspection workflow progress.
  - `results_view.py`: Shows inspection results and verdicts.
  - `sidebar.py`: Navigation or context panel.
  - `verdict_display.py`: Visualizes final inspection verdict.

#### app/services/
- **Purpose:** Backend logic for app features.
- **Files:**
  - `file_handler.py`: File I/O, uploads, and storage.
  - `session_manager.py`: Manages user/session state.

#### app/styles/
- **Purpose:** Custom CSS for UI theming.
- **Files:**
  - `custom.css`: Style overrides and branding.

---

## config/
- **Purpose:** Configuration files for models, prompts, and safety rules.
- **Files:**
  - `models.yaml`: Model selection and parameters.
  - `prompts.yaml`: Prompt templates for AI agents.
  - `safety_rules.yaml`: Safety/quality rules for inspections.

---

## data/
- **Purpose:** Persistent data storage.
- **Files/Folders:**
  - `inspections.db`: SQLite database for inspection records.
  - `logs/`: Log files for audits and debugging.
  - `reports/`: Generated PDF or image reports.
  - `uploads/`: Uploaded images for inspection.

---

## src/
- **Purpose:** Main source code for business logic, agents, orchestration, and utilities.

### src/agents/
- **Purpose:** AI agent classes for inspection and explanation.
- **Files:**
  - `base.py`: Base class for all agents (common logic, interfaces).
  - `explainer.py`: Agent for generating human-readable explanations.
  - `vlm_auditor.py`: Vision-Language Model (VLM) agent for audit/verification.
  - `vlm_inspector.py`: VLM agent for primary inspection.

### src/database/
- **Purpose:** Database models and access logic.
- **Files:**
  - `models.py`: ORM or schema definitions for inspection data.
  - `repository.py`: Data access layer (CRUD operations).

### src/orchestration/
- **Purpose:** Workflow and state management.
- **Files:**
  - `graph.py`: Workflow graph or DAG for inspection steps.
  - `nodes.py`: Individual workflow nodes (actions, decisions).
  - `session_aggregation.py`: Aggregates session data for reporting.
  - `state.py`: Manages workflow state and transitions.

### src/reporting/
- **Purpose:** Report generation and export.
- **Files:**
  - `pdf_generator.py`: Generates professional PDF reports with annotated images, summaries, and audit trails.

### src/safety/
- **Purpose:** Safety and quality logic.
- **Files:**
  - `consensus.py`: Aggregates multiple agent results for consensus.
  - `gates.py`: Safety gates/checks for inspection results.
  - `image_quality.py`: Validates image quality before inspection.

### src/schemas/
- **Purpose:** Data schemas for API or internal use.
- **Files:**
  - `models.py`: Pydantic or similar schemas for validation and serialization.

---

## tests/
- **Purpose:** Unit and integration tests.
- **Files:**
  - `test_safety_gates.py`: Tests for safety gate logic.
  - `conftest.py`: Pytest fixtures and setup.

---

## utils/
- **Purpose:** Utility functions and helpers.
- **Files:**
  - `config.py`: Loads and manages configuration settings.
  - `image_utils.py`: Image loading, resizing, annotation, and validation (OpenCV/PIL logic for bounding boxes, heatmaps, etc.).
  - `logger.py`: Logging setup and helpers.
  - `prompts.py`: Prompt management for AI agents.
  - `validators.py`: Input and config validation functions.

---

## company_logo.jpg / Mouri.jpg
- **Purpose:** Branding images for reports and UI.

---

This document is designed to help you answer any file-specific questions from your manager. For deeper technical details, refer to the code and docstrings in each file.