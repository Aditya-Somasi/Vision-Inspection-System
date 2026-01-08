# Vision Inspection System - Issues & Fixes Report

**Generated:** January 8, 2026

## 🔴 CRITICAL ISSUES

### 1. Import Error: Non-existent Class Aliases

**File:** `src/agents/__init__.py`

**Issue:** Lines 5-6 import `InspectorAgent` and `AuditorAgent` which don't exist. Only `VLMInspectorAgent` and `VLMAuditorAgent` are defined.

```python
# WRONG:
from src.agents.vlm_inspector import VLMInspectorAgent, InspectorAgent
from src.agents.vlm_auditor import VLMAuditorAgent, AuditorAgent
```

**Fix:** Remove non-existent imports or create type aliases:

```python
# CORRECT:
from src.agents.vlm_inspector import VLMInspectorAgent
from src.agents.vlm_auditor import VLMAuditorAgent
from src.agents.explainer import ExplainerAgent

# Aliases for compatibility
InspectorAgent = VLMInspectorAgent
AuditorAgent = VLMAuditorAgent
```

---

### 2. Missing Function: `health_check()` in Explainer Agent

**File:** `src/agents/explainer.py`

**Issue:** `ExplainerAgent` class doesn't implement `health_check()` method, but `health_check_agents()` in `__init__.py` (line 56) calls it. This will cause `AttributeError` at startup.

**Fix:** Add health check method to `ExplainerAgent`:

```python
def health_check(self) -> bool:
    """Perform health check on the LLM."""
    try:
        test_response = self._call_llm("Say 'health check passed' in one word.")
        return "passed" in test_response.lower() or "health" in test_response.lower()
    except Exception as e:
        self.logger.error(f"Health check failed: {e}")
        return False
```

---

### 3. Undefined Functions in Orchestration Module Exports

**File:** `src/orchestration/__init__.py`

**Issue:** Exports `run_inspection_streaming`, `resume_inspection`, `get_pending_reviews` which are not defined in `graph.py`. Only `run_inspection` exists.

```python
from src.orchestration.graph import (
    create_inspection_workflow,
    run_inspection,
    run_inspection_streaming,      # ❌ DOESN'T EXIST
    resume_inspection,              # ❌ DOESN'T EXIST
    get_pending_reviews,            # ❌ DOESN'T EXIST
)
```

**Fix:** Export only what exists or implement the missing functions. If they're not used, remove them:

```python
from src.orchestration.graph import (
    create_inspection_workflow,
    run_inspection,
)
```

---

### 4. Missing Function: `get_memory_manager()` and `get_session_history()`

**File:** `app/ui.py` (lines 23-24)

**Issue:** These functions are imported from `src.chat_memory` but don't exist in that module. Only `SQLiteChatHistory` class and utility functions exist.

```python
from src.chat_memory import get_memory_manager, get_session_history
```

**Fix:** Implement missing factory functions in `src/chat_memory.py`:

```python
def get_memory_manager(session_id: str):
    """Get or create memory manager for session."""
    return SQLiteChatHistory(session_id)

def get_session_history(session_id: str):
    """Get chat history for session."""
    return SQLiteChatHistory(session_id).messages
```

---

### 5. Database Initialization on Import (Side Effect)

**File:** `src/database/repository.py` (lines 21-27)

**Issue:** Database engine and session factory are created at import time, causing automatic database file creation and connection before it's needed.

```python
# Lines 21-27 execute immediately on import
engine = create_engine(...)
SessionLocal = sessionmaker(...)
```

**Fix:** Lazy-load database connection:

```python
# At module level
_engine = None
_session_factory = None

def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            f"sqlite:///{config.database_path}",
            echo=config.database_echo,
            connect_args={"check_same_thread": False}
        )
    return _engine

def _get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=_get_engine()
        )
    return _session_factory

class InspectionRepository:
    def get_session(self) -> Session:
        return _get_session_factory()()
```

---

### 6. Incomplete Function: `validate_inspection_context()` in utils

**File:** `utils/config.py` OR missing in `utils/__init__.py` and `utils/validators.py`

**Issue:** `utils/__init__.py` (line 42) exports `validate_inspection_context` which doesn't exist in `validators.py`. Only `validate_criticality`, `validate_domain`, `validate_image_path`, `validate_user_notes` exist.

**Fix:** Either remove from `__all__` or implement it:

```python
def validate_inspection_context(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate inspection context dict."""
    required_fields = ["image_id", "criticality"]
    for field in required_fields:
        if field not in context:
            return False, f"Missing required field: {field}"
    
    is_valid, error, _ = validate_criticality(context["criticality"])
    if not is_valid:
        return False, error
    
    return True, None
```

---

### 7. Missing Prompt Templates in utils/prompts.py

**File:** `utils/__init__.py` (lines 30-34)

**Issue:** Exports `get_prompt` function which doesn't exist in `utils/prompts.py`. File contains `INSPECTOR_PROMPT`, `AUDITOR_PROMPT` but no `EXPLAINER_PROMPT` constant or `get_prompt` function.

```python
from utils.prompts import (
    INSPECTOR_PROMPT,
    AUDITOR_PROMPT,
    EXPLAINER_PROMPT,    # ❌ Not found
    get_prompt           # ❌ Function doesn't exist
)
```

**Fix:** In `utils/prompts.py`, add missing constant and function:

```python
EXPLAINER_PROMPT = """You are a technical writer creating safety inspection reports..."""

def get_prompt(prompt_type: str, **kwargs) -> str:
    """Get and format prompt template."""
    prompts = {
        "inspector": INSPECTOR_PROMPT,
        "auditor": AUDITOR_PROMPT,
        "explainer": EXPLAINER_PROMPT,
    }
    template = prompts.get(prompt_type.lower())
    if not template:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return template.format(**kwargs) if kwargs else template
```

---

### 8. Async Mismatch: `health_check_database()` Expected But Not Found

**File:** `src/database/__init__.py` and `src/database/repository.py`

**Issue:** `__init__.py` (line 9) exports `health_check_database` but it's not defined in `repository.py`. Function `health_check()` might exist with different name.

**Fix:** Verify function exists or implement:

```python
# In src/database/repository.py
def health_check_database() -> bool:
    """Check if database is healthy."""
    try:
        session = SessionLocal()
        # Try a simple query
        session.execute("SELECT 1")
        session.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
```

---

### 9. Undefined Function: `init_database()`

**File:** `src/database/__init__.py` (line 8) and `app/main.py` (line 94)

**Issue:** `init_database()` is exported and called, but not defined in `repository.py`.

**Fix:** Add to `src/database/repository.py`:

```python
def init_database() -> bool:
    """Initialize database and create tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False
```

---

## 🟠 HIGH PRIORITY ISSUES

### 10. Model Aliases in Config vs Actual Models

**File:** `config/models.yaml` and `utils/config.py`

**Issue:** `config/models.yaml` is orphaned—configuration comes from environment variables via Pydantic. The YAML file is read but never used (seen in `graph.py` and health checks).

**Fix:** Either:
- Delete unused `config/models.yaml`, OR
- Integrate it: Load YAML in `utils/config.py` to override defaults

**Recommended:** Delete if not used, or standardize on env vars only.

---

### 11. Hardcoded Font Path in PDF Generator

**File:** `src/reporting/pdf_generator.py` (likely around image drawing)

**Issue:** May use `C:/Windows/Fonts/` or similar hardcoded path which fails on non-Windows systems.

**Fix:** Use font resolution:

```python
import matplotlib.font_manager as fm

def get_system_font():
    """Get available system font."""
    try:
        # Try to find a common font
        for font_name in ['DejaVuSans', 'Arial', 'Helvetica']:
            fonts = fm.findSystemFonts()
            matching = [f for f in fonts if font_name.lower() in f.lower()]
            if matching:
                return matching[0]
    except:
        pass
    return None  # Use PIL default
```

---

### 12. Agent Inferred Criticality Upgrade Logic Can Silently Override User Input

**File:** `src/orchestration/nodes.py` (lines 55-74)

**Issue:** Inspector can upgrade criticality level without explicit user confirmation. While flagged, this changes inspection context mid-flow which could be confusing.

**Safer approach:** Keep original criticality, store both values, and flag for review:

```python
# Store both values clearly
state["context"]["criticality_original"] = context.criticality
state["context"]["criticality_inferred_by_agent"] = result.inferred_criticality
state["context"]["criticality_upgraded"] = (
    criticality_order.get(result.inferred_criticality, 1) > 
    criticality_order.get(context.criticality, 1)
)
# Don't modify state["context"]["criticality"] - keep original for audit trail
```

---

### 13. Consensus Defect Merging Strategy Unclear

**File:** `src/schemas/models.py` (lines 108-119)

**Issue:** `combined_defects` only adds Auditor's unique defect types to Inspector's list, but what if Inspector missed a critical defect? The logic favors Inspector.

**Consider:** Merge by location proximity, not just type uniqueness:

```python
@model_validator(mode="after")
def compute_combined_defects(self):
    """Combine defects from both models, prioritizing critical ones."""
    combined = list(self.inspector_result.defects)
    
    # Add Auditor defects that don't overlap with Inspector (by location similarity)
    for auditor_defect in self.auditor_result.defects:
        # Check if similar defect already exists
        is_duplicate = any(
            self._is_similar_defect(auditor_defect, inspector_defect)
            for inspector_defect in combined
        )
        if not is_duplicate:
            combined.append(auditor_defect)
    
    # Sort by criticality (CRITICAL first)
    self.combined_defects = sorted(
        combined,
        key=lambda d: {"CRITICAL": 0, "MODERATE": 1, "COSMETIC": 2}.get(d.safety_impact, 1)
    )
    return self

def _is_similar_defect(self, d1: DefectInfo, d2: DefectInfo) -> bool:
    """Check if two defects are the same or overlapping."""
    # Same type AND location keywords overlap
    return (
        d1.type == d2.type and
        any(kw in d2.location.lower() for kw in d1.location.lower().split())
    )
```

---

### 14. Safety Gates: No Default Conservative Fallback

**File:** `src/safety/gates.py` (line 42)

**Issue:** `GATE_DEFAULT_CONSERVATIVE` constant defined but likely not used. If all gates pass, there's no conservative default to flag borderline cases.

**Fix:** Add to `SafetyGateEngine.evaluate()`:

```python
if all_gates_passed and any([
    result.agreement_score < 0.85,  # Even if models agree, it's weak
    any(d.confidence == "low" for d in defects),  # Low confidence defects
]):
    gate_results.append(GateResult(
        gate_id=GATE_DEFAULT_CONSERVATIVE,
        passed=False,
        message="Borderline case - confidence not high enough despite no failures"
    ))
    verdict = SafetyVerdict(
        verdict="REQUIRES_HUMAN_REVIEW",
        reason="Conservative fallback: low confidence consensus",
        ...
    )
```

---

### 15. Missing Bounding Box Validation

**File:** `src/schemas/models.py` (BoundingBox class)

**Issue:** BoundingBox coordinates can be 0-100 (percentage) or 0-width/height (pixels), but no validation to catch mixed formats.

**Fix:** Add strict validation:

```python
class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    format: Literal["percentage", "pixel"] = "percentage"
    
    @model_validator(mode="after")
    def validate_bounds(self):
        if self.format == "percentage":
            for val in [self.x, self.y, self.width, self.height]:
                if not 0 <= val <= 100:
                    raise ValueError(
                        f"Percentage bounds must be 0-100, got x={self.x}, y={self.y}, "
                        f"width={self.width}, height={self.height}"
                    )
        elif self.format == "pixel":
            if self.width > 5000 or self.height > 5000:
                raise ValueError("Pixel dimensions exceed reasonable limits")
        return self
```

---

### 16. LangSmith Conditional Enable/Disable Not Honored

**File:** `utils/config.py` (lines 148-151)

**Issue:** `langsmith_enabled` property checks env var, but some code might use it before it's set, or environment variable isn't properly respected in LangChain setup.

**Fix:** Ensure explicit initialization:

```python
@property
def langsmith_enabled(self) -> bool:
    """Check if LangSmith is enabled."""
    is_enabled = self.langchain_tracing_v2.lower() == "true"
    has_api_key = bool(self.langsmith_api_key)
    return is_enabled and has_api_key

class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    def __init__(self, **data):
        super().__init__(**data)
        # Explicitly set LangSmith variables if enabled
        if self.langsmith_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

---

## 🟡 MEDIUM PRIORITY ISSUES

### 17. Inconsistent Error Handling in VLM Agents

**File:** `src/agents/vlm_inspector.py` and `src/agents/vlm_auditor.py`

**Issue:** Different retry logic between Inspector (exponential backoff) and Auditor (unclear). No timeout handling.

**Fix:** Unify with timeouts:

```python
def _call_api_with_retry(
    self,
    messages: list,
    max_retries: int = 3,
    timeout: int = 60
) -> str:
    """Call API with unified retry logic."""
    for attempt in range(max_retries):
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=timeout
            )
            return response.choices[0].message.content
        except TimeoutError:
            if attempt < max_retries - 1:
                self.logger.warning(f"Timeout, retry {attempt+1}/{max_retries}")
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as e:
            # ... existing retry logic
```

---

### 18. Chat Memory SQLite Schema Not Thread-Safe

**File:** `src/chat_memory.py` (sqlite3 usage without proper locking)

**Issue:** Multiple concurrent Streamlit sessions could corrupt chat history DB with simultaneous writes.

**Fix:** Add connection pooling with thread-safety:

```python
import threading

class SQLiteChatHistory(BaseChatMessageHistory):
    _lock = threading.Lock()
    
    def add_message(self, message):
        with SQLiteChatHistory._lock:
            conn = sqlite3.connect(self.db_path, timeout=5)
            try:
                # ... insert logic
            finally:
                conn.close()
```

---

### 19. Logging Masks API Keys But Might Miss New Patterns

**File:** `utils/logger.py` (SensitiveDataFilter)

**Issue:** Hardcoded patterns. If new API key formats appear, they won't be masked.

**Fix:** Add more robust pattern:

```python
SENSITIVE_PATTERNS = [
    ("hf_", "hf_***MASKED***"),
    ("gsk_", "gsk_***MASKED***"),
    ("sk-", "sk-***MASKED***"),
    ("api_key=", "api_key=***MASKED***"),
    ("bearer ", "bearer ***MASKED***"),
    ("authorization:", "authorization: ***MASKED***"),
]

# Also add regex for generic token patterns:
generic_token = r'([a-zA-Z0-9_-]{20,})'  # Generic 20+ char token
```

---

### 20. Image Resizing Might Lose Quality Details

**File:** `utils/image_utils.py` and `src/agents/vlm_inspector.py`

**Issue:** VLM Inspector resizes to max 1024px. Small defects on large objects may become undetectable.

**Recommendation:** Add quality level configuration:

```python
class Config:
    # Add to config
    vlm_inspector_quality_level: Literal["high", "medium", "low"] = "high"
    
    @property
    def max_image_dimension(self) -> int:
        levels = {"high": 2048, "medium": 1024, "low": 512}
        return levels[self.vlm_inspector_quality_level]
```

---

### 21. Decision Support Feature Incomplete

**File:** `app/components/decision_support.py`

**Issue:** Component renders `decision_support` data but this key is never populated in workflow. Dead code.

**Fix:** Either implement decision support logic in workflow or remove the component:

```python
# In src/orchestration/nodes.py, add node:
def generate_decision_support(state: InspectionState) -> InspectionState:
    """Generate repair vs replace recommendations."""
    if state.get("safety_verdict", {}).get("verdict") != "UNSAFE":
        return state
    
    # Logic to estimate repair vs replace costs
    defects = state.get("consensus", {}).get("combined_defects", [])
    
    state["decision_support"] = {
        "recommendation": "REPAIR" if len(defects) <= 2 else "REPLACE",
        "repair_cost": "$500-1000",
        "repair_time": "2-4 hours",
        "replace_cost": "$2000-5000",
        "replace_time": "1-2 weeks",
        "reasoning": "Cost-benefit analysis..."
    }
    return state
```

---

### 22. PDF Report Path Not Returned to UI

**File:** `src/reporting/pdf_generator.py` and workflow nodes

**Issue:** PDF is generated but state doesn't propagate `report_path` correctly to UI, so download link might not appear.

**Fix:** Ensure state["report_path"] is set:

```python
def save_to_database(state: InspectionState) -> InspectionState:
    """Save inspection to database and generate report."""
    # ... existing code
    
    # Generate report
    report_path = generate_report(
        inspection_data=inspection_data,
        final_state=state
    )
    
    # CRITICAL: Set state properly
    state["report_path"] = str(report_path) if report_path else None
    
    return state
```

---

### 23. Consensus Score Calculation Might Not Match Agreement Assessment

**File:** `src/safety/consensus.py` (lines 52-61)

**Issue:** Agreement score uses weighted average, but threshold (0.7) might be arbitrary. Validation of score vs. agreement determination is unclear.

**Fix:** Document clearly or separate concerns:

```python
# Add documentation
AGREEMENT_THRESHOLD = 0.7  # Empirically determined threshold
AGREEMENT_SCORE_WEIGHTS = {
    "condition": 0.4,      # Overall condition agreement
    "types": 0.3,          # Defect type agreement
    "count": 0.2,          # Defect count similarity
    "confidence": 0.1,     # Confidence level alignment
}

def compute_agreement_score(self, ...) -> float:
    """
    Compute agreement score with explicit weights.
    Returns: 0-1 score
    """
    # ... implementation with weights
```

---

### 24. No Logging of Which Safety Gates Triggered

**File:** `src/safety/gates.py`

**Issue:** Safety verdict is determined, but audit trail of which gate caused verdict is weak. Critical for transparency.

**Fix:** Enhance logging:

```python
def evaluate(self, consensus: ConsensusResult, ...) -> SafetyVerdict:
    """Evaluate all gates."""
    triggered = []
    passed = []
    
    for gate_result in gate_results:
        if not gate_result.passed:
            triggered.append(gate_result.gate_id)
            self.logger.warning(
                f"⚠️  GATE TRIGGERED: {gate_result.display_name} - {gate_result.message}"
            )
        else:
            passed.append(gate_result.gate_id)
    
    self.logger.info(
        f"SAFETY GATES: {len(passed)} passed, {len(triggered)} triggered"
    )
    
    verdict = SafetyVerdict(
        verdict=final_verdict,
        triggered_gates=triggered,
        ...
    )
```

---

### 25. Bounding Box Visualization Might Not Match Inspector's Coordinates

**File:** `utils/image_utils.py` (drawing functions)

**Issue:** If Inspector provides percentage-based bbox (0-100) but drawing function expects pixels (0-width), overlay will be wrong. No validation.

**Fix:** Explicit coordinate conversion:

```python
def draw_bounding_boxes(
    image: Image.Image,
    defects: List[DefectInfo],
    bbox_format: Literal["percentage", "pixel"] = "percentage"
) -> Image.Image:
    """Draw bounding boxes on image."""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    for defect in defects:
        if not defect.bbox:
            continue
        
        # Convert to pixel coordinates if needed
        if bbox_format == "percentage":
            x_px = defect.bbox.x * width / 100
            y_px = defect.bbox.y * height / 100
            w_px = defect.bbox.width * width / 100
            h_px = defect.bbox.height * height / 100
        else:
            x_px, y_px, w_px, h_px = defect.bbox.x, defect.bbox.y, defect.bbox.width, defect.bbox.height
        
        # Draw box
        box = [x_px, y_px, x_px + w_px, y_px + h_px]
        draw.rectangle(box, outline="red", width=2)
```

---

### 26. Missing `decision_support` in Final Inspection Results

**File:** `app/ui.py` and workflow

**Issue:** Lines 225+ try to display decision_support, but this key is never populated in inspection results dict.

**Fix:** Add to workflow or remove from UI. If keeping, populate in nodes.

---

### 27. Chat Widget History Persists Across Inspections

**File:** `app/services/session_manager.py`

**Issue:** `reset_chat_state()` clears in-memory messages but SQLite history persists, potentially confusing users.

**Fix:** Explicitly clear old history:

```python
def reset_chat_state():
    """Reset chat state for new conversation."""
    st.session_state.chat_messages = []
    old_session_id = st.session_state.chat_session_id
    st.session_state.chat_session_id = str(uuid.uuid4())
    
    # Clear old session from DB if needed
    from src.chat_memory import SQLiteChatHistory
    try:
        history = SQLiteChatHistory(old_session_id)
        history.clear()  # Add this method
    except:
        pass
```

---

## 🟢 LOW PRIORITY / RECOMMENDATIONS

### 28. Prompt Injection Vulnerability: User Notes Not Sanitized

**File:** `utils/prompts.py` and Agent classes

**Issue:** User notes are directly interpolated into prompts without escaping:

```python
INSPECTOR_PROMPT = """...
- User Notes: {user_notes}
"""
```

**Recommendation:** Escape user input:

```python
def escape_prompt_injection(text: str) -> str:
    """Escape potential prompt injection."""
    if not text:
        return ""
    # Remove or escape special prompt tokens
    dangerous = ["```", ">>", "###", "SYSTEM:", "IGNORE:"]
    safe = text
    for token in dangerous:
        safe = safe.replace(token, f"\\{token}")
    return safe
```

---

### 29. No Timeout on VLM Inference

**File:** `src/agents/vlm_*.py`

**Issue:** API calls could hang indefinitely if network fails.

**Recommendation:** Add timeout parameter:

```python
completion = self.client.chat.completions.create(
    model=self.model_id,
    messages=messages,
    temperature=self.temperature,
    max_tokens=self.max_tokens,
    timeout=config.api_timeout  # Add this
)
```

---

### 30. No Rate Limiting Between Consecutive Inspections

**File:** App UI workflow

**Recommendation:** Add minimum delay between inspections to avoid API rate limits:

```python
if "last_inspection_time" in st.session_state:
    time_since_last = time.time() - st.session_state.last_inspection_time
    if time_since_last < 5:  # 5 second minimum gap
        st.warning(f"Please wait {5 - time_since_last:.1f}s before next inspection")
        st.stop()
```

---

### 31. Analytics Query Could Be Expensive

**File:** `app/ui.py` (analytics_dashboard function)

**Issue:** Stats queries might do full table scans on large inspection databases.

**Recommendation:** Add pagination and limit:

```python
def get_inspection_stats(limit: int = 1000):
    """Get stats with limit."""
    session = SessionLocal()
    # Only query last N records
    recent = session.query(InspectionRecord).order_by(
        InspectionRecord.created_at.desc()
    ).limit(limit).all()
    # ... compute stats from recent records
```

---

### 32. PDF Report Generated Every Time Even If Unchanged

**File:** `src/reporting/pdf_generator.py`

**Recommendation:** Cache report path if results unchanged:

```python
def generate_report(inspection_data, final_state):
    # Check if report already exists
    report_key = hashlib.md5(
        json.dumps(inspection_data, sort_keys=True).encode()
    ).hexdigest()
    existing_path = REPORT_DIR / f"{report_key}.pdf"
    
    if existing_path.exists():
        logger.info(f"Report already generated: {existing_path}")
        return existing_path
    
    # Generate new report
    ...
```

---

### 33. Missing `.gitignore` Entries

**Recommendation:** Ensure these are in `.gitignore`:

```
uploads/
reports/
logs/
*.db
*.db-journal
.env
.env.local
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

---

## Summary Statistics

| Severity | Count | Status |
|----------|-------|--------|
| 🔴 Critical | 9 | Must fix before production |
| 🟠 High | 7 | Fix soon |
| 🟡 Medium | 9 | Fix in next sprint |
| 🟢 Low | 8 | Nice to have |
| **Total** | **33** | - |

---

## Fix Priority Order

1. **Import errors** (Items 1-9): Will cause immediate runtime failures
2. **Database & config** (Items 10-11): Will cause startup issues
3. **Safety gates** (Items 12-15): Core business logic correctness
4. **Agent reliability** (Items 17-19): Production stability
5. **Features** (Items 20-27): Complete incomplete features
6. **Polish** (Items 28-33): Long-term maintenance
