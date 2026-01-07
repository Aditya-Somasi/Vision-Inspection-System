# 🔍 Vision Inspection System — Issues & Enhancements Report

**Generated:** January 7, 2026  
**Repository:** Vision Inspection System  
**Assessment Type:** Pre-Production Review  
**Status:** ⚠️ **Not Production Ready**

---

## 📋 Executive Summary

This report identifies **30+ issues** across the Vision Inspection System that must be addressed before production deployment. The most critical finding is that **both Inspector and Auditor agents use the same model**, completely defeating the consensus architecture designed to catch AI mistakes.

### Key Statistics

| Category | Count | Status |
|----------|-------|--------|
| 🔴 **Critical (Must Fix)** | 13 | Blocking |
| 🟠 **High Priority** | 12 | Important |
| 🟡 **Medium Priority** | 6 | Improvements |
| 🟢 **Low Priority** | 5 | Polish |

---

## 🎯 Recommended Model Configuration

### Current Setup (❌ BROKEN)
```
Inspector:  Qwen2.5-VL-7B (HuggingFace)  ──┐
                                           ├── SAME MODEL = NO REAL VERIFICATION
Auditor:    Qwen2.5-VL-7B (HuggingFace)  ──┘
Explainer:  Llama 3.3 70B (Groq)
```

### Recommended Setup (✅ PROPER CONSENSUS)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DUAL-MODEL CONSENSUS ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  INSPECTOR: Qwen2.5-VL-7B-Instruct                                  │   │
│  │  Provider:  HuggingFace Inference API                               │   │
│  │  Strengths: Fine-grained detection, consistent JSON output          │   │
│  │  Model ID:  Qwen/Qwen2.5-VL-7B-Instruct                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│                        ┌──────────┐                                         │
│                        │  IMAGE   │                                         │
│                        └──────────┘                                         │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  AUDITOR: Llama 4 Maverick 17B-128E-Instruct                        │   │
│  │  Provider:  Groq API (LPU-accelerated)                              │   │
│  │  Strengths: MoE architecture, native multimodal, ultra-fast         │   │
│  │  Model ID:  meta-llama/llama-4-maverick-17b-128e-instruct           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  EXPLAINER: Llama 3.3 70B Versatile                                 │   │
│  │  Provider:  Groq API                                                │   │
│  │  Strengths: Best-in-class text generation, coherent explanations    │   │
│  │  Model ID:  llama-3.3-70b-versatile                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Configuration Is Optimal

| Aspect | Benefit |
|--------|---------|
| **True Independence** | Qwen (Alibaba) vs Llama 4 (Meta) = different architectures, training data, biases |
| **Multi-Provider Resilience** | HuggingFace + Groq = if one fails, other still works |
| **Architecture Diversity** | Qwen: Dense transformer vs Llama 4: MoE (Mixture of Experts) |
| **Speed** | Groq LPUs provide ultra-fast inference for Auditor + Explainer |
| **Cost** | Both have free tiers; Groq is especially cost-effective |
| **Vision Quality** | Both are native multimodal (not bolted-on vision) |

### Alternative Configurations

| Option | Inspector | Auditor | Notes |
|--------|-----------|---------|-------|
| **A (Recommended)** | Qwen2.5-VL-7B (HF) | Llama 4 Maverick (Groq) | Best balance |

---

## 🔴 CRITICAL ISSUES (Production Blockers)

These issues MUST be fixed before the system can be considered safe for production use.

---

### 1. 🚨 Same Model Used for Both Agents — Defeats Consensus Architecture

> **This is the most dangerous issue in the entire system.**

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `.env.example`, `utils/config.py`, `src/agents/vlm_inspector.py`, `src/agents/vlm_auditor.py` |
| **Current State** | Both Inspector and Auditor use `Qwen/Qwen2.5-VL-7B-Instruct` |

#### Why This Is Dangerous

```
┌──────────────────────────────────────────────────────────────────┐
│  ❌ CURRENT: Same model = Same blind spots = Same mistakes      │
│                                                                  │
│    Image → Qwen2.5-VL → "No defects"                            │
│                ↓                                                 │
│    Image → Qwen2.5-VL → "No defects"   ← IDENTICAL!             │
│                ↓                                                 │
│    Agreement Score: 100% ← MEANINGLESS                          │
│                ↓                                                 │
│    Verdict: SAFE ← POTENTIALLY WRONG                            │
└──────────────────────────────────────────────────────────────────┘
```

- **No true independent verification** — both models share identical training biases
- **Both agents will make identical mistakes** — if one misses a crack, so will the other
- **"Agreement Score 100%" is meaningless** — it's the same model agreeing with itself
- **Defeats the entire purpose of consensus architecture** — designed to catch single-model failures

#### Required Fix

Update `.env` and config:
```env
# Inspector (HuggingFace)
VLM_INSPECTOR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLM_INSPECTOR_PROVIDER=huggingface

# Auditor (Groq - Different Model!)
VLM_AUDITOR_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
VLM_AUDITOR_PROVIDER=groq
```

---

### 2. 🚨 No Groq SDK Fallback — LangChain-Groq May Not Work Everywhere

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/agents/explainer.py`, `app/ui.py`, `requirements.txt` |
| **Current State** | Uses `langchain-groq` exclusively, no native Groq SDK fallback |

#### Why This Is A Problem

- `langchain-groq` has additional dependencies that may conflict
- Some systems/environments have issues with LangChain wrappers
- Native Groq SDK (`groq`) is more stable and universally compatible
- No fallback if `ChatGroq` fails to initialize

#### Required Fix

**1. Add `groq` to requirements:**
```
groq>=0.5.0  # Native Groq SDK - more compatible than langchain-groq
```

**2. Implement fallback pattern:**
```python
# In src/agents/explainer.py

def _init_llm(self):
    """Initialize LLM with fallback to native Groq SDK."""
    try:
        # Try LangChain-Groq first (for chain compatibility)
        from langchain_groq import ChatGroq
        self.llm = ChatGroq(
            model=config.explainer_model,
            api_key=config.groq_api_key,
            temperature=config.explainer_temperature,
        )
        self.use_langchain = True
        self.logger.info("Initialized Explainer with LangChain-Groq")
    except Exception as e:
        # Fallback to native Groq SDK
        self.logger.warning(f"LangChain-Groq failed ({e}), using native Groq SDK")
        from groq import Groq
        self.client = Groq(api_key=config.groq_api_key)
        self.use_langchain = False
        self.logger.info("Initialized Explainer with native Groq SDK")

def _call_llm(self, prompt: str) -> str:
    """Call LLM with automatic fallback."""
    if self.use_langchain:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    else:
        # Native Groq SDK
        response = self.client.chat.completions.create(
            model=config.explainer_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.explainer_temperature,
            max_tokens=config.explainer_max_tokens,
        )
        return response.choices[0].message.content
```

**3. Create new Groq-based VLM Auditor:**
```python
# In src/agents/vlm_auditor.py - Add Groq vision support

from groq import Groq

class GroqVisionAuditor(BaseVLMAgent):
    """Auditor using Groq's Llama 4 Maverick for vision."""
    
    def __init__(self):
        self.client = Groq(api_key=config.groq_api_key)
        self.model_id = config.vlm_auditor_model  # llama-4-maverick-17b-128e-instruct
        ...
    
    def verify(self, image_path: Path, context: InspectionContext, 
               inspector_result: VLMAnalysisResult) -> VLMAnalysisResult:
        # Encode image
        image_data = self._encode_image_to_base64(image_path)
        
        # Call Groq vision API
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        ...
```

---

### 3. Reports Only Show 1 Triggered Gate — Should Show All Gates

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/safety/gates.py`, `src/reporting/pdf_generator.py` |
| **Current State** | PDF shows only the gate that triggered final verdict |

#### Current Behavior (Problematic)

```
📄 Report-1: Triggered Gates: [GATE_DEFAULT_CONSERVATIVE]
📄 Report-2: Triggered Gates: [GATE_7_NO_DEFECTS]  
📄 Report-3: Triggered Gates: [GATE_1_AGENT_CRITICAL]
```

#### Expected Behavior (Transparent)

```
┌─────────────────────────────────────────────────────────────┐
│  SAFETY GATE EVALUATION                                     │
├─────────────────────────────────────────────────────────────┤
│  ✅ GATE_1: Critical Defect Check         → PASSED         │
│  ✅ GATE_2: Domain Zero Tolerance         → PASSED         │
│  ✅ GATE_3: Model Agreement Check         → PASSED (95%)   │
│  ✅ GATE_4: Confidence Threshold          → PASSED         │
│  ✅ GATE_5: Defect Count Limit            → PASSED (2/10)  │
│  ✅ GATE_6: High Criticality Check        → PASSED         │
│  ✅ GATE_7: No Defects Verification       → TRIGGERED ←    │
├─────────────────────────────────────────────────────────────┤
│  FINAL VERDICT: ✅ SAFE                                     │
└─────────────────────────────────────────────────────────────┘
```

#### Required Fix

Modify `SafetyGateEngine.evaluate()` to collect all gate results (pass/fail) in a list and return them alongside the verdict.

---

### 4. No Auditor Failure Handling — Uncertain Results Accepted

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/safety/gates.py`, `src/orchestration/nodes.py` |
| **Current State** | If auditor returns `"unknown"` or low confidence, verdict can still be SAFE |

#### Required Fix

```python
# Add new gate in safety evaluation:
if (auditor_result.overall_condition == "uncertain" or 
    auditor_conf == "low" or
    confidence_numeric < 0.40):
    triggered_gates.append("GATE_AUDITOR_UNCERTAIN")
    return SafetyVerdict(
        verdict="REQUIRES_HUMAN_REVIEW",
        reason="Auditor analysis inconclusive - human verification required",
        requires_human=True,
        confidence_level="low",
        ...
    )
```

---

### 5. Fragile JSON Response Parsing — Crashes on Bad Output

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/agents/base.py` (lines 112-144) |
| **Current State** | Uses naive `{` / `}` substring extraction |

**Current Code (Brittle):**
```python
start_idx = text.find("{")
end_idx = text.rfind("}") + 1
text = text[start_idx:end_idx]  # Fails on nested braces!
```

**Problems:**
- Fails on nested JSON structures
- Crashes on markdown with code examples
- No re-prompt mechanism on failure
- No fallback to "uncertain" status

**Required Fix:**
- Use regex for balanced brace matching
- Validate with Pydantic before accepting
- If validation fails, mark as "uncertain" and trigger human review

---

### 6. No Image Resize Before API Call — Payload Failures

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/agents/base.py` (lines 49-74) |
| **Current State** | Full image encoded as base64 without compression |

**Impact:** Large images (10MB+) will cause:
- HTTP 413 (Payload Too Large)
- HTTP 429 (Rate Limit)
- Timeouts
- Excessive API costs

**Required Fix:**
```python
def _encode_image_to_base64(self, image_path: Path, max_size: int = 1024) -> str:
    """Encode image with resize and compression."""
    from PIL import Image
    import io
    
    img = Image.open(image_path)
    
    # Resize to max dimension
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Compress to JPEG
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=85)
    
    # Enforce max payload size (10MB)
    if buffer.tell() > 10_000_000:
        # Try lower quality
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=60)
    
    base64_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{base64_str}"
```

---

### 7. No HF API Retry/Backoff — Single Failure Kills Workflow

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/agents/vlm_inspector.py`, `src/agents/vlm_auditor.py` |
| **Current State** | Direct API calls with no retry logic |

**Required Fix:** Create centralized client wrapper with:
- Exponential backoff (1s, 2s, 4s, 8s)
- Honor `Retry-After` header
- Max 3 retries
- Graceful degradation to "uncertain" status

---

### 8. Bounding Box Values Trusted Without Validation

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/schemas/models.py`, `utils/image_utils.py` |
| **Current State** | Only checks non-negative, not image bounds |

**Problem:** LLM can return `bbox: {x: 1500, y: 2000, width: 500}` for a 800x600 image.

**Required Fix:**
```python
def validate_bbox_against_image(bbox: BoundingBox, img_width: int, img_height: int) -> tuple[bool, str]:
    if bbox.x + bbox.width > img_width:
        return False, "bbox extends beyond image width"
    if bbox.y + bbox.height > img_height:
        return False, "bbox extends beyond image height"
    if bbox.width < 5 or bbox.height < 5:
        return False, "bbox too small to be meaningful"
    return True, "valid"
```

---

### 9. Safety Gate Names Don't Match Tests

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/safety/gates.py`, `tests/test_safety_gates.py` |
| **Current State** | All safety gate tests fail due to name mismatch |

| Test Expects | Code Uses |
|--------------|-----------|
| `GATE_1_CRITICAL_DEFECT` | `GATE_1_AGENT_CRITICAL` |
| `GATE_2_MODEL_DISAGREEMENT` | `GATE_3_MODEL_DISAGREEMENT` |
| `GATE_6_NO_DEFECTS` | `GATE_7_NO_DEFECTS` |

**Required Fix:** Extract gate names to module-level constants.

---

### 10. Test Expects SAFE for Cosmetic, Code Returns UNSAFE

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `tests/test_safety_gates.py`, `src/safety/gates.py` |
| **Current State** | "EMERGENCY FIX" conservative gate marks ALL defects as UNSAFE |

---

### 11. Invalid Dependency Blocks Installation

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `requirements.txt` |
| **Current State** | Lists `langchain-classic` which doesn't exist on PyPI |

**Fix:** Delete the line `langchain-classic`.

---

### 12. Human-in-Loop Feature Completely Disabled

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `src/orchestration/graph.py` (lines 37-48) |
| **Current State** | `should_run_human_review()` always returns `"generate_explanation"` |

---

### 13. No Fallback When Groq Explainer Unavailable

| Attribute | Details |
|-----------|---------|
| **Severity** | 🔴 **CRITICAL** |
| **Files** | `app/ui.py`, `src/agents/explainer.py` |
| **Current State** | System crashes if Groq API key missing or service down |

**Required Fix:** Add fallback to HuggingFace text model or native Groq SDK (see issue #2).

---

## 🟠 HIGH PRIORITY (Important Enhancements)

---

### 14. Missing: Visual Evidence 3-Panel Layout

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `app/ui.py`, `src/reporting/pdf_generator.py` |
| **Current State** | Only shows side-by-side (original + annotated) |

**Required Layout:**
```
┌───────────────────┬───────────────────┬─────────────────────┐
│   ORIGINAL        │   HEATMAP         │   ANNOTATED         │
│   [Raw image]     │  [Attention map]  │  [Boxes + labels]   │
└───────────────────┴───────────────────┴─────────────────────┘
```

---

### 15. Missing: Confidence Visualization with Progress Bars

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `app/ui.py`, `src/reporting/pdf_generator.py` |
| **Current State** | Shows only "High/Medium/Low" text labels |

**Required Enhancement:**
```
┌─────────────────────────────────────────────────────────────┐
│  CONFIDENCE METRICS                                         │
├─────────────────────────────────────────────────────────────┤
│  Inspector Confidence:  ████████████████████░░░░  87%       │
│  Auditor Confidence:    ██████████████████░░░░░░  76%       │
│  Agreement Score:       █████████████████████░░░  92%       │
├─────────────────────────────────────────────────────────────┤
│  Overall:  🟢 HIGH CONFIDENCE                               │
└─────────────────────────────────────────────────────────────┘
```

**Color Coding:**
- 🟢 **Green (>80%):** High confidence — safe to proceed
- 🟡 **Yellow (50-80%):** Medium confidence — review recommended
- 🔴 **Red (<50%):** Low confidence — human review required

---

### 16. Missing: Green "All Clear" Indicator for Safe Images

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `app/ui.py`, `src/reporting/pdf_generator.py` |
| **Current State** | No visual distinction for "no problems found" |

**When image has NO defects:**
```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│              ✅ ✅ ✅  ALL CLEAR  ✅ ✅ ✅                  │
│                                                             │
│         No defects detected by either agent                 │
│                                                             │
│    Inspector: ✅ No issues    Auditor: ✅ Confirmed         │
│                                                             │
│              [Green border/background throughout]           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 17. Missing: Grad-CAM Heatmaps (Explainability)

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `utils/image_utils.py`, `src/reporting/pdf_generator.py` |
| **Purpose** | Show which image regions influenced the model's decision |

---

### 18. Missing: Counterfactual Explanations

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `src/agents/explainer.py`, `utils/prompts.py` |

**Example Output:**
```
Current Finding: Crack detected (8mm length, CRITICAL)

What would change the verdict?
• If crack were <3mm → Would be classified MODERATE
• If crack were <1mm → Would be classified COSMETIC
• If crack not on load-bearing surface → MODERATE
```

---

### 19. Missing: Reasoning Chains from Both Agents

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `src/reporting/pdf_generator.py` |
| **Current State** | Only final summary shown |

**Required: Show step-by-step reasoning:**
```
INSPECTOR REASONING CHAIN
1. Object identified as: Industrial fastener (bolt)
2. Scanning for structural defects...
3. Detected: Visible crack on thread area
4. Measured: Approximately 8mm in length
5. Safety assessment: CRITICAL

AUDITOR VERIFICATION CHAIN
1. Reviewing Inspector's claim: crack on threads
2. Independent analysis confirms: crack present
3. Verification: CONFIRMED
```

---

### 20. Missing: Real-Time Streaming Progress

| Attribute | Details |
|-----------|---------|
| **Severity** | 🟠 **HIGH** |
| **Files** | `app/ui.py`, `src/orchestration/graph.py` |
| **Current State** | UI freezes during analysis |

**Required:**
```
✅ Step 1/5: Image preprocessing          [Complete]
✅ Step 2/5: Inspector analysis           [Complete]
🔄 Step 3/5: Auditor verification         [In Progress...]
⏳ Step 4/5: Consensus analysis           [Waiting]
⏳ Step 5/5: Safety evaluation            [Waiting]

████████████████████░░░░░░░░░░  60%
```

---

### 21-25. Additional High Priority Items

| # | Issue | Files |
|---|-------|-------|
| 21 | Chat Q&A Can Contradict Recorded Verdict | `app/ui.py` |
| 22 | Streamlit Blocking Calls — No Batch Support | `app/ui.py` |
| 23 | Comparison Mode in UI (slider view) | `app/ui.py` |
| 24 | Color-coded confidence badges | `app/ui.py` |
| 25 | Defect-by-defect highlighting toggle | `app/ui.py` |

---

## 🟡 MEDIUM PRIORITY

| # | Issue | Files |
|---|-------|-------|
| 26 | Database Initializes on Import | `src/database/repository.py` |
| 27 | Orphaned Config File (`models.yaml`) | `config/models.yaml` |
| 28 | Hardcoded Windows Absolute Path | `src/reporting/pdf_generator.py` |
| 29 | Dead Code in UI (duplicate columns) | `app/ui.py` |
| 30 | Inconsistent ChatGroq Initialization | Multiple files |
| 31 | Overlay May Mislead for Low-Confidence BBox | `utils/image_utils.py` |

---

## 🟢 LOW PRIORITY

| # | Issue | Files |
|---|-------|-------|
| 32 | Lock File Both Committed and Gitignored | `.gitignore` |
| 33 | May Log Sensitive API Keys | `utils/logger.py` |
| 34 | Missing Test Coverage for JSON Parsing | `tests/` |
| 35 | README Performance Claims Unverified | `README.md` |
| 36 | LangChain Version Unpinned | `pyproject.toml` |

---

## 🎯 Implementation Priority Matrix

```
                        IMPACT
                 Low    Medium    High
              ┌────────┬────────┬────────┐
        Low   │ 32-36  │ 26-31  │ 14-20  │
              ├────────┼────────┼────────┤
EFFORT Medium │        │ 21-25  │  3-8   │
              ├────────┼────────┼────────┤
        High  │        │        │  1,2   │
              └────────┴────────┴────────┘
```

---

## 📋 Recommended Implementation Order

### Phase 1: Critical Fixes (Week 1) — ~12 hours

| # | Task | Effort | Priority |
|---|------|--------|----------|
| 1 | **Use different models** (Qwen + Llama 4 Maverick) | 2h | 🔴 |
| 2 | **Add native Groq SDK fallback** | 3h | 🔴 |
| 3 | Remove `langchain-classic` from requirements | 5m | 🔴 |
| 4 | Fix gate name constants + tests | 2h | 🔴 |
| 5 | Add image resize before API call | 2h | 🔴 |
| 6 | Add retry/backoff wrapper | 3h | 🔴 |

### Phase 2: Safety Enhancements (Week 2) — ~14 hours

| # | Task | Effort |
|---|------|--------|
| 7 | Improve JSON parsing robustness | 4h |
| 8 | Add auditor failure gate | 2h |
| 9 | Show all gates (pass/fail) in report | 3h |
| 10 | Add bbox validation | 2h |
| 11 | Re-enable human-in-loop | 3h |

### Phase 3: UX Enhancements (Week 3) — ~20 hours

| # | Task | Effort |
|---|------|--------|
| 12 | Add 3-panel visual evidence | 4h |
| 13 | Add confidence progress bars | 3h |
| 14 | Add green "All Clear" indicator | 2h |
| 15 | Add real-time streaming progress | 6h |
| 16 | Add reasoning chains to PDF | 4h |

### Phase 4: Advanced Features (Week 4+) — ~20 hours

| # | Task | Effort |
|---|------|--------|
| 17 | Counterfactual explanations | 6h |
| 18 | Batch processing support | 8h |
| 19 | Comparison slider mode | 4h |
| 20 | Chat verdict guard | 2h |

---

## ✅ Pre-Production Checklist

### Critical (Must Have)
- [ ] Different models for Inspector (Qwen) and Auditor (Llama 4 Maverick)
- [ ] Native Groq SDK fallback implemented
- [ ] All safety gate tests passing
- [ ] Image resize before API calls
- [ ] Retry logic on API failures
- [ ] BBox validation against image dimensions
- [ ] JSON parsing handles malformed responses

### High Priority (Should Have)
- [ ] All gates shown in PDF report (pass/fail)
- [ ] Confidence shown as percentages/progress bars
- [ ] Green indicator for safe images
- [ ] Human-in-loop re-enabled for critical cases
- [ ] 3-panel visual evidence layout

### Nice to Have
- [ ] Grad-CAM heatmaps
- [ ] Counterfactual explanations
- [ ] Real-time streaming progress
- [ ] Batch processing

---

## 📝 Configuration Reference

### Recommended `.env` Configuration:

```env
# =============================================================================
# API KEYS
# =============================================================================
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxx

# =============================================================================
# INSPECTOR (Primary Analysis - HuggingFace)
# =============================================================================
VLM_INSPECTOR_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLM_INSPECTOR_PROVIDER=huggingface
VLM_INSPECTOR_TEMPERATURE=0.1
VLM_INSPECTOR_MAX_TOKENS=2048

# =============================================================================
# AUDITOR (Verification - Groq with Llama 4 Maverick)
# =============================================================================
VLM_AUDITOR_MODEL=meta-llama/llama-4-maverick-17b-128e-instruct
VLM_AUDITOR_PROVIDER=groq
VLM_AUDITOR_TEMPERATURE=0.1
VLM_AUDITOR_MAX_TOKENS=2048

# =============================================================================
# EXPLAINER (Report Generation - Groq)
# =============================================================================
EXPLAINER_MODEL=llama-3.3-70b-versatile
EXPLAINER_TEMPERATURE=0.3
EXPLAINER_MAX_TOKENS=4096

# =============================================================================
# CHAT (Follow-up Questions - Groq)
# =============================================================================
CHAT_MODEL=llama-3.3-70b-versatile
```

---

*Report generated by automated code review. All issues verified as present in codebase.*
*Model recommendations based on current API availability as of January 2026.*
