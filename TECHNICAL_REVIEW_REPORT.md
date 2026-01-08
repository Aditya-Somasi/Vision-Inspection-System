# Technical Review Report: Vision Inspection System
**Date**: 2026-01-06  
**Scope**: `src/` and `utils/` directories  
**Focus**: Defect highlighting, false positives, missing summaries, PDF quality, prompt optimization

---

## PHASE 1 — FILE-BY-FILE TECHNICAL REVIEW

### Summary Table

| File | Responsibility | Critical Issues | High Issues | Medium/Low Issues |
|------|---------------|----------------|-------------|-------------------|
| `utils/image_utils.py` | Image processing, bbox drawing, heatmaps | **3** | **4** | 2 |
| `utils/prompts.py` | Prompt templates for VLMs | **2** | **3** | 1 |
| `src/agents/vlm_inspector.py` | Primary inspection agent | **2** | **2** | 2 |
| `src/agents/vlm_auditor.py` | Verification agent | **1** | **2** | 1 |
| `src/agents/explainer.py` | Explanation generation | **2** | **1** | 1 |
| `src/agents/base.py` | Base VLM agent class | 0 | 1 | 0 |
| `src/safety/consensus.py` | Consensus analysis | 0 | 2 | 0 |
| `src/safety/gates.py` | Safety gate evaluation | 1 | 1 | 0 |
| `src/schemas/models.py` | Data models | 0 | 1 | 0 |
| `src/reporting/pdf_generator.py` | PDF report generation | **2** | **3** | 2 |
| `utils/config.py` | Configuration management | 0 | 0 | 0 |
| `src/orchestration/nodes.py` | Workflow nodes | 1 | 1 | 0 |

---

## DETAILED FILE ANALYSIS

### 1. `utils/image_utils.py` (563 lines)

**Responsibility**: Image loading, resizing, bounding box drawing, heatmap generation

#### 🔴 CRITICAL ISSUES

**C1.1: Ambiguous Coordinate System Detection (Lines 186-203)**
- **Problem**: Logic `is_percentage = all(v <= 100 for v in [raw_x, raw_y, raw_w, raw_h] if v > 0)` can incorrectly classify pixel coordinates as percentages
- **Impact**: If model returns bbox like `{x: 50, y: 50, width: 100, height: 100}` for a 1000x1000px image, it's treated as percentages, placing box at wrong location
- **Root Cause**: Assumes all values ≤100 are percentages, but pixel coordinates can legitimately be ≤100 for small boxes in large images
- **Severity**: 🔴 Critical - causes misplaced highlights

**C1.2: No Validation of Bounding Box Reasonableness (Lines 176-212)**
- **Problem**: After coordinate conversion, no check if bbox is reasonable (e.g., width/height >50% of image, x/y outside bounds before clamping)
- **Impact**: Extreme outlier bboxes (hallucinated by model) are still drawn, creating false positives
- **Root Cause**: Trusts model output blindly, no sanity checks
- **Severity**: 🔴 Critical - causes false positives

**C1.3: Heatmap Uses Fixed Scale Factor Logic (Lines 307-314)**
- **Problem**: Assumes model processes images at exactly 1024px max dimension, but actual resizing happens in `vlm_inspector.py` with `max_image_size` (default 2048 in config)
- **Impact**: If image is resized to 2048px but heatmap assumes 1024px, coordinates are scaled incorrectly
- **Root Cause**: Hardcoded `MODEL_MAX_SIZE = 1024` doesn't match actual model input size
- **Severity**: 🔴 Critical - systematic coordinate errors

#### 🟡 HIGH PRIORITY ISSUES

**H1.1: Minimum Size Enforcement May Hide Small Defects (Lines 204-206)**
- **Problem**: Forces `w = max(w, 20)` and `h = max(h, 20)`, expanding tiny defects to 20px minimum
- **Impact**: Small defects get oversized boxes, creating false impression of defect size
- **Fix**: Only enforce minimum for marker placement, not box size

**H1.2: No Confidence Threshold for Highlighting (Lines 322-409)**
- **Problem**: All defects are highlighted regardless of confidence level
- **Impact**: Low-confidence defects (hallucinations) are highlighted as if certain
- **Fix**: Skip highlighting if `confidence == "low"` or `confidence < threshold`

**H1.3: Widespread Defect Detection is Keyword-Based (Lines 337-359)**
- **Problem**: Relies on keywords like "entire surface" to detect widespread defects, applies Gaussian blob to entire image
- **Impact**: Misinterprets location descriptions, creates heatmaps covering entire image incorrectly
- **Fix**: Only apply full-image heat if `bbox is None` AND explicit widespread flag

**H1.4: No Bounding Box Sanity Check Before Drawing (Lines 176-212)**
- **Problem**: Doesn't verify that bbox coordinates make logical sense relative to defect location description
- **Impact**: Box can be in wrong quadrant if model hallucinates or misplaces coordinates
- **Fix**: Validate bbox against location string (e.g., "top-left" → bbox should be in top-left quadrant)

#### 🟢 MEDIUM/LOW ISSUES

**M1.1: Marker Position Logic for Small Boxes (Lines 249-256)**
- **Problem**: Centers marker for boxes <50px, which may hide the actual defect location
- **Impact**: Minor - visual clarity issue
- **Fix**: Use offset marker even for small boxes

**M1.2: Heatmap Intensity Based Only on Severity (Lines 328-334)**
- **Problem**: Doesn't consider confidence level when setting heatmap intensity
- **Impact**: Low-confidence defects appear with same heat as high-confidence ones
- **Fix**: Multiply intensity by confidence factor

---

### 2. `utils/prompts.py` (315 lines)

**Responsibility**: Prompt templates for Inspector, Auditor, Explainer agents

#### 🔴 CRITICAL ISSUES

**C2.1: Inspector Prompt Too Verbose for Free-Tier VLMs (Lines 18-94)**
- **Problem**: 94-line prompt with detailed instructions, examples, multiple tasks competing for token budget
- **Impact**: Free-tier models may:
  - Truncate output (no JSON)
  - Skip less-critical sections (analysis_reasoning, summary)
  - Return incomplete defect lists
- **Root Cause**: Single-shot prompting tries to do everything (object ID, defect detection, bbox, reasoning, summary)
- **Severity**: 🔴 Critical - directly causes missing/incomplete summaries

**C2.2: Percentage Coordinate Instructions May Be Ignored (Lines 31-38, 60-65)**
- **Problem**: Instructions about percentage-based coordinates are buried in middle of long prompt
- **Impact**: Models may return pixel coordinates instead, causing coordinate mismatch
- **Root Cause**: Critical instruction not emphasized early, easily missed by model
- **Severity**: 🔴 Critical - contributes to misplaced highlights

#### 🟡 HIGH PRIORITY ISSUES

**H2.1: Auditor Prompt Contradicts Inspector Instructions (Lines 100-172)**
- **Problem**: Auditor is told "Do NOT search for problems" but also "Be especially vigilant if Inspector found nothing"
- **Impact**: Confusing instructions lead to inconsistent behavior, false positives when auditor "over-corrects"
- **Fix**: Clarify: "If Inspector found nothing, perform thorough check. Otherwise, verify existing findings."

**H2.2: Explainer Prompt Doesn't Guarantee Summary Section (Lines 178-225)**
- **Problem**: Prompt says "REQUIRED SECTIONS" but doesn't enforce them, allows model to skip sections if token-limited
- **Impact**: Missing SUMMARY section in PDF because explainer truncated output
- **Fix**: Add: "You MUST include ALL required sections. If output is truncated, prioritize SUMMARY and FINAL RECOMMENDATION."

**H2.3: No Token Budget Guidance in Prompts**
- **Problem**: Prompts don't specify target output length or warn about token limits
- **Impact**: Models may generate verbose responses, hitting limits before completing all required sections
- **Fix**: Add: "Keep responses concise. Target: 200-300 tokens for summary sections."

#### 🟢 MEDIUM/LOW ISSUES

**M2.1: Defect Type Examples Are Too Specific (Line 29)**
- **Problem**: Lists specific defect types, models may only look for those types
- **Impact**: Misses novel defect types
- **Fix**: Use broader categories: "structural (cracks, fractures), surface (scratches, dents), material (corrosion, wear)"

---

### 3. `src/agents/vlm_inspector.py` (427 lines)

**Responsibility**: Primary inspection using HuggingFace Qwen2.5-VL

#### 🔴 CRITICAL ISSUES

**C3.1: JSON Parsing Failure Doesn't Retry with Different Strategy (Lines 142-234)**
- **Problem**: `_parse_json_robust()` tries multiple strategies but if all fail, raises ValueError and returns empty result
- **Impact**: Valid analysis is lost if JSON is malformed, leading to "no defects" false negative or missing summary
- **Root Cause**: No fallback to extract structured data from unstructured text
- **Severity**: 🔴 Critical - causes missing summaries and false negatives

**C3.2: No Validation of Bounding Box Coordinates in Parsed Result (Lines 236-300)**
- **Problem**: `_validate_and_fix_result()` checks bbox exists but doesn't validate coordinates are reasonable (0-100 range, non-negative, width/height >0)
- **Impact**: Invalid coordinates (negative, >100, zero-size) pass through, causing drawing errors
- **Root Cause**: Validation only checks presence, not validity
- **Severity**: 🔴 Critical - contributes to misplaced highlights

#### 🟡 HIGH PRIORITY ISSUES

**H3.1: Default Confidence Set to "medium" for Missing Fields (Line 268)**
- **Problem**: If model omits confidence, defaults to "medium" instead of "low" (conservative)
- **Impact**: False positives - uncertain defects treated as medium confidence
- **Fix**: Default to "low" for missing/invalid confidence

**H3.2: Image Resize Logic Doesn't Preserve Aspect Ratio Correctly for API (Lines 63-64)**
- **Problem**: `img.thumbnail((max_size, max_size))` resizes but model may expect different aspect ratio handling
- **Impact**: Coordinate scaling mismatch if model processes differently
- **Fix**: Document exact resize behavior, ensure consistency

#### 🟢 MEDIUM/LOW ISSUES

**M3.1: Retry Logic Doesn't Distinguish Transient vs Permanent Failures (Lines 90-140)**
- **Problem**: Retries on all exceptions, including permanent ones (invalid API key)
- **Impact**: Wastes time and rate limit quota
- **Fix**: Classify exceptions (rate limit → retry, auth error → fail fast)

**M3.2: Logging of Raw Response is Truncated (Line 349)**
- **Problem**: Only logs first 300 chars, making debugging incomplete responses difficult
- **Impact**: Hard to diagnose truncation issues
- **Fix**: Log full response in DEBUG mode, truncate only in INFO

---

### 4. `src/agents/vlm_auditor.py` (291 lines)

**Responsibility**: Independent verification using Groq/HuggingFace

#### 🔴 CRITICAL ISSUES

**C4.1: Auditor Doesn't Use Robust JSON Parser (Lines 210)**
- **Problem**: Uses `_parse_json_response()` from base class (simple), not `_parse_json_robust()` like Inspector
- **Impact**: More prone to JSON parsing failures, leading to missing results
- **Root Cause**: Inconsistent parsing strategies between agents
- **Severity**: 🔴 Critical - causes missing summaries

#### 🟡 HIGH PRIORITY ISSUES

**H4.1: Auditor Prompt Formatting Issue (Lines 186-193)**
- **Problem**: `_format_inspector_findings()` creates formatted string but `AUDITOR_PROMPT` doesn't include `{inspector_findings}` placeholder
- **Impact**: Inspector findings are formatted but never inserted into prompt
- **Root Cause**: Prompt template doesn't match formatting function
- **Fix**: Add `{inspector_findings}` to `AUDITOR_PROMPT` or remove formatting function

**H4.2: Fallback Model Selection is Hardcoded (Lines 79-81)**
- **Problem**: If Groq unavailable, falls back to specific HuggingFace model, may not be available
- **Impact**: Runtime failure if fallback model doesn't exist
- **Fix**: Use configurable fallback model from config

#### 🟢 MEDIUM/LOW ISSUES

**M4.1: Health Check Uses Different Model (Lines 265-269)**
- **Problem**: Health check uses text-only model "llama-3.3-70b-versatile" instead of vision model
- **Impact**: Health check may pass but vision API may be unavailable
- **Fix**: Use actual vision model for health check or skip if Groq vision unavailable

---

### 5. `src/agents/explainer.py` (386 lines)

**Responsibility**: Generate human-readable explanations using Groq text model

#### 🔴 CRITICAL ISSUES

**C5.1: Explanation Generation Has No Output Length Validation (Lines 167-283)**
- **Problem**: No check if explanation is truncated (missing required sections) before returning
- **Impact**: Incomplete explanations reach PDF, causing missing summary sections
- **Root Cause**: Trusts model output without validation
- **Severity**: 🔴 Critical - directly causes missing PDF summaries

**C5.2: Prompt Doesn't Enforce Section Order or Completeness (Lines 232-252)**
- **Problem**: Prompt lists required sections but model may skip sections if hitting token limit
- **Impact**: SUMMARY section missing, PDF parser fails
- **Root Cause**: No explicit enforcement or validation
- **Severity**: 🔴 Critical - causes missing summaries in PDF

#### 🟡 HIGH PRIORITY ISSUES

**H5.1: Counterfactual Generation Can Fail Silently (Lines 97-131)**
- **Problem**: If counterfactual generation fails, returns empty string, no indication to user
- **Impact**: Explanation appears complete but missing counterfactual section
- **Fix**: Log warning, include placeholder text if generation fails

#### 🟢 MEDIUM/LOW ISSUES

**M5.1: Reasoning Chain Formatting Uses Markdown (Lines 133-165)**
- **Problem**: Uses `**` markdown which PDF parser strips out (line 111 in pdf_generator.py)
- **Impact**: Formatting lost in PDF
- **Fix**: Use plain text formatting or ensure PDF parser preserves markdown

---

### 6. `src/reporting/pdf_generator.py` (1235 lines)

**Responsibility**: Generate professional PDF reports

#### 🔴 CRITICAL ISSUES

**C6.1: Explanation Parsing Relies on Fragile Regex (Lines 53-205)**
- **Problem**: `parse_explanation_sections()` uses regex to find sections, easily breaks if model uses different formatting
- **Impact**: If model doesn't use exact `---\n## SECTION_NAME` format, sections aren't parsed, SUMMARY appears empty
- **Root Cause**: Over-reliance on specific markdown formatting
- **Severity**: 🔴 Critical - causes missing summaries in PDF

**C6.2: No Validation That Required Sections Exist (Lines 675-848)**
- **Problem**: `_build_executive_summary()` doesn't check if "SUMMARY" section exists before displaying
- **Impact**: If parsing fails, displays empty or placeholder text, user doesn't know summary is missing
- **Root Cause**: Assumes parsing always succeeds
- **Severity**: 🔴 Critical - causes missing summaries in PDF

#### 🟡 HIGH PRIORITY ISSUES

**H6.1: Defect Details Table Can Truncate Long Text (Lines 968-1040)**
- **Problem**: Uses `Paragraph` for long text but table cell height isn't adjusted, text may be cut off
- **Impact**: Long reasoning/recommendation text is truncated in PDF
- **Fix**: Use `KeepTogether` flowable or allow cells to expand vertically

**H6.2: Image Path Validation Doesn't Check Existence Early (Line 856)**
- **Problem**: Only checks `image_path.exists()` after starting PDF generation
- **Impact**: PDF generation fails midway, partial PDF saved
- **Fix**: Validate all image paths before starting PDF build

**H6.3: Heatmap and Annotated Images Created Every Time (Lines 869-892)**
- **Problem**: Regenerates heatmap/annotated images even if they exist, wasting compute
- **Impact**: Slower PDF generation, potential race conditions
- **Fix**: Check if files exist, reuse if recent (<5 minutes old)

#### 🟢 MEDIUM/LOW ISSUES

**M6.1: Model Names Extracted by String Splitting (Lines 280-281)**
- **Problem**: Uses `.split("/")[-1]` which breaks if model ID format changes
- **Impact**: Display name may be incorrect
- **Fix**: Use proper model registry or config mapping

**M6.2: Page Count Calculated After All Content Added (Line 288)**
- **Problem**: `num_pages = len(self._saved_page_states)` calculated before pages are rendered
- **Impact**: Page count in footer may be inaccurate
- **Fix**: Calculate after rendering or use `canvas.getPageNumber()`

---

### 7. `src/safety/consensus.py` (151 lines)

**Responsibility**: Analyze agreement between Inspector and Auditor

#### 🟡 HIGH PRIORITY ISSUES

**H7.1: Consensus Calculation Doesn't Consider Bounding Box Overlap (Lines 44-92)**
- **Problem**: Agreement based only on defect type and count, not spatial location
- **Impact**: Two models finding same defect type in different locations counted as agreement
- **Fix**: Add spatial agreement metric (IoU of bounding boxes) to consensus score

**H7.2: "No Defects" Agreement Logic May Be Too Strict (Lines 54-70)**
- **Problem**: Requires BOTH models HIGH confidence for "no defects" agreement, but low confidence "no defects" may be valid (uncertainty)
- **Impact**: May incorrectly flag as disagreement when both models are uncertain but agree
- **Fix**: Distinguish "uncertain but agreeing" vs "confident but disagreeing"

#### 🟢 MEDIUM/LOW ISSUES

**M7.1: Agreement Score Weighting is Hardcoded (Lines 86-91)**
- **Problem**: Weights (0.4, 0.3, 0.2, 0.1) are fixed, may not be optimal for all scenarios
- **Fix**: Make weights configurable or learn from validation data

---

### 8. `src/safety/gates.py` (597 lines)

**Responsibility**: Deterministic safety gate evaluation

#### 🔴 CRITICAL ISSUES

**C8.1: Gate 7 Clean Verification Doesn't Check Bounding Box Validity (Lines 380-431)**
- **Problem**: Verifies "no defects" but doesn't validate that any reported defects have valid bboxes
- **Impact**: Defects with invalid bboxes (hallucinations) may pass clean verification
- **Root Cause**: Only checks defect count, not defect quality
- **Severity**: 🔴 Critical - contributes to false positives

#### 🟡 HIGH PRIORITY ISSUES

**H8.1: Default Conservative Gate May Be Too Aggressive (Lines 508-552)**
- **Problem**: High criticality + cosmetic defects → UNSAFE, but cosmetic defects shouldn't trigger UNSAFE
- **Impact**: False positives - safe components marked UNSAFE due to cosmetic issues
- **Fix**: Reclassify as "REQUIRES_HUMAN_REVIEW" instead of "UNSAFE" for cosmetic-only defects

---

### 9. `src/schemas/models.py` (249 lines)

**Responsibility**: Pydantic data models for validation

#### 🟡 HIGH PRIORITY ISSUES

**H9.1: BoundingBox Validation Doesn't Check Reasonableness (Lines 11-23)**
- **Problem**: Only validates non-negative, doesn't check if coordinates are within image bounds or if width/height are reasonable
- **Impact**: Invalid bboxes (x=200, width=500 for 100px image) pass validation
- **Fix**: Add bounds checking or reasonableness checks (width < 2x image width)

---

### 10. `src/orchestration/nodes.py` (Partial - explainer node)

**Responsibility**: Workflow node functions

#### 🔴 CRITICAL ISSUES

**C10.1: Explanation Node Doesn't Validate Output Completeness (Lines 567-607)**
- **Problem**: After generating explanation, doesn't check if all required sections are present
- **Impact**: Incomplete explanations stored in state, passed to PDF generator
- **Root Cause**: No validation step after explanation generation
- **Severity**: 🔴 Critical - causes missing summaries

#### 🟡 HIGH PRIORITY ISSUES

**H10.1: Error Handling Swallows Explanation Errors (Lines 606-608)**
- **Problem**: If explanation generation fails, creates generic fallback but doesn't log reason
- **Impact**: Silent failures, hard to diagnose
- **Fix**: Log full exception, include failure reason in fallback text

---

## PHASE 2 — ROOT CAUSE ANALYSIS

### 2.1 Defect Region Highlighting Accuracy

#### Root Causes Identified:

1. **Coordinate System Ambiguity**
   - **Location**: `utils/image_utils.py:186-203`
   - **Issue**: Logic assumes all values ≤100 are percentages, but pixel coordinates can also be ≤100
   - **Example**: Bbox `{x: 50, y: 50, width: 80, height: 80}` on 1000x1000px image:
     - If model intended pixels → should be at (50, 50)
     - If treated as percentages → drawn at (500, 500) - **WRONG**
   - **Impact**: Systematic misplacement of highlights

2. **No Coordinate Validation After Conversion**
   - **Location**: `utils/image_utils.py:176-212`
   - **Issue**: After converting percentages to pixels, no sanity check if result is reasonable
   - **Example**: If conversion produces bbox covering 90% of image, should be flagged as suspicious
   - **Impact**: Extreme outliers (hallucinated bboxes) are drawn

3. **Scale Factor Mismatch**
   - **Location**: `utils/image_utils.py:307-314`, `src/agents/vlm_inspector.py:63-64`
   - **Issue**: Heatmap assumes 1024px max, but actual resize may use 2048px (config default)
   - **Impact**: Heatmap coordinates are systematically off by 2x for legacy pixel format

4. **Missing Confidence-Based Filtering**
   - **Location**: `utils/image_utils.py:322-409`
   - **Issue**: All defects are highlighted regardless of confidence
   - **Impact**: Low-confidence hallucinations appear as certain defects

5. **Prompt Instructions About Coordinates Are Buried**
   - **Location**: `utils/prompts.py:31-38, 60-65`
   - **Issue**: Critical percentage coordinate instructions are in middle of long prompt
   - **Impact**: Models may miss instruction, return pixel coordinates instead

#### Proposed Fixes (High-Level):

1. **Eliminate Coordinate Ambiguity**
   - Store coordinate system metadata (percentage vs pixel) with each bbox
   - Or: Always normalize to percentages at model output stage
   - Add validation: if bbox area > 50% of image, flag as suspicious

2. **Add Confidence Thresholds**
   - Don't highlight defects with `confidence == "low"` unless criticality is "high"
   - Or: Use semi-transparent overlay for low-confidence defects

3. **Validate Bounding Boxes**
   - Check: x, y, width, height all in valid range (0-100 for percentages)
   - Check: width + x ≤ 100, height + y ≤ 100 (doesn't exceed image)
   - Check: width > 0, height > 0, area > minimum (e.g., 0.1% of image)

4. **Fix Scale Factor**
   - Use actual resize dimension from config, not hardcoded 1024
   - Or: Always use percentage-based coordinates (eliminate pixel format)

5. **Improve Prompt**
   - Move coordinate instructions to top of prompt
   - Add explicit example: `{"x": 25, "y": 30, "width": 10, "height": 8}` means "25% from left, 30% from top, 10% width, 8% height"
   - Repeat instruction in JSON schema comment

---

### 2.2 False Positives on Clean Images

#### Root Causes Identified:

1. **Over-Aggressive Conservative Gate**
   - **Location**: `src/safety/gates.py:508-552`
   - **Issue**: High criticality + cosmetic defects → UNSAFE (should be REVIEW)
   - **Impact**: Clean images with minor cosmetic issues marked as UNSAFE

2. **Low Confidence Defaults to Medium**
   - **Location**: `src/agents/vlm_inspector.py:268`
   - **Issue**: Missing confidence field defaults to "medium" instead of "low"
   - **Impact**: Uncertain detections treated as medium confidence, contributing to false positives

3. **No Confidence Threshold for Highlighting**
   - **Location**: `utils/image_utils.py:322-409`
   - **Issue**: All defects highlighted regardless of confidence
   - **Impact**: Low-confidence hallucinations appear visually as real defects

4. **Auditor Prompt Confusion**
   - **Location**: `utils/prompts.py:108-151`
   - **Issue**: Contradictory instructions ("Don't search for problems" vs "Be vigilant if Inspector found nothing")
   - **Impact**: Auditor over-corrects, finds false defects to "disagree" with Inspector

5. **Clean Verification Doesn't Check Bbox Validity**
   - **Location**: `src/safety/gates.py:380-431`
   - **Issue**: Gate 7 only checks defect count, not if defects have valid bboxes
   - **Impact**: Defects with invalid coordinates (hallucinations) may pass verification

#### Proposed Fixes (High-Level):

1. **Fix Conservative Gate Logic**
   - Cosmetic-only defects → SAFE or REVIEW (not UNSAFE)
   - Only UNSAFE if CRITICAL or MODERATE defects found

2. **Default Missing Confidence to "low"**
   - Change default from "medium" to "low" in validation
   - Require explicit high/medium confidence from model

3. **Add Confidence-Based Filtering**
   - Skip highlighting if confidence < threshold (e.g., "low")
   - Or: Use visual distinction (dashed line for low confidence)

4. **Clarify Auditor Prompt**
   - Remove contradiction: "If Inspector found nothing, perform thorough independent check. Otherwise, verify existing findings."

5. **Validate Defect Quality in Gates**
   - Check bbox validity before counting defects
   - Filter out defects with invalid coordinates or confidence="low" (unless criticality="high")

---

### 2.3 Missing/Incomplete Analysis Summaries

#### Root Causes Identified:

1. **Prompt Too Verbose for Free-Tier Models**
   - **Location**: `utils/prompts.py:18-94` (Inspector), `utils/prompts.py:178-225` (Explainer)
   - **Issue**: Long prompts compete for token budget, models truncate output
   - **Impact**: `analysis_reasoning` field empty, summary sections missing

2. **No Output Validation**
   - **Location**: `src/agents/explainer.py:167-283`, `src/orchestration/nodes.py:567-607`
   - **Issue**: No check if explanation contains required sections before storing
   - **Impact**: Incomplete explanations stored, passed to PDF

3. **Fragile PDF Parsing**
   - **Location**: `src/reporting/pdf_generator.py:53-205`
   - **Issue**: Regex-based parsing breaks if model uses different formatting
   - **Impact**: Sections not parsed, SUMMARY appears empty even if present in raw text

4. **JSON Parsing Failures Don't Preserve Summary**
   - **Location**: `src/agents/vlm_inspector.py:142-234`
   - **Issue**: If JSON parsing fails, raises error, `analysis_reasoning` field lost
   - **Impact**: Valid summary text in response is discarded if JSON is malformed

5. **No Retry with Shorter Prompt**
   - **Location**: `src/agents/vlm_inspector.py:90-140`
   - **Issue**: Retries use same long prompt, doesn't try shorter version if truncation suspected
   - **Impact**: Repeated failures, no summary generated

#### Proposed Fixes (High-Level):

1. **Split Prompting Strategy**
   - **Stage 1**: Short prompt → Get defects + bboxes only (JSON, ~500 tokens)
   - **Stage 2**: If defects found, separate call → Generate summary (`analysis_reasoning`)
   - **Benefit**: Reduces token competition, ensures summary is generated

2. **Add Output Validation**
   - After explanation generation, check for required sections (SUMMARY, FINAL RECOMMENDATION)
   - If missing, retry with emphasis on missing sections
   - Or: Generate summary from structured data if model output incomplete

3. **Improve PDF Parsing Robustness**
   - Fallback to line-by-line text extraction if regex fails
   - Use keyword matching instead of exact formatting
   - Or: Use structured JSON output from explainer (separate from narrative)

4. **Preserve Summary on JSON Failure**
   - Extract `analysis_reasoning` text before JSON parsing
   - If JSON fails, still preserve summary text in result

5. **Add Token Budget Management**
   - Monitor response length, detect truncation (ends mid-sentence, no closing brace)
   - If truncated, retry with shorter prompt focusing on critical sections

---

## PHASE 3 — PROPOSED FIX STRATEGY

### 3.1 Localization Accuracy Fixes

#### Priority 1: Eliminate Coordinate Ambiguity
- **Action**: Always normalize coordinates to percentages at model output stage
- **Files**: `src/agents/vlm_inspector.py`, `src/agents/vlm_auditor.py`, `utils/image_utils.py`
- **Changes**:
  - Add `coordinate_system: Literal["percentage", "pixel"]` field to `BoundingBox` model
  - After parsing model output, convert all bboxes to percentages (if pixel) and set `coordinate_system="percentage"`
  - Remove ambiguous detection logic from `draw_bounding_boxes()` and `create_heatmap_overlay()`

#### Priority 2: Add Bounding Box Validation
- **Action**: Validate bboxes before drawing
- **Files**: `src/schemas/models.py`, `utils/image_utils.py`
- **Changes**:
  - Add `@field_validator` to `BoundingBox` checking: 0 ≤ x, y, width, height ≤ 100 (if percentage)
  - Add `@model_validator` checking: x + width ≤ 100, y + height ≤ 100
  - In `draw_bounding_boxes()`, skip defects with invalid bboxes, log warning

#### Priority 3: Confidence-Based Filtering
- **Action**: Don't highlight low-confidence defects
- **Files**: `utils/image_utils.py`
- **Changes**:
  - Add `confidence_threshold` parameter to `draw_bounding_boxes()` and `create_heatmap_overlay()`
  - Skip defects where `confidence == "low"` unless `criticality == "high"`
  - Or: Use visual distinction (dashed line, lower opacity) for low confidence

#### Priority 4: Fix Scale Factor
- **Action**: Use actual resize dimension from config
- **Files**: `utils/image_utils.py`, `src/agents/vlm_inspector.py`
- **Changes**:
  - Pass `actual_model_size` parameter to `create_heatmap_overlay()` from agent
  - Remove hardcoded `MODEL_MAX_SIZE = 1024`
  - Use config value: `config.max_image_dimension` (default 2048)

---

### 3.2 False Positive Reduction

#### Priority 1: Fix Conservative Gate Logic
- **Action**: Reclassify cosmetic-only defects
- **Files**: `src/safety/gates.py`
- **Changes**:
  - Modify default gate (lines 508-552): Cosmetic-only → "REQUIRES_HUMAN_REVIEW" (not UNSAFE)
  - Only UNSAFE if CRITICAL or MODERATE defects found

#### Priority 2: Default Confidence to "low"
- **Action**: Change default in validation
- **Files**: `src/agents/vlm_inspector.py`
- **Changes**:
  - Line 268: Change `defect.setdefault("confidence", "medium")` to `"low"`

#### Priority 3: Clarify Auditor Prompt
- **Action**: Remove contradiction
- **Files**: `utils/prompts.py`
- **Changes**:
  - Rewrite lines 108-151: "If Inspector found no defects, perform thorough independent check. Otherwise, verify Inspector's findings are accurate."

#### Priority 4: Validate Defect Quality in Gates
- **Action**: Filter invalid defects before gate evaluation
- **Files**: `src/safety/gates.py`
- **Changes**:
  - In `evaluate()`, filter `combined_defects` to remove:
    - Defects with invalid bboxes (x+width > 100, etc.)
    - Defects with confidence="low" (unless criticality="high")
  - Recalculate defect counts after filtering

---

### 3.3 Summary Generation Reliability

#### Priority 1: Split Prompting Strategy (Two-Stage)
- **Action**: Separate defect detection from summary generation
- **Files**: `src/agents/vlm_inspector.py`, `utils/prompts.py`
- **Changes**:
  - **Stage 1 Prompt**: Short, focused on defects + bboxes only (~30 lines, ~300 tokens)
  - **Stage 2 Prompt** (if defects found): "Based on these defects: {defects_json}, generate a 2-3 sentence summary explaining the findings."
  - Store `analysis_reasoning` from Stage 2, defects from Stage 1

#### Priority 2: Add Output Validation
- **Action**: Validate explanation completeness
- **Files**: `src/orchestration/nodes.py`, `src/agents/explainer.py`
- **Changes**:
  - After `generate_explanation()`, check if output contains "SUMMARY" or "Executive Summary" keywords
  - If missing, generate fallback summary from structured data: `f"Inspection of {object} found {len(defects)} defect(s). Verdict: {verdict}."`
  - Log warning if fallback used

#### Priority 3: Improve PDF Parsing Robustness
- **Action**: Use keyword matching instead of regex
- **Files**: `src/reporting/pdf_generator.py`
- **Changes**:
  - Replace regex parsing with keyword-based section detection:
    - Find "EXECUTIVE SUMMARY" or "SUMMARY" keyword, extract text until next section keyword
    - More forgiving of formatting variations
  - Add fallback: If no sections found, treat entire explanation as "SUMMARY"

#### Priority 4: Preserve Summary on JSON Failure
- **Action**: Extract summary text before JSON parsing
- **Files**: `src/agents/vlm_inspector.py`
- **Changes**:
  - In `_parse_json_robust()`, try to extract `analysis_reasoning` field value as text before full JSON parse
  - If JSON parse fails, still return result with extracted summary text

#### Priority 5: Add Token Budget Management
- **Action**: Detect and handle truncation
- **Files**: `src/agents/vlm_inspector.py`, `src/agents/explainer.py`
- **Changes**:
  - Check if response ends with `}` or `]` (complete JSON) or ends mid-sentence (truncated)
  - If truncated, retry with shorter prompt focusing on critical sections
  - Or: Request summary first, then details (prioritize summary)

---

### 3.4 Prompt Quality & Model Limitations

#### Priority 1: Shorten Inspector Prompt
- **Action**: Reduce verbosity, focus on critical instructions
- **Files**: `utils/prompts.py`
- **Changes**:
  - Cut prompt from 94 lines to ~40 lines
  - Move coordinate instructions to top (lines 31-38 → top)
  - Remove redundant examples, keep only one clear example
  - Remove "INFER CRITICALITY" task (not critical, uses tokens)

#### Priority 2: Make Coordinate Instructions Explicit
- **Action**: Emphasize percentage requirement
- **Files**: `utils/prompts.py`
- **Changes**:
  - Add at top: "CRITICAL: All bounding box coordinates MUST be PERCENTAGES (0-100), not pixels."
  - Repeat in JSON schema: `"bbox": {"x": 25, "y": 30, "width": 10, "height": 8}  // All values are PERCENTAGES (0-100)`

#### Priority 3: Add Token Budget Guidance
- **Action**: Warn models about token limits
- **Files**: `utils/prompts.py`
- **Changes**:
  - Add to Inspector prompt: "Keep responses concise. Target: 300 tokens for JSON, 100 tokens for analysis_reasoning."
  - Add to Explainer prompt: "You have ~1500 tokens available. Prioritize SUMMARY and FINAL RECOMMENDATION sections."

#### Priority 4: Enforce Section Completeness
- **Action**: Make sections mandatory in prompt
- **Files**: `utils/prompts.py`
- **Changes**:
  - Add to Explainer prompt: "You MUST include ALL required sections. If output is truncated, prioritize SUMMARY and FINAL RECOMMENDATION above other sections."

---

### 3.5 PDF Report Enhancement (Design Only)

#### Proposed Structure:

1. **Cover Page** (Optional)
   - Logo, Inspection ID, Date, Verdict Stamp

2. **Executive Summary** (Always First)
   - Key metrics table (defect count, criticality, verdict)
   - 2-3 sentence narrative summary (from `analysis_reasoning` or fallback)
   - **Enhancement**: If summary missing, generate from structured data: `"Inspection of {object} identified {defect_count} defect(s). {Verdict: SAFE/UNSAFE/REVIEW}."`

3. **Visual Evidence** (3-Panel Layout)
   - Original, Heatmap, Annotated (current implementation is good)
   - **Enhancement**: Add confidence indicators on annotated image (dashed lines for low confidence)

4. **Defect Details** (Per-Defect Pages)
   - Current table format is good
   - **Enhancement**: Add "Defect #1 Detail Page" with:
     - Larger image crop of defect region
     - Full reasoning text (no truncation)
     - Confidence visualization (bar chart)

5. **Safety Gates Dashboard**
   - Current table showing all gates is good
   - **Enhancement**: Add visual indicators (✓/✗ icons, color coding)

6. **Model Comparison**
   - Current table is adequate
   - **Enhancement**: Add agreement visualization (bar chart of agreement score)

7. **Audit Trail**
   - Current processing info is good
   - **Enhancement**: Add "Analysis Completeness" indicator:
     - "Summary: ✓ Present / ✗ Missing"
     - "Reasoning: ✓ Complete / ⚠ Truncated"

#### Design Principles:

1. **Graceful Degradation**: If summary missing, show structured fallback instead of empty section
2. **Visual Hierarchy**: Use color coding, icons, spacing to guide reader
3. **Completeness Indicators**: Always show if sections are missing/truncated
4. **Professional Appearance**: Consistent fonts, spacing, branding

---

## SUMMARY OF CRITICAL FIXES NEEDED

### Immediate Actions (Fix False Positives & Missing Summaries):

1. ✅ Fix coordinate system ambiguity (always use percentages)
2. ✅ Add bounding box validation (skip invalid bboxes)
3. ✅ Default confidence to "low" (not "medium")
4. ✅ Fix conservative gate (cosmetic → REVIEW, not UNSAFE)
5. ✅ Split prompting strategy (defects first, summary second)
6. ✅ Add output validation (check for required sections)
7. ✅ Improve PDF parsing (keyword-based, fallback to structured data)
8. ✅ Shorten prompts (reduce verbosity for free-tier models)

### High Priority (Improve Accuracy):

9. ✅ Add confidence-based filtering (don't highlight low-confidence defects)
10. ✅ Fix scale factor mismatch (use actual resize dimension)
11. ✅ Clarify auditor prompt (remove contradiction)
12. ✅ Preserve summary on JSON failure (extract before parsing)

### Medium Priority (Polish & Robustness):

13. ✅ Improve error handling (log truncation, preserve partial results)
14. ✅ Add token budget management (detect truncation, retry with shorter prompt)
15. ✅ Enhance PDF structure (completeness indicators, visual hierarchy)

---

**END OF REPORT**

**Next Steps**: Review this report, approve fixes, then proceed with implementation in order of priority.
