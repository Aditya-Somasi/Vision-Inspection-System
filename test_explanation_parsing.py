#!/usr/bin/env python3
"""
Test script to verify explanation parsing works correctly.
"""

from src.reporting.pdf_generator import parse_explanation_sections

# Sample explanation in the format that the explainer produces
sample_explanation = """The inspection of the object in question, identified as a hammer, has yielded a verdict of "SAFE" based on the comprehensive analysis conducted by both the inspector and the auditor. This conclusion is supported by a thorough examination of the object's condition, the absence of defects, and the high confidence levels expressed by both the inspector and the auditor in their assessments.

Inspector Analysis: The inspector's analysis revealed that the hammer is in good condition, with no visible defects or damage observed. Specifically, the handle and head of the hammer showed no signs of wear, cracks, or deformations, leading to a high confidence assessment in the overall condition of the object.

Auditor Verification: The auditor's verification process confirmed the inspector's findings, identifying the object as a hammer and assessing its condition as good. The auditor also verified the absence of defects and expressed high confidence in the assessment. The consistency between the inspector's and auditor's evaluations reinforces the reliability of the findings.

Safety Implications: The safety implications of these findings are significant. Given that the hammer is a general-purpose tool designed for manual use and does not pose a significant safety risk when used properly, the absence of defects and the good condition of the object suggest that it is safe for use. The high confidence levels in the assessments by both the inspector and the auditor further support this conclusion.

---

## REASONING CHAINS

INSPECTOR ANALYSIS:

1. Object identified: hammer
2. Overall condition: good
3. Defects found: 0
4. Confidence: high
5. Reasoning: The hammer appears to be in good condition with no visible defects or damage. The handle and head show no signs of wear, cracks, or deformations....

AUDITOR VERIFICATION:

1. Object confirmed: Hammer
2. Condition assessment: good
3. Defects verified: 0
4. Confidence: high

---

## COUNTERFACTUAL ANALYSIS

• If minor surface scratches were present, they would be classified as cosmetic defects
• If the handle showed small cracks, severity would be assessed as MODERATE
• If corrosion was detected on the head, it would require maintenance recommendations"""

print("Testing explanation parsing...")
print("=" * 80)

sections = parse_explanation_sections(sample_explanation)

print(f"\nParsed {len(sections)} sections:")
print("-" * 80)

for section_name, content in sections.items():
    print(f"\n[{section_name}]")
    print(f"Content length: {len(content)} characters")
    print(f"Preview: {content[:100]}...")
    print()

print("=" * 80)
print("\nTest completed successfully!")
print(f"Total sections: {len(sections)}")
print(f"Sections found: {list(sections.keys())}")

# Verify all expected sections are there
expected_sections = ["SUMMARY", "REASONING CHAINS", "COUNTERFACTUAL"]
found_sections = set(sections.keys())

for expected in expected_sections:
    if expected in found_sections:
        print(f"✓ {expected} found")
    else:
        print(f"✗ {expected} NOT found")
