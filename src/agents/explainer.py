"""
Explainer Agent using Groq for fast text generation.
Includes native Groq SDK fallback and counterfactual explanations.
"""

import json
from typing import Optional

from src.schemas.models import VLMAnalysisResult
from utils.config import config
from utils.logger import setup_logger
from utils.prompts import EXPLAINER_PROMPT


class ExplainerAgent:
    """
    Text-based LLM for generating human-readable explanations.
    Uses Groq for ultra-fast inference with fallback to native SDK.
    Includes counterfactual explanations and reasoning chains.
    """
    
    def __init__(self):
        self.model = config.explainer_model
        self.temperature = config.explainer_temperature
        self.max_tokens = config.explainer_max_tokens
        
        self.logger = setup_logger(
            "agent.explainer",
            level=config.log_level,
            component="EXPLAINER"
        )
        
        # Initialize with fallback
        self._init_client()
        
        self.logger.info(f"Initialized Explainer with model: {self.model}")
    
    def _init_client(self):
        """Initialize LLM client with fallback to native Groq SDK."""
        self.use_langchain = False
        self.use_native_groq = False
        
        # Try LangChain-Groq first (for chain compatibility)
        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage
            
            self.llm = ChatGroq(
                model=self.model,
                api_key=config.groq_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.use_langchain = True
            self.logger.info("Using LangChain-Groq for Explainer")
            return
        except ImportError:
            self.logger.warning("langchain-groq not installed, trying native Groq SDK")
        except Exception as e:
            self.logger.warning(f"LangChain-Groq init failed: {e}, trying native SDK")
        
        # Fallback to native Groq SDK
        try:
            from groq import Groq
            self.client = Groq(api_key=config.groq_api_key)
            self.use_native_groq = True
            self.logger.info("Using native Groq SDK for Explainer")
            return
        except ImportError:
            self.logger.error("Neither langchain-groq nor groq package installed!")
            raise ImportError(
                "Please install either 'langchain-groq' or 'groq' package: "
                "pip install groq"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with automatic fallback between implementations."""
        if self.use_langchain:
            from langchain_core.messages import HumanMessage
            message = HumanMessage(content=prompt)
            response = self.llm.invoke([message])
            return response.content
        elif self.use_native_groq:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        else:
            raise RuntimeError("No LLM client available")
    
    def _generate_counterfactual(
        self,
        defects: list,
        verdict: dict
    ) -> str:
        """
        Generate counterfactual explanations for the findings.
        "What would need to change for a different verdict?"
        """
        if not defects:
            return ""
        
        counterfactual_prompt = f"""Based on these defects: {json.dumps([{
            'type': d.get('type', 'unknown'),
            'safety_impact': d.get('safety_impact', 'MODERATE'),
            'location': d.get('location', 'unspecified')
        } for d in defects[:3]], indent=2)}

Current verdict: {verdict.get('verdict', 'UNKNOWN')}

Generate 2-3 brief counterfactual statements explaining what would need to change 
for a different safety verdict. Format as bullet points starting with "If...".

Example format:
• If the crack were less than 2mm, it would be classified as COSMETIC
• If the defect were not on a load-bearing surface, severity would be MODERATE

Keep each statement under 15 words. Be specific and actionable."""

        try:
            response = self._call_llm(counterfactual_prompt)
            return response.strip()
        except Exception as e:
            self.logger.warning(f"Counterfactual generation failed: {e}")
            return ""
    
    def _format_reasoning_chain(
        self,
        inspector_result: VLMAnalysisResult,
        auditor_result: VLMAnalysisResult
    ) -> str:
        """Format the reasoning chains from both agents."""
        chains = []
        
        # Inspector chain
        chains.append("**INSPECTOR ANALYSIS:**")
        chains.append(f"1. Object identified: {inspector_result.object_identified}")
        chains.append(f"2. Overall condition: {inspector_result.overall_condition}")
        chains.append(f"3. Defects found: {len(inspector_result.defects)}")
        if inspector_result.defects:
            for i, d in enumerate(inspector_result.defects[:3], 1):
                chains.append(f"   {i}. {d.type} at {d.location} → {d.safety_impact}")
        chains.append(f"4. Confidence: {inspector_result.overall_confidence}")
        if inspector_result.analysis_reasoning:
            chains.append(f"5. Reasoning: {inspector_result.analysis_reasoning[:200]}...")
        
        chains.append("")
        
        # Auditor chain
        chains.append("**AUDITOR VERIFICATION:**")
        chains.append(f"1. Object confirmed: {auditor_result.object_identified}")
        chains.append(f"2. Condition assessment: {auditor_result.overall_condition}")
        chains.append(f"3. Defects verified: {len(auditor_result.defects)}")
        if auditor_result.defects:
            for i, d in enumerate(auditor_result.defects[:3], 1):
                chains.append(f"   {i}. {d.type} → {d.safety_impact}")
        chains.append(f"4. Confidence: {auditor_result.overall_confidence}")
        
        return "\n".join(chains)

    def _detect_truncation(self, response: str) -> bool:
        """Detect if the LLM response appears truncated or incomplete."""
        if not response:
            return True
        text = response.strip()
        # Look for the final recommendation marker with the new format
        has_final_marker = "--- FINAL RECOMMENDATION ---" in text
        has_final_keyword = "FINAL RECOMMENDATION" in text.upper()
        
        if not has_final_marker and not has_final_keyword:
            return True
        
        # If it ends without terminal punctuation, likely cut off
        if not text.endswith((".", "!", "?", "\n")):
            return True
        
        return False
    
    def generate_explanation(
        self,
        inspector_result: VLMAnalysisResult,
        auditor_result: VLMAnalysisResult,
        consensus: dict,
        safety_verdict: dict
    ) -> str:
        """
        Generate human-readable explanation of findings.
        Includes reasoning chains and counterfactual analysis.
        
        Args:
            inspector_result: Inspector analysis
            auditor_result: Auditor analysis
            consensus: Consensus result
            safety_verdict: Final safety verdict
        
        Returns:
            Natural language explanation
        """
        self.logger.info("Generating explanation...")
        
        try:
            # Build structured findings for LLM
            findings = {
                "inspector": {
                    "object": inspector_result.object_identified,
                    "condition": inspector_result.overall_condition,
                    "defects": [
                        {
                            "type": d.type,
                            "location": d.location,
                            "safety_impact": d.safety_impact,
                            "reasoning": d.reasoning
                        }
                        for d in inspector_result.defects
                    ],
                    "confidence": inspector_result.overall_confidence
                },
                "auditor": {
                    "object": auditor_result.object_identified,
                    "condition": auditor_result.overall_condition,
                    "defects": [
                        {
                            "type": d.type,
                            "location": d.location,
                            "safety_impact": d.safety_impact
                        }
                        for d in auditor_result.defects
                    ],
                    "confidence": auditor_result.overall_confidence
                },
                "consensus": consensus,
                "verdict": safety_verdict
            }
            
            # Build prompt with datetime-safe serialization
            def serialize_safe(obj):
                """Convert to JSON-serializable format."""
                if hasattr(obj, 'isoformat'):
                    return obj.isoformat()
                if hasattr(obj, '__dict__'):
                    return str(obj)
                return str(obj)
            
            # Use prompt template from utils/prompts.py
            findings_str = json.dumps(findings, indent=2, default=serialize_safe)
            prompt = EXPLAINER_PROMPT.format(findings=findings_str)
            
            self.logger.debug("Calling LLM API...")
            
            # Generate main explanation
            explanation = self._call_llm(prompt)

            # Detect truncation and supplement if needed - try up to 3 continuations
            max_continuations = 3
            for attempt in range(max_continuations):
                if not self._detect_truncation(explanation):
                    break  # Complete, no need to continue
                    
                self.logger.warning(f"Explanation appears truncated (attempt {attempt + 1}/{max_continuations}); requesting continuation.")
                
                # Identify which sections are missing
                required_sections = [
                    "EXECUTIVE SUMMARY", "INSPECTION DETAILS", "DEFECT ANALYSIS",
                    "REASONING CHAINS", "COUNTERFACTUAL ANALYSIS", "FINAL RECOMMENDATION"
                ]
                missing = [s for s in required_sections if f"--- {s} ---" not in explanation]
                
                try:
                    continuation_prompt = (
                        f"Continue the inspection report. The following sections are MISSING and MUST be added:\n"
                        f"{', '.join(missing)}\n\n"
                        f"Use the EXACT marker format: --- SECTION NAME ---\n"
                        f"Do NOT repeat sections already present. Start from where the text ended.\n\n"
                        f"PREVIOUS TEXT (last 2000 chars):\n"
                        f"{explanation[-2000:]}"
                    )
                    continuation = self._call_llm(continuation_prompt)
                    if continuation:
                        explanation = explanation.rstrip() + "\n\n" + continuation.strip()
                except Exception as cont_err:
                    self.logger.warning(f"Continuation request {attempt + 1} failed: {cont_err}")
                    break

                # If still truncated, add a minimal final block using the new marker format
                if self._detect_truncation(explanation):
                    self.logger.warning("Continuation still appears incomplete; adding fallback completion block.")
                    verdict_str = safety_verdict.get("verdict", "UNKNOWN")
                    action_required = (
                        "No action required" if verdict_str == "SAFE"
                        else "Further inspection or remediation recommended"
                    )
                    safety_line = (
                        "Safety Assessment: Component appears safe based on current findings."
                        if verdict_str == "SAFE"
                        else "Safety Assessment: Risks remain; address detected issues."
                    )
                    # Ensure the response ends cleanly with a complete final recommendation block
                    explanation = explanation.rstrip()
                    if not explanation.endswith(("\n", "\r")):
                        explanation += "\n"
                    explanation += (
                        "\n--- FINAL RECOMMENDATION ---\n\n"
                        f"Verdict: {verdict_str}\n"
                        f"Action Required: {action_required}\n"
                        f"{safety_line}\n"
                    )
            
            # Validate explanation contains required sections
            explanation_lower = explanation.lower()
            has_summary = any(kw in explanation_lower for kw in ["executive summary", "summary", "overview"])
            has_recommendation = any(kw in explanation_lower for kw in ["final recommendation", "recommendation", "verdict", "action required"])
            
            if not has_summary or not has_recommendation:
                self.logger.warning(f"Generated explanation missing sections: summary={has_summary}, recommendation={has_recommendation}")
                # Explanation will be validated and fixed in nodes.py
            
            self.logger.info("Explanation generated successfully")
            
            return explanation.strip()
        
        except Exception as e:
            self.logger.error(f"Explanation generation failed: {e}")
            
            return (
                f"Inspection complete. The system detected "
                f"{len(inspector_result.defects)} defects. "
                f"Final verdict: {safety_verdict.get('verdict', 'UNKNOWN')}. "
                f"Please review the detailed findings in the report."
            )

    def generate_decision_support(
        self,
        defects: list,
        verdict: str
    ) -> dict:
        """
        Generate decision support metrics (Cost, Time, Recommendation).
        Returns a dict with cost estimates in USD.
        """
        if not defects:
            return {
                "repair_cost": "$0",
                "replace_cost": "N/A", 
                "repair_time": "N/A",
                "replace_time": "N/A",
                "recommendation": "No Action Required",
                "reasoning": "No defects detected."
            }
            
        prompt = f"""
        You are a repair cost estimator. Based on the following defects, estimate repair vs replace costs in US DOLLARS ($).
        
        DEFECTS:
        {json.dumps([{
            'type': d.get('type', 'unknown') if isinstance(d, dict) else getattr(d, 'type', 'unknown'),
            'severity': d.get('safety_impact', 'MODERATE') if isinstance(d, dict) else getattr(d, 'safety_impact', 'MODERATE'),
            'location': d.get('location', 'unspecified') if isinstance(d, dict) else getattr(d, 'location', 'unspecified')
        } for d in defects], indent=2)}
        
        VERDICT: {verdict}
        
        Provide output as a strictly valid JSON object with these keys:
        - repair_cost_min: number (USD)
        - repair_cost_max: number (USD)
        - replace_cost_estimate: number (USD)
        - repair_time_estimate: string (e.g. "2-4 hours")
        - replace_lead_time: string (e.g. "3-5 days")
        - recommendation: string ("REPAIR" or "REPLACE")
        - reasoning: string (brief reason)
        
        Make realistic estimates assuming standard electronics/components.
        Use realistic USD market rates (e.g. simple soldering $20-50, screen replacement $100-300, laptop replacement $500-1500).
        Output ONLY the JSON.
        """
        
        try:
            response = self._call_llm(prompt)
            # Clean response to ensure json parsing works
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
                
            data = json.loads(response)
            
            # Format currency strings
            return {
                "repair_cost": f"${data.get('repair_cost_min', 0):,} - ${data.get('repair_cost_max', 0):,}",
                "replace_cost": f"${data.get('replace_cost_estimate', 0):,}",
                "repair_time": data.get("repair_time_estimate", "Unknown"),
                "replace_time": data.get("replace_lead_time", "Unknown"),
                "recommendation": data.get("recommendation", "Review"),
                "reasoning": data.get("reasoning", "")
            }
        except Exception as e:
            self.logger.error(f"Decision support generation failed: {e}")
            return {
                "repair_cost": "N/A",
                "replace_cost": "N/A",
                "repair_time": "N/A",
                "replace_time": "N/A",
                "recommendation": "Manual Review Required",
                "reasoning": "Could not generate estimates."
            }
    
    def health_check(self) -> bool:
        """Perform health check on Groq API."""
        try:
            self.logger.info(f"Health check: {self.model} (Groq)")
            
            response = self._call_llm("Respond with only the word 'OK'")
            
            if response and len(response) > 0:
                self.logger.info("✓ Explainer (Groq) is healthy")
                return True
            
            return False
        
        except Exception as e:
            self.logger.error(f"✗ Explainer health check failed: {e}")
            return False
