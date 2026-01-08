"""
Chat Widget Component for Vision Inspection System.
Features:
- LangChain history-aware conversation
- Streaming word-by-word output
- Typing indicator animation
- Context from inspection results
"""

import streamlit as st
import time
from typing import Dict, Any, Optional, Generator
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.logger import setup_logger
from utils.config import config

logger = setup_logger(__name__, component="CHAT_WIDGET")


def render_typing_indicator():
    """Render animated typing indicator."""
    return st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    """, unsafe_allow_html=True)


def render_message(role: str, content: str, is_streaming: bool = False):
    """Render a chat message with proper styling."""
    if role == "user":
        st.markdown(f"""
            <div class="chat-message user">
                <strong>You:</strong><br/>
                {content}
            </div>
        """, unsafe_allow_html=True)
    else:
        icon = "🤖" if not is_streaming else "💭"
        st.markdown(f"""
            <div class="chat-message assistant">
                <strong>{icon} Assistant:</strong><br/>
                {content}
            </div>
        """, unsafe_allow_html=True)


def stream_response(response_text: str, delay: float = 0.02) -> Generator[str, None, None]:
    """
    Generator that yields response text word by word for streaming effect.
    
    Args:
        response_text: Full response to stream
        delay: Delay between words in seconds
    """
    words = response_text.split()
    accumulated = ""
    
    for i, word in enumerate(words):
        accumulated += word + " "
        yield accumulated.strip()
        time.sleep(delay)


def get_chat_chain(inspection_results: Dict[str, Any], session_id: str = None):
    """
    Create a LangChain conversation chain with inspection context.
    Uses RunnableWithMessageHistory for proper history management (LangChain best practice).
    Falls back to native Groq SDK if LangChain fails.
    
    Args:
        inspection_results: Inspection results to provide context
        session_id: Session ID for history management
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        from langchain_groq import ChatGroq
        from langchain_core.runnables.history import RunnableWithMessageHistory
        from src.chat_memory import get_memory_manager
        
        # Build comprehensive context from inspection results
        verdict = inspection_results.get("safety_verdict", {}).get("verdict", "UNKNOWN")
        consensus = inspection_results.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        explanation = inspection_results.get("explanation", "")
        decision_support = inspection_results.get("decision_support", {})
        inspector_result = inspection_results.get("inspector_result", {})
        auditor_result = inspection_results.get("auditor_result", {})
        
        # Build detailed context
        defects_summary = []
        for i, defect in enumerate(defects[:10], 1):  # Limit to 10 for context
            defects_summary.append(
                f"  {i}. Type: {defect.get('type', 'unknown')}, "
                f"Severity: {defect.get('safety_impact', 'UNKNOWN')}, "
                f"Location: {defect.get('location', 'unknown')}, "
                f"Confidence: {defect.get('confidence', 'unknown')}"
            )
        
        context = f"""
INSPECTION RESULTS CONTEXT:
- Safety Verdict: {verdict}
- Total Defects Found: {len(defects)}
- Critical Defects: {sum(1 for d in defects if d.get('safety_impact') == 'CRITICAL')}
- Moderate Defects: {sum(1 for d in defects if d.get('safety_impact') == 'MODERATE')}
- Cosmetic Defects: {sum(1 for d in defects if d.get('safety_impact') == 'COSMETIC')}
- Inspector Confidence: {inspector_result.get('overall_confidence', 'unknown')}
- Auditor Confidence: {auditor_result.get('overall_confidence', 'unknown')}
- Models Agreement: {consensus.get('models_agree', False)} (Score: {consensus.get('agreement_score', 0):.1%})

DEFECTS DETAILS:
{chr(10).join(defects_summary) if defects_summary else '  No defects found'}

DECISION SUPPORT:
- Recommendation: {decision_support.get('recommendation', 'N/A')}
- Repair Cost: {decision_support.get('repair_cost', 'N/A')}
- Replace Cost: {decision_support.get('replace_cost', 'N/A')}
- Repair Time: {decision_support.get('repair_time', 'N/A')}
- Replace Time: {decision_support.get('replace_time', 'N/A')}

DETAILED ANALYSIS:
{explanation[:1000] if explanation and len(explanation) > 100 else 'Analysis summary not available'}
"""
        
        # Use proper LangChain prompt template with MessagesPlaceholder
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant for a Vision Inspection System.
You help users understand inspection results, defects, safety implications, and recommendations.

{inspection_context}

GUIDELINES:
- Answer questions about the inspection results accurately and thoroughly
- Explain defects, their severity, and safety implications clearly
- Provide actionable safety recommendations based on the verdict
- Reference specific defects by number when relevant
- If asked about something not in the results, clearly state that information is not available
- NEVER change or override the safety verdict - it is determined by the inspection system
- Use the defect details, decision support, and analysis provided above to give comprehensive answers
- Be conversational but professional
- If the user asks follow-up questions, use the chat history to maintain context"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create LLM with structured output support
        llm = ChatGroq(
            model=config.explainer_model,
            temperature=0.3,
            api_key=config.groq_api_key,
            streaming=True
        )
        
        # Create chain
        chain = prompt.partial(inspection_context=context) | llm
        
        # Use RunnableWithMessageHistory for proper history management (LangChain best practice)
        if session_id and config.enable_chat_memory:
            try:
                memory_manager = get_memory_manager()
                history_store = memory_manager.get_history(session_id)
                
                # Wrap chain with message history
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    lambda session_id: history_store,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
                
                return ("langchain_history", (chain_with_history, session_id))
            except Exception as e:
                logger.warning(f"History management failed: {e}. Using basic chain.")
        
        # Fallback to basic chain without history management
        return ("langchain", chain)
        
    except ImportError as e:
        logger.warning(f"LangChain not available: {e}. Using Groq SDK fallback.")
        return ("groq_sdk", inspection_results)
    except Exception as e:
        logger.warning(f"LangChain initialization failed: {e}. Using Groq SDK fallback.")
        return ("groq_sdk", inspection_results)


def get_groq_response(user_input: str, inspection_results: Dict[str, Any], chat_history: list):
    """
    Native Groq SDK fallback for chat when LangChain is unavailable.
    Streams response chunks for real-time display.
    """
    try:
        from groq import Groq
        
        if not config.groq_api_key:
            yield "Error: Groq API key not configured. Please set GROQ_API_KEY in your environment variables."
            return
        
        try:
            client = Groq(api_key=config.groq_api_key)
        except Exception as e:
            yield f"Error initializing Groq client: {str(e)}"
            return
        
        # Build context from inspection results
        verdict = inspection_results.get("safety_verdict", {}).get("verdict", "UNKNOWN")
        consensus = inspection_results.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        explanation = inspection_results.get("explanation", "")
        decision_support = inspection_results.get("decision_support", {})
        
        # Check if we have actual inspection results or just placeholder
        has_results = verdict != "UNKNOWN" or len(defects) > 0 or explanation and explanation != "No inspection results available yet."
        
        if has_results:
            system_context = f"""You are an expert assistant for a Vision Inspection System.
You help users understand inspection results, defects, and safety recommendations.

INSPECTION RESULTS CONTEXT:
- Verdict: {verdict}
- Total Defects: {len(defects)}
- Defects: {[{'type': d.get('type'), 'severity': d.get('safety_impact'), 'location': d.get('location')} for d in defects[:5]]}
- Decision Support: {decision_support}
- Summary: {explanation[:500] if explanation else 'Not available'}

GUIDELINES:
- Answer questions about the inspection results
- Explain defects and their severity
- Provide safety recommendations
- Be concise but thorough
- NEVER change or override the safety verdict"""
        else:
            system_context = """You are an expert assistant for a Vision Inspection System.
You help users understand how the inspection system works, answer questions about the inspection process,
and provide general guidance about vision-based quality inspection.

GUIDELINES:
- Answer questions about the inspection system and its capabilities
- Explain how vision inspection works
- Provide general guidance about quality inspection
- Be helpful and informative"""

        # Build messages including history
        messages = [{"role": "system", "content": system_context}]
        
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        messages.append({"role": "user", "content": user_input})
        
        # Stream response
        stream = client.chat.completions.create(
            model=config.explainer_model,  # Use explainer_model, not vlm_explainer_model
            messages=messages,
            temperature=0.3,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Groq SDK chat failed: {e}")
        yield f"Sorry, I couldn't process your question. Error: {str(e)}"


def format_chat_history(messages: list) -> list:
    """Convert session messages to LangChain message format."""
    from langchain_core.messages import HumanMessage, AIMessage
    
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))
    return history


def chat_widget(results: Dict[str, Any]):
    """
    Render the interactive chat widget with streaming responses.
    Uses LangChain best practices: RunnableWithMessageHistory, proper prompt templates.
    
    Args:
        results: Inspection results to provide context for chat
    """
    st.subheader("💬 Ask Questions About This Inspection")
    
    # Initialize chat session ID if needed
    if "chat_session_id" not in st.session_state:
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())
    
    # Initialize chat messages if needed
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Chat container with scroll
    chat_container = st.container()
    
    with chat_container:
        # Display existing messages
        for message in st.session_state.chat_messages:
            render_message(message["role"], message["content"])
    
    # Input area
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Your question:",
                placeholder="Ask about defects, recommendations, costs...",
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button("Send", width='stretch')
    
    # Process user input
    if submit and user_input.strip():
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show typing indicator
        typing_placeholder = st.empty()
        with typing_placeholder:
            st.markdown("🤖 Assistant is thinking...")
        
        try:
            # Get chat chain with session ID for history management
            chain_result = get_chat_chain(results, st.session_state.chat_session_id)
            
            if chain_result:
                backend_type, chain_data = chain_result
                response_placeholder = st.empty()
                full_response = ""
                
                if backend_type == "langchain_history":
                    # Use LangChain with RunnableWithMessageHistory (best practice)
                    chain_with_history, session_id = chain_data
                    config = {"configurable": {"session_id": session_id}}
                    
                    try:
                        # Stream with history management
                        for chunk in chain_with_history.stream(
                            {"input": user_input},
                            config=config
                        ):
                            if hasattr(chunk, 'content'):
                                full_response += chunk.content
                                with response_placeholder:
                                    render_message("assistant", full_response, is_streaming=True)
                    except Exception as e:
                        logger.error(f"LangChain history streaming error: {e}")
                        # Fallback to basic LangChain
                        chain_result_basic = get_chat_chain(results)
                        if chain_result_basic and chain_result_basic[0] == "langchain":
                            for chunk in chain_result_basic[1].stream({"input": user_input, "chat_history": format_chat_history(st.session_state.chat_messages[:-1])}):
                                if hasattr(chunk, 'content'):
                                    full_response += chunk.content
                                    with response_placeholder:
                                        render_message("assistant", full_response, is_streaming=True)
                        else:
                            # Final fallback to Groq SDK
                            for chunk in get_groq_response(user_input, results, st.session_state.chat_messages[:-1]):
                                full_response += chunk
                                with response_placeholder:
                                    render_message("assistant", full_response, is_streaming=True)
                
                elif backend_type == "langchain":
                    # Use basic LangChain streaming
                    chat_history = format_chat_history(st.session_state.chat_messages[:-1])
                    
                    try:
                        for chunk in chain_data.stream({
                            "input": user_input,
                            "chat_history": chat_history
                        }):
                            if hasattr(chunk, 'content'):
                                full_response += chunk.content
                                with response_placeholder:
                                    render_message("assistant", full_response, is_streaming=True)
                    except Exception as e:
                        logger.error(f"LangChain streaming error: {e}")
                        # Fallback to Groq SDK
                        for chunk in get_groq_response(user_input, results, st.session_state.chat_messages[:-1]):
                            full_response += chunk
                            with response_placeholder:
                                render_message("assistant", full_response, is_streaming=True)
                
                elif backend_type == "groq_sdk":
                    # Use native Groq SDK fallback
                    for chunk in get_groq_response(
                        user_input, 
                        chain_data,  # This is inspection_results
                        st.session_state.chat_messages[:-1]
                    ):
                        full_response += chunk
                        with response_placeholder:
                            render_message("assistant", full_response, is_streaming=True)
                
                # Clear typing indicator
                typing_placeholder.empty()
                
                # Add response to history (only if we got a response)
                if full_response.strip():
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                else:
                    # If no response, add error message
                    error_msg = "Sorry, I didn't receive a response. Please try again."
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    
            else:
                # Fallback if chain creation failed
                typing_placeholder.empty()
                fallback_response = "I'm sorry, I couldn't process your question. Please check your API keys and try again."
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": fallback_response
                })
                
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
            typing_placeholder.empty()
            error_response = f"An error occurred: {str(e)}. Please check your configuration and try again."
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": error_response
            })
        
        # Rerun to update display
        st.rerun()
    
    # Clear chat button and quick action buttons
    col_clear, col1, col2, col3 = st.columns([1, 2, 2, 2])
    
    with col_clear:
        if st.button("🗑️ Clear Chat", width='stretch', help="Clear chat history"):
            clear_chat()
            st.rerun()
    
    st.markdown("**Quick Questions:**")
    with col1:
        if st.button("🔍 Explain defects", width='stretch'):
            st.session_state.quick_question = "Can you explain each defect found and its severity?"
            st.rerun()
    
    with col2:
        if st.button("💰 Cost breakdown", width='stretch'):
            st.session_state.quick_question = "What's the detailed cost breakdown for repair vs replacement?"
            st.rerun()
    
    with col3:
        if st.button("⚠️ Safety impact", width='stretch'):
            st.session_state.quick_question = "What are the safety implications of using this item as-is?"
            st.rerun()
    
    # Handle quick questions
    if "quick_question" in st.session_state and st.session_state.quick_question:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": st.session_state.quick_question
        })
        st.session_state.quick_question = None
        st.rerun()


def clear_chat():
    """Clear chat history."""
    st.session_state.chat_messages = []
