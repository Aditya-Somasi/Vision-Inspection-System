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


def get_chat_chain(inspection_results: Dict[str, Any]):
    """
    Create a LangChain conversation chain with inspection context.
    Uses history-aware retrieval for follow-up questions.
    Falls back to native Groq SDK if LangChain fails.
    """
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_groq import ChatGroq
        
        # Build context from inspection results
        verdict = inspection_results.get("safety_verdict", {}).get("verdict", "UNKNOWN")
        consensus = inspection_results.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        explanation = inspection_results.get("explanation", "")
        decision_support = inspection_results.get("decision_support", {})
        
        context = f"""
        INSPECTION RESULTS CONTEXT:
        - Verdict: {verdict}
        - Total Defects: {len(defects)}
        - Defects: {[{'type': d.get('type'), 'severity': d.get('safety_impact'), 'location': d.get('location')} for d in defects[:5]]}
        - Decision Support: {decision_support}
        - Summary: {explanation[:500] if explanation else 'Not available'}
        """
        
        # Create prompt with history
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert assistant for a Vision Inspection System.
            You help users understand inspection results, defects, and safety recommendations.
            
            {context}
            
            GUIDELINES:
            - Answer questions about the inspection results
            - Explain defects and their severity
            - Provide safety recommendations
            - Be concise but thorough
            - If asked about something not in the results, say so clearly
            - NEVER change or override the safety verdict
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Create LLM
        llm = ChatGroq(
            model=config.vlm_explainer_model,
            temperature=0.3,
            api_key=config.groq_api_key,
            streaming=True
        )
        
        chain = prompt | llm
        
        return ("langchain", chain)
        
    except Exception as e:
        logger.warning(f"LangChain initialization failed: {e}. Using Groq SDK fallback.")
        # Return indicator to use native Groq SDK
        return ("groq_sdk", inspection_results)


def get_groq_response(user_input: str, inspection_results: Dict[str, Any], chat_history: list):
    """
    Native Groq SDK fallback for chat when LangChain is unavailable.
    Streams response chunks for real-time display.
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=config.groq_api_key)
        
        # Build context from inspection results
        verdict = inspection_results.get("safety_verdict", {}).get("verdict", "UNKNOWN")
        consensus = inspection_results.get("consensus", {})
        defects = consensus.get("combined_defects", [])
        explanation = inspection_results.get("explanation", "")
        decision_support = inspection_results.get("decision_support", {})
        
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
            model=config.vlm_explainer_model,
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
    
    Args:
        results: Inspection results to provide context for chat
    """
    st.subheader("💬 Ask Questions About This Inspection")
    
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
            submit = st.form_submit_button("Send", use_container_width=True)
    
    # Process user input
    if submit and user_input.strip():
        # Add user message
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Show typing indicator
        with st.spinner(""):
            typing_placeholder = st.empty()
            with typing_placeholder:
                render_typing_indicator()
            
            try:
                # Get chat chain (returns tuple: (backend_type, chain_or_results))
                chain_result = get_chat_chain(results)
                
                if chain_result:
                    backend_type, chain_data = chain_result
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    if backend_type == "langchain":
                        # Use LangChain streaming
                        chat_history = format_chat_history(st.session_state.chat_messages[:-1])
                        
                        for chunk in chain_data.stream({
                            "input": user_input,
                            "chat_history": chat_history
                        }):
                            if hasattr(chunk, 'content'):
                                full_response += chunk.content
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
                    
                    # Add response to history
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                else:
                    # Fallback if chain creation failed
                    typing_placeholder.empty()
                    fallback_response = "I'm sorry, I couldn't process your question. Please try again."
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": fallback_response
                    })
                    
            except Exception as e:
                logger.error(f"Chat error: {e}")
                typing_placeholder.empty()
                error_response = f"An error occurred: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_response
                })
        
        # Rerun to update display
        st.rerun()
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Explain defects", use_container_width=True):
            st.session_state.quick_question = "Can you explain each defect found and its severity?"
            st.rerun()
    
    with col2:
        if st.button("💰 Cost breakdown", use_container_width=True):
            st.session_state.quick_question = "What's the detailed cost breakdown for repair vs replacement?"
            st.rerun()
    
    with col3:
        if st.button("⚠️ Safety impact", use_container_width=True):
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
