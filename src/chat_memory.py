"""
Chat memory and history-aware conversation management.
Implements SQLite-backed chat history with context-aware query rewriting.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from logger import setup_logger
from config import config
from prompts import CHAT_SYSTEM_PROMPT, CHAT_HISTORY_AWARE_PROMPT

logger = setup_logger(__name__, level=config.log_level, component="CHAT_MEMORY")


# ============================================================================
# SQLITE CHAT HISTORY
# ============================================================================

class SQLiteChatHistory(BaseChatMessageHistory):
    """
    Chat message history backed by SQLite.
    Implements LangChain's BaseChatMessageHistory interface.
    """
    
    def __init__(self, session_id: str, db_path: Optional[str] = None):
        """
        Initialize SQLite chat history.
        
        Args:
            session_id: Unique session identifier
            db_path: Path to SQLite database file
        """
        self.session_id = session_id
        self.db_path = db_path or config.chat_history_db
        self._init_db()
        
        logger.debug(f"Initialized chat history for session: {session_id}")
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP,
                inspection_id TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_messages 
            ON chat_messages(session_id, timestamp)
        """)
        
        # Create session if doesn't exist
        cursor.execute("""
            INSERT OR IGNORE INTO chat_sessions (session_id) 
            VALUES (?)
        """, (self.session_id,))
        
        conn.commit()
        conn.close()
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve all messages for this session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message_type, content, metadata 
            FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (self.session_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for msg_type, content, metadata_json in rows:
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            if msg_type == "human":
                messages.append(HumanMessage(content=content, additional_kwargs=metadata))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content, additional_kwargs=metadata))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content, additional_kwargs=metadata))
        
        return messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine message type
        if isinstance(message, HumanMessage):
            msg_type = "human"
        elif isinstance(message, AIMessage):
            msg_type = "ai"
        elif isinstance(message, SystemMessage):
            msg_type = "system"
        else:
            msg_type = "unknown"
        
        # Serialize metadata
        metadata_json = json.dumps(message.additional_kwargs) if message.additional_kwargs else None
        
        cursor.execute("""
            INSERT INTO chat_messages (session_id, message_type, content, metadata)
            VALUES (?, ?, ?, ?)
        """, (self.session_id, msg_type, message.content, metadata_json))
        
        # Update session last_active
        cursor.execute("""
            UPDATE chat_sessions 
            SET last_active = CURRENT_TIMESTAMP 
            WHERE session_id = ?
        """, (self.session_id,))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Added {msg_type} message to session {self.session_id}")
    
    def clear(self) -> None:
        """Clear all messages for this session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM chat_messages 
            WHERE session_id = ?
        """, (self.session_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared chat history for session {self.session_id}")
    
    def get_message_count(self) -> int:
        """Get total message count for this session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM chat_messages 
            WHERE session_id = ?
        """, (self.session_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def trim_messages(self, max_messages: int = None):
        """
        Trim old messages to stay within limit.
        
        Args:
            max_messages: Maximum messages to keep (defaults to config)
        """
        max_messages = max_messages or config.max_chat_history
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Keep only the latest N messages
        cursor.execute("""
            DELETE FROM chat_messages 
            WHERE session_id = ? 
            AND id NOT IN (
                SELECT id FROM chat_messages 
                WHERE session_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            )
        """, (self.session_id, self.session_id, max_messages))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Trimmed {deleted} old messages from session {self.session_id}")


# ============================================================================
# CHAT MEMORY MANAGER
# ============================================================================

class ChatMemoryManager:
    """
    Manages chat sessions and provides history-aware conversation.
    """
    
    def __init__(self):
        self.logger = logger
        self.db_path = config.chat_history_db
    
    def get_history(self, session_id: str) -> SQLiteChatHistory:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            SQLiteChatHistory instance
        """
        return SQLiteChatHistory(session_id, self.db_path)
    
    def create_session(
        self,
        session_id: str,
        inspection_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create a new chat session.
        
        Args:
            session_id: Unique session identifier
            inspection_id: Associated inspection ID
            metadata: Additional session metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO chat_sessions 
            (session_id, inspection_id, metadata, created_at, last_active)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (session_id, inspection_id, metadata_json))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Created chat session: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT created_at, last_active, inspection_id, metadata
            FROM chat_sessions
            WHERE session_id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        created_at, last_active, inspection_id, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        return {
            "session_id": session_id,
            "created_at": created_at,
            "last_active": last_active,
            "inspection_id": inspection_id,
            "metadata": metadata
        }
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent chat sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, created_at, last_active, inspection_id
            FROM chat_sessions
            ORDER BY last_active DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": row[0],
                "created_at": row[1],
                "last_active": row[2],
                "inspection_id": row[3]
            }
            for row in rows
        ]
    
    def delete_session(self, session_id: str):
        """Delete a chat session and all its messages."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Deleted chat session: {session_id}")


# ============================================================================
# HISTORY-AWARE QUERY REWRITING
# ============================================================================

def rewrite_query_with_history(
    query: str,
    chat_history: List[BaseMessage],
    llm = None
) -> str:
    """
    Rewrite user query with chat history context.
    Makes queries standalone for better retrieval/understanding.
    
    Args:
        query: User's current question
        chat_history: Previous conversation messages
        llm: Language model for rewriting (optional)
    
    Returns:
        Rewritten standalone query
    """
    if not chat_history or len(chat_history) == 0:
        # No history, return original query
        return query
    
    # If no LLM provided, do simple contextual rewriting
    if llm is None:
        # Simple heuristic-based rewriting
        last_messages = chat_history[-4:]  # Last 2 exchanges
        
        # Check if query is a follow-up (starts with "it", "this", "that", etc.)
        followup_indicators = ["it", "this", "that", "they", "those", "where", "how"]
        
        first_word = query.lower().split()[0] if query.split() else ""
        
        if first_word in followup_indicators:
            # Extract context from last AI message
            for msg in reversed(last_messages):
                if isinstance(msg, AIMessage):
                    # Simple context addition
                    return f"Regarding the previous response about inspection results: {query}"
        
        return query
    
    # Use LLM for sophisticated rewriting
    try:
        # Format chat history
        history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in chat_history[-6:]  # Last 3 exchanges
        ])
        
        prompt = CHAT_HISTORY_AWARE_PROMPT.format(
            chat_history=history_str,
            question=query
        )
        
        # Call LLM
        rewritten = llm.invoke(prompt)
        
        logger.debug(f"Rewrote query: '{query}' -> '{rewritten}'")
        
        return rewritten.strip()
    
    except Exception as e:
        logger.warning(f"Query rewriting failed, using original: {e}")
        return query


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_memory_manager = None

def get_memory_manager() -> ChatMemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ChatMemoryManager()
    return _memory_manager


def get_session_history(session_id: str) -> SQLiteChatHistory:
    """Get chat history for a session."""
    manager = get_memory_manager()
    return manager.get_history(session_id)