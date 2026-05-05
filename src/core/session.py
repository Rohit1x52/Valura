import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from pathlib import Path
from ..models import Session, ConversationTurn, ClassificationResult
from ..config import config


class SessionManager:
    """
    Manages user sessions and conversation history
    Provides persistence via SQLite
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize session manager with database"""
        self.db_path = db_path or config.DB_PATH
        
        # Create data directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        print(f"[SessionManager] Initialized with database: {self.db_path}")
    
    def _init_db(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Conversation turns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                agent TEXT NOT NULL,
                response TEXT NOT NULL,
                classification TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_turns 
            ON turns(session_id, turn_id)
        """)
        
        conn.commit()
        conn.close()
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session with conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session metadata
        cursor.execute(
            "SELECT session_id, user_id, created_at, updated_at FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Get conversation turns
        cursor.execute(
            """SELECT turn_id, timestamp, query, agent, response, classification 
               FROM turns WHERE session_id = ? ORDER BY turn_id""",
            (session_id,)
        )
        
        turns = []
        for turn_row in cursor.fetchall():
            turn_id, timestamp, query, agent, response_json, classification_json = turn_row
            
            # Parse JSON fields
            response = json.loads(response_json)
            classification = None
            if classification_json:
                classification = ClassificationResult(**json.loads(classification_json))
            
            turns.append(ConversationTurn(
                turn_id=turn_id,
                timestamp=datetime.fromisoformat(timestamp),
                query=query,
                agent=agent,
                response=response,
                classification=classification
            ))
        
        conn.close()
        
        return Session(
            session_id=row[0],
            user_id=row[1],
            created_at=datetime.fromisoformat(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
            turns=turns
        )
    
    def create_session(self, session_id: str, user_id: str) -> Session:
        """Create a new session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute(
            """INSERT INTO sessions (session_id, user_id, created_at, updated_at)
               VALUES (?, ?, ?, ?)""",
            (session_id, user_id, now, now)
        )
        
        conn.commit()
        conn.close()
        
        return Session(
            session_id=session_id,
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            turns=[]
        )
    
    def add_turn(self,
                 session_id: str,
                 query: str,
                 agent: str,
                 response: dict,
                 classification: Optional[ClassificationResult] = None):
        """Add a conversation turn to the session"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current turn count
        cursor.execute(
            "SELECT MAX(turn_id) FROM turns WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        next_turn_id = (result[0] or 0) + 1
        
        # Insert turn
        now = datetime.now().isoformat()
        classification_json = classification.model_dump_json() if classification else None
        
        cursor.execute(
            """INSERT INTO turns (session_id, turn_id, timestamp, query, agent, response, classification)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, next_turn_id, now, query, agent, 
             json.dumps(response), classification_json)
        )
        
        # Update session timestamp
        cursor.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id)
        )
        
        conn.commit()
        conn.close()
        
        # Cleanup old turns if we exceed max
        self._cleanup_old_turns(session_id)
    
    def _cleanup_old_turns(self, session_id: str):
        """Remove old turns if session exceeds max length"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count turns
        cursor.execute(
            "SELECT COUNT(*) FROM turns WHERE session_id = ?",
            (session_id,)
        )
        count = cursor.fetchone()[0]
        
        if count > config.SESSION_MAX_TURNS:
            # Delete oldest turns
            cursor.execute(
                """DELETE FROM turns WHERE session_id = ? 
                   AND turn_id <= (
                       SELECT MAX(turn_id) - ? FROM turns WHERE session_id = ?
                   )""",
                (session_id, config.SESSION_MAX_TURNS, session_id)
            )
            conn.commit()
        
        conn.close()
    
    def get_or_create_session(self, session_id: str, user_id: str) -> Session:
        """Get existing session or create new one"""
        session = self.get_session(session_id)
        if session:
            return session
        return self.create_session(session_id, user_id)


# Quick test
if __name__ == "__main__":
    manager = SessionManager(db_path="./test_sessions.db")
    
    print("Session Manager Test:")
    
    # Create session
    session = manager.create_session("test_123", "user_001")
    print(f"Created session: {session.session_id}")
    
    # Add some turns
    manager.add_turn(
        session_id="test_123",
        query="How is my portfolio?",
        agent="portfolio_health",
        response={"status": "ok"},
        classification=ClassificationResult(
            intent="portfolio health",
            agent="portfolio_health",
            entities={},
            confidence=0.95
        )
    )
    
    # Retrieve session
    retrieved = manager.get_session("test_123")
    print(f"Retrieved session with {len(retrieved.turns)} turns")
    print(f"First turn: {retrieved.turns[0].query}")
    
    # Cleanup test db
    import os
    os.remove("./test_sessions.db")
    print("Test completed and cleaned up")