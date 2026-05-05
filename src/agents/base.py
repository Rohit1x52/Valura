"""
Base Agent Interface
All agents inherit from this to ensure consistent structure
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from ..models import ClassificationResult, UserContext


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    Defines the interface that all agents must implement
    """
    
    def __init__(self):
        """Initialize the agent"""
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> Any:
        """
        Execute the agent's logic
        
        Args:
            query: Original user query
            classification: Classification result with intent and entities
            user_context: User profile, portfolio, etc.
            
        Returns:
            Agent-specific response (typically a Pydantic model)
        """
        pass
    
    def _validate_inputs(self, 
                        query: str,
                        classification: ClassificationResult,
                        user_context: UserContext):
        """
        Basic input validation
        Override in subclasses for specific validation
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not classification:
            raise ValueError("Classification result is required")
        
        if not user_context:
            raise ValueError("User context is required")
    
    def __repr__(self):
        return f"<{self.name}>"