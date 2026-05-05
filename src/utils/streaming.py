import json
from typing import Dict, Any, AsyncGenerator


class SSEStreamManager:
    """
    Manager for SSE streaming
    Provides helpers for formatting events according to SSE protocol
    """
    
    @staticmethod
    def format_event(event: str, data: Dict[str, Any]) -> str:
        """
        Format a single SSE event
        
        SSE Protocol format:
        event: <event_name>
        data: <json_data>
        
        (blank line to separate events)
        """
        
        formatted = f"event: {event}\n"
        formatted += f"data: {json.dumps(data)}\n\n"
        
        return formatted
    
    @staticmethod
    def format_comment(comment: str) -> str:
        """
        Format an SSE comment (starts with :)
        Comments are ignored by clients but useful for debugging
        """
        return f": {comment}\n\n"
    
    @staticmethod
    async def stream_events(events: list) -> AsyncGenerator[str, None]:
        """
        Stream a list of events
        
        Args:
            events: List of (event_name, data) tuples
            
        Yields:
            Formatted SSE strings
        """
        
        for event_name, event_data in events:
            yield SSEStreamManager.format_event(event_name, event_data)
    
    @staticmethod
    def create_error_event(code: str, message: str) -> str:
        """
        Create a standard error event
        """
        return SSEStreamManager.format_event(
            "error",
            {
                "code": code,
                "message": message
            }
        )
    
    @staticmethod
    def create_done_event() -> str:
        """
        Create a standard done event to signal stream completion
        """
        return SSEStreamManager.format_event("done", {})


# Helper functions for common use cases

def sse_event(event: str, **data) -> str:
    """
    Quick helper to create an SSE event
    
    Usage:
        sse_event("classification", intent="portfolio", agent="portfolio_health")
    """
    return SSEStreamManager.format_event(event, data)


def sse_chunk(text: str) -> str:
    """
    Create a text chunk event (for streaming text responses)
    """
    return SSEStreamManager.format_event("chunk", {"text": text})


def sse_error(code: str, message: str) -> str:
    """
    Quick error event
    """
    return SSEStreamManager.create_error_event(code, message)


def sse_done() -> str:
    """
    Quick done event
    """
    return SSEStreamManager.create_done_event()


# Testing
if __name__ == "__main__":
    print("SSE Stream Manager Test:--")
    
    # Test event formatting
    event1 = sse_event("test", message="Hello world", value=123)
    print("Event 1:")
    print(event1)
    
    # Test error
    error = sse_error("TEST_ERROR", "This is a test error")
    print("Error event:")
    print(error)
    
    # Test done
    done = sse_done()
    print("Done event:")
    print(done)