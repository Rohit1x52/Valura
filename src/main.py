import asyncio
import json
import time
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from .models import QueryRequest, UserContext, SSEEvent, SafetyResult
from .core import SafetyGuard, IntentClassifier, Router, SessionManager
from .config import config


# Global instances (initialized at startup)
safety_guard: SafetyGuard = None
classifier: IntentClassifier = None
router: Router = None
session_manager: SessionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global safety_guard, classifier, router, session_manager
    
    print("[App] Starting Valura AI microservice...")
    
    # Initialize components
    safety_guard = SafetyGuard()
    classifier = IntentClassifier()
    router = Router()
    session_manager = SessionManager()
    
    print(f"[App] Using model: {config.OPENAI_MODEL}")
    print(f"[App] Environment: {config.ENV}")
    print("[App] All components initialized successfully")
    
    yield
    
    # Shutdown (cleanup if needed)
    print("[App] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Valura AI Microservice",
    description="AI co-investor for wealth management",
    version="0.1.0",
    lifespan=lifespan
)


async def process_query_pipeline(
    query: str,
    user_context: UserContext
) -> AsyncGenerator[str, None]:
    """
    Main pipeline: safety → classifier → router → agent
    Yields SSE-formatted events
    """
    
    start_time = time.perf_counter()
    
    try:
        # Step 1: Safety Guard (synchronous, <10ms)
        yield SSEEvent(
            event="safety_check",
            data={"status": "running"}
        ).format()
        
        safety_result = safety_guard.check(query)
        
        if not safety_result.is_safe:
            # Query blocked by safety guard
            yield SSEEvent(
                event="safety_blocked",
                data={
                    "category": safety_result.blocked_category,
                    "message": safety_result.response_message
                }
            ).format()
            
            yield SSEEvent(event="done", data={}).format()
            return
        
        yield SSEEvent(
            event="safety_passed",
            data={"status": "safe"}
        ).format()
        
        # Step 2: Get or create session
        session = session_manager.get_or_create_session(
            session_id=user_context.session_id,
            user_id=user_context.user_id
        )
        
        # Step 3: Intent Classification (1 LLM call)
        yield SSEEvent(
            event="classification_start",
            data={"status": "running"}
        ).format()
        
        classification_start = time.perf_counter()
        
        # Pass conversation history for follow-up handling
        classification = await asyncio.wait_for(
            classifier.classify(query, session.turns),
            timeout=config.MAX_CLASSIFICATION_TIME
        )
        
        classification_time = time.perf_counter() - classification_start
        
        yield SSEEvent(
            event="classification_complete",
            data={
                "intent": classification.intent,
                "agent": classification.agent,
                "entities": classification.entities.model_dump(),
                "confidence": classification.confidence,
                "time_taken": round(classification_time, 3)
            }
        ).format()
        
        # Step 4: Route to agent
        yield SSEEvent(
            event="agent_start",
            data={"agent": classification.agent}
        ).format()
        
        agent_start = time.perf_counter()
        
        agent_response = await asyncio.wait_for(
            router.route(classification, user_context, query),
            timeout=config.MAX_AGENT_TIME
        )
        
        agent_time = time.perf_counter() - agent_start
        
        yield SSEEvent(
            event="agent_complete",
            data={
                "time_taken": round(agent_time, 3)
            }
        ).format()
        
        # Step 5: Stream final result
        yield SSEEvent(
            event="result",
            data={
                "agent": classification.agent,
                "response": agent_response,
                "classification": {
                    "intent": classification.intent,
                    "confidence": classification.confidence,
                    "safety_verdict": classification.safety_verdict
                }
            }
        ).format()
        
        # Save turn to session
        session_manager.add_turn(
            session_id=user_context.session_id,
            query=query,
            agent=classification.agent,
            response=agent_response,
            classification=classification
        )
        
        # Step 6: Done
        total_time = time.perf_counter() - start_time
        
        yield SSEEvent(
            event="metrics",
            data={
                "total_time": round(total_time, 3),
                "classification_time": round(classification_time, 3),
                "agent_time": round(agent_time, 3)
            }
        ).format()
        
        yield SSEEvent(event="done", data={}).format()
        
    except asyncio.TimeoutError:
        # Timeout exceeded
        yield SSEEvent(
            event="error",
            data={
                "code": "TIMEOUT",
                "message": "Request exceeded maximum processing time"
            }
        ).format()
        yield SSEEvent(event="done", data={}).format()
        
    except Exception as e:
        # Unexpected error
        print(f"[App] Pipeline error: {e}")
        yield SSEEvent(
            event="error",
            data={
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred processing your request"
            }
        ).format()
        yield SSEEvent(event="done", data={}).format()


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint - processes user queries with SSE streaming
    
    Pipeline:
    1. Safety guard (sync, <10ms)
    2. Intent classification (1 LLM call)
    3. Agent routing
    4. Response streaming
    """
    
    # Extract query and context
    query = request.query
    user_context = request.user_context
    
    print(f"[App] New query from user {user_context.user_id}: {query[:50]}...")
    
    # Return SSE streaming response
    return StreamingResponse(
        process_query_pipeline(query, user_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": config.OPENAI_MODEL,
        "environment": config.ENV,
        "version": "0.1.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Valura AI Microservice",
        "version": "0.1.0",
        "endpoints": {
            "query": "/query (POST)",
            "health": "/health (GET)"
        }
    }


# For local development
if __name__ == "__main__":
    import uvicorn
    
    print("Starting Valura AI microservice...")
    print(f"Model: {config.OPENAI_MODEL}")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )