from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class PortfolioHolding(BaseModel):
    """Single position in a portfolio"""
    ticker: str
    quantity: float
    purchase_price: Optional[float] = None
    purchase_date: Optional[str] = None  # ISO format


class UserProfile(BaseModel):
    """User profile and KYC information"""
    user_id: str
    risk_profile: Literal["conservative", "moderate", "aggressive", "very_aggressive"]
    investment_goals: List[str] = []
    kyc_status: Literal["pending", "approved", "rejected"] = "pending"
    country: str = "US"
    preferred_currency: str = "USD"


class UserContext(BaseModel):
    """Complete user context passed with each query"""
    user_id: str
    session_id: str
    profile: UserProfile
    portfolio: List[PortfolioHolding] = []
    metadata: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    """Incoming query from the user"""
    query: str
    user_context: UserContext
    stream: bool = True  

class ExtractedEntities(BaseModel):
    """Entities extracted from user query"""
    tickers: List[str] = []
    amounts: List[float] = []
    time_periods: List[str] = []
    topics: List[str] = []
    sectors: List[str] = []
    
    amount: Optional[float] = None  
    rate: Optional[float] = None  
    period_years: Optional[int] = None  


class ClassificationResult(BaseModel):
    """Result of intent classification"""
    intent: str  
    agent: str  
    entities: ExtractedEntities
    confidence: float = Field(ge=0.0, le=1.0)
    safety_verdict: Literal["safe", "educational", "flagged"] = "safe"
    reasoning: Optional[str] = None  

class Observation(BaseModel):
    """Single observation or insight"""
    severity: Literal["info", "warning", "critical"]
    text: str

class ConcentrationRisk(BaseModel):
    """Portfolio concentration metrics"""
    top_position_pct: float
    top_3_positions_pct: float
    flag: Literal["low", "medium", "high", "extreme"]

class Performance(BaseModel):
    """Portfolio performance metrics"""
    total_return_pct: float
    annualized_return_pct: Optional[float] = None
    time_period: Optional[str] = None

class BenchmarkComparison(BaseModel):
    """Comparison against market benchmark"""
    benchmark: str  # e.g., "S&P 500"
    portfolio_return_pct: float
    benchmark_return_pct: float
    alpha_pct: float  # outperformance

class PortfolioHealthResponse(BaseModel):
    """Complete portfolio health check response"""
    concentration_risk: Optional[ConcentrationRisk] = None
    performance: Optional[Performance] = None
    benchmark_comparison: Optional[BenchmarkComparison] = None
    observations: List[Observation]
    disclaimer: str
    
    # For empty portfolios
    is_empty: bool = False
    build_mode_message: Optional[str] = None


class StubAgentResponse(BaseModel):
    """Response from unimplemented agents"""
    agent: str
    intent: str
    entities: ExtractedEntities
    message: str
    status: str = "not_implemented"

class SafetyResult(BaseModel):
    """Result from safety guard check"""
    is_safe: bool
    blocked_category: Optional[str] = None
    response_message: Optional[str] = None
    confidence: float = 1.0  # safety guard is deterministic

class ConversationTurn(BaseModel):
    """Single turn in a conversation"""
    turn_id: int
    timestamp: datetime
    query: str
    agent: str
    response: Dict[str, Any]  # flexible to store any agent response
    classification: Optional[ClassificationResult] = None


class Session(BaseModel):
    """User session with conversation history"""
    session_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    turns: List[ConversationTurn] = []
    
    def get_recent_turns(self, limit: int = 3) -> List[ConversationTurn]:
        """Get the most recent N turns"""
        return self.turns[-limit:] if len(self.turns) > limit else self.turns

class SSEEvent(BaseModel):
    """Server-Sent Event structure"""
    event: str
    data: Dict[str, Any]
    
    def format(self) -> str:
        """Format as SSE protocol message"""
        import json
        # SSE format: event: xxx\ndata: {...}\n\n
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


def normalize_ticker(ticker: str) -> str:
    """Normalize ticker symbols for comparison"""
    # Remove exchange suffix if present, convert to uppercase
    base_ticker = ticker.split('.')[0].upper()
    return base_ticker


def compare_entities(expected: ExtractedEntities, actual: ExtractedEntities, tolerance: float = 0.05) -> bool:
    expected_tickers = set(normalize_ticker(t) for t in expected.tickers)
    actual_tickers = set(normalize_ticker(t) for t in actual.tickers)
    
    if not expected_tickers.issubset(actual_tickers):
        return False
    
    if expected.amount is not None:
        if actual.amount is None:
            return False
        diff = abs(expected.amount - actual.amount) / expected.amount
        if diff > tolerance:
            return False
    
    return True