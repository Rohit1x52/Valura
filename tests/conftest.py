import pytest
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from src.models import (
    UserContext, UserProfile, PortfolioHolding,
    ClassificationResult, ExtractedEntities
)

@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to fixtures directory"""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session")
def safety_queries(fixtures_dir):
    """Load safety test queries"""
    path = fixtures_dir / "test_queries" / "safety_pairs.json"
    if not path.exists():
        # Return empty list if fixtures not available yet
        return []
    
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def intent_queries(fixtures_dir):
    """Load intent classification queries"""
    path = fixtures_dir / "test_queries" / "intent_classification.json"
    if not path.exists():
        return []
    
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def user_profiles(fixtures_dir):
    """Load user profile fixtures"""
    profiles_dir = fixtures_dir / "user_profiles"
    if not profiles_dir.exists():
        return {}
    
    profiles = {}
    for profile_file in profiles_dir.glob("*.json"):
        with open(profile_file) as f:
            profile_data = json.load(f)
            profiles[profile_file.stem] = profile_data
    
    return profiles

@pytest.fixture
def mock_user_context():
    """Basic mock user context for testing"""
    return UserContext(
        user_id="test_user_001",
        session_id="test_session_001",
        profile=UserProfile(
            user_id="test_user_001",
            risk_profile="moderate",
            kyc_status="approved"
        ),
        portfolio=[
            PortfolioHolding(ticker="AAPL", quantity=10, purchase_price=150.0),
            PortfolioHolding(ticker="MSFT", quantity=5, purchase_price=300.0)
        ]
    )


@pytest.fixture
def empty_portfolio_context():
    """User context with empty portfolio"""
    return UserContext(
        user_id="test_user_empty",
        session_id="test_session_empty",
        profile=UserProfile(
            user_id="test_user_empty",
            risk_profile="conservative",
            kyc_status="approved"
        ),
        portfolio=[]
    )


@pytest.fixture
def aggressive_trader_context():
    """User with aggressive risk profile and concentrated portfolio"""
    return UserContext(
        user_id="test_user_aggressive",
        session_id="test_session_aggressive",
        profile=UserProfile(
            user_id="test_user_aggressive",
            risk_profile="very_aggressive",
            kyc_status="approved"
        ),
        portfolio=[
            PortfolioHolding(ticker="NVDA", quantity=100, purchase_price=400.0),
            PortfolioHolding(ticker="TSLA", quantity=20, purchase_price=200.0)
        ]
    )

@pytest.fixture
def mock_classification():
    """Basic mock classification result"""
    return ClassificationResult(
        intent="check portfolio health",
        agent="portfolio_health",
        entities=ExtractedEntities(),
        confidence=0.95,
        safety_verdict="safe"
    )


@pytest.fixture
def mock_market_research_classification():
    """Mock classification for market research query"""
    return ClassificationResult(
        intent="research stock",
        agent="market_research",
        entities=ExtractedEntities(tickers=["AAPL"]),
        confidence=0.92,
        safety_verdict="safe"
    )

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    def create_response(content: str, input_tokens: int = 100, output_tokens: int = 200):
        """Create a mock OpenAI response"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = input_tokens
        mock_response.usage.completion_tokens = output_tokens
        mock_response.usage.total_tokens = input_tokens + output_tokens
        return mock_response
    
    return create_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client for testing without API calls"""
    
    with patch('src.core.classifier.OpenAI') as mock_openai:
        # Create mock client instance
        mock_client = Mock()
        
        # Default classification response
        default_classification = {
            "intent": "test intent",
            "agent": "portfolio_health",
            "entities": {
                "tickers": [],
                "amounts": [],
                "time_periods": [],
                "topics": [],
                "sectors": []
            },
            "confidence": 0.95,
            "safety_verdict": "safe",
            "reasoning": "Test classification"
        }
        
        # Setup the mock to return our response
        mock_client.chat.completions.create.return_value = mock_openai_response(
            json.dumps(default_classification)
        )
        
        # Make OpenAI() constructor return our mock client
        mock_openai.return_value = mock_client
        
        yield mock_client

@pytest.fixture
def mock_yfinance():
    """Mock yfinance for testing without network calls"""
    
    with patch('src.agents.portfolio_health.yf') as mock_yf:
        # Create mock ticker
        def create_mock_ticker(ticker: str):
            mock_ticker = Mock()
            
            # Mock info
            mock_ticker.info = {
                "currentPrice": 150.0 if ticker == "AAPL" else 300.0,
                "sector": "Technology",
                "exchange": "NASDAQ",
                "regularMarketPrice": 150.0 if ticker == "AAPL" else 300.0
            }
            
            # Mock history
            import pandas as pd
            dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
            prices = [100 + i * 0.2 for i in range(252)]
            
            mock_ticker.history.return_value = pd.DataFrame({
                'Close': prices,
                'Open': prices,
                'High': [p * 1.02 for p in prices],
                'Low': [p * 0.98 for p in prices],
                'Volume': [1000000] * 252
            }, index=dates)
            
            return mock_ticker
        
        # Setup yfinance.Ticker to return our mock
        mock_yf.Ticker.side_effect = create_mock_ticker
        
        yield mock_yf

@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database path for testing"""
    db_path = tmp_path / "test_sessions.db"
    return str(db_path)


@pytest.fixture
def mock_session_manager(temp_db_path):
    """SessionManager with temporary database"""
    from src.core.session import SessionManager
    manager = SessionManager(db_path=temp_db_path)
    yield manager
    # Cleanup happens automatically with tmp_path

@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing FastAPI endpoints"""
    from fastapi.testclient import TestClient
    from src.main import app
    
    # Note: This will use actual components
    # For full mocking, we'd need to override dependencies
    return TestClient(app)

@pytest.fixture
def normalize_ticker():
    """Helper function to normalize tickers for comparison"""
    def _normalize(ticker: str) -> str:
        # Remove exchange suffix, convert to uppercase
        return ticker.split('.')[0].upper()
    return _normalize


@pytest.fixture
def entity_matcher():
    """Helper for matching entities with tolerance"""
    def _match(expected: Dict, actual: Dict, tolerance: float = 0.05) -> bool:
        """
        Match entities with tolerance for numeric values
        """
        # Ticker matching (normalized)
        if "tickers" in expected:
            expected_tickers = set(t.split('.')[0].upper() for t in expected["tickers"])
            actual_tickers = set(t.split('.')[0].upper() for t in actual.get("tickers", []))
            
            if not expected_tickers.issubset(actual_tickers):
                return False
        
        # Numeric fields with tolerance
        numeric_fields = ["amount", "rate", "period_years"]
        for field in numeric_fields:
            if field in expected and expected[field] is not None:
                if field not in actual or actual[field] is None:
                    return False
                
                expected_val = expected[field]
                actual_val = actual[field]
                diff = abs(expected_val - actual_val) / expected_val if expected_val != 0 else 0
                
                if diff > tolerance:
                    return False
        
        return True
    
    return _match

@pytest.fixture
def async_mock():
    """Helper to create async mocks"""
    def _create_async_mock(*args, **kwargs):
        return AsyncMock(*args, **kwargs)
    return _create_async_mock

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Print useful info when tests start
def pytest_report_header(config):
    """Add custom header to pytest output"""
    return [
        "Valura AI Microservice Test Suite",
        "=" * 60
    ]