import pytest
from src.agents.portfolio_health import PortfolioHealthAgent
from src.models import (
    UserContext, UserProfile, PortfolioHolding,
    ClassificationResult, ExtractedEntities
)


class TestPortfolioHealthAgent:
    """Test suite for portfolio health agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance"""
        return PortfolioHealthAgent()
    
    @pytest.fixture
    def basic_classification(self):
        """Basic classification for portfolio health"""
        return ClassificationResult(
            intent="check portfolio health",
            agent="portfolio_health",
            entities=ExtractedEntities(),
            confidence=0.95,
            safety_verdict="safe"
        )
    
    @pytest.mark.asyncio
    async def test_handles_empty_portfolio(self, agent, empty_portfolio_context, basic_classification):
        """Should handle empty portfolio without crashing"""
        
        result = await agent.execute(
            query="How is my portfolio?",
            classification=basic_classification,
            user_context=empty_portfolio_context
        )
        
        # Should not crash
        assert result is not None
        assert result.is_empty is True
        assert result.build_mode_message is not None
        assert len(result.observations) > 0
        assert result.disclaimer  # Must include disclaimer
    
    @pytest.mark.asyncio
    async def test_empty_portfolio_suggests_building(self, agent, empty_portfolio_context, basic_classification):
        """Empty portfolio should guide user toward building"""
        
        result = await agent.execute(
            query="What should I do?",
            classification=basic_classification,
            user_context=empty_portfolio_context
        )
        
        # Should suggest starting to invest
        message = result.build_mode_message.lower()
        assert "build" in message or "start" in message or "invest" in message
    
    @pytest.mark.asyncio
    async def test_calculates_concentration_risk(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Should calculate concentration metrics correctly"""
        
        result = await agent.execute(
            query="How concentrated is my portfolio?",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        assert result.concentration_risk is not None
        assert result.concentration_risk.top_position_pct > 0
        assert result.concentration_risk.top_3_positions_pct > 0
        assert result.concentration_risk.flag in ["low", "medium", "high", "extreme"]
    
    @pytest.mark.asyncio
    async def test_detects_high_concentration(self, agent, basic_classification, mock_yfinance):
        """Should flag high concentration portfolios"""
        
        # Create highly concentrated portfolio (90% in one stock)
        concentrated_context = UserContext(
            user_id="test",
            session_id="test",
            profile=UserProfile(user_id="test", risk_profile="aggressive"),
            portfolio=[
                PortfolioHolding(ticker="NVDA", quantity=100, purchase_price=400),
                PortfolioHolding(ticker="AAPL", quantity=1, purchase_price=150)
            ]
        )
        
        result = await agent.execute(
            query="Portfolio health check",
            classification=basic_classification,
            user_context=concentrated_context
        )
        
        # Should flag as high or extreme concentration
        assert result.concentration_risk.flag in ["high", "extreme"]
        assert result.concentration_risk.top_position_pct > 50
    
    @pytest.mark.asyncio
    async def test_calculates_performance_metrics(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Should calculate portfolio performance"""
        
        result = await agent.execute(
            query="What's my return?",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        assert result.performance is not None
        assert result.performance.total_return_pct is not None
        # Return can be positive or negative
        assert isinstance(result.performance.total_return_pct, (int, float))
    
    @pytest.mark.asyncio
    async def test_compares_to_benchmark(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Should compare portfolio to appropriate benchmark"""
        
        result = await agent.execute(
            query="Am I beating the market?",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        assert result.benchmark_comparison is not None
        assert result.benchmark_comparison.benchmark  # Should have benchmark name
        assert result.benchmark_comparison.portfolio_return_pct is not None
        assert result.benchmark_comparison.benchmark_return_pct is not None
        assert result.benchmark_comparison.alpha_pct is not None
    
    @pytest.mark.asyncio
    async def test_detects_us_benchmark(self, agent, basic_classification, mock_yfinance):
        """Should use S&P 500 for US-heavy portfolios"""
        
        us_portfolio_context = UserContext(
            user_id="test",
            session_id="test",
            profile=UserProfile(user_id="test", risk_profile="moderate"),
            portfolio=[
                PortfolioHolding(ticker="AAPL", quantity=10, purchase_price=150),
                PortfolioHolding(ticker="MSFT", quantity=10, purchase_price=300),
                PortfolioHolding(ticker="GOOGL", quantity=5, purchase_price=100)
            ]
        )
        
        # Mock the benchmark detection
        benchmark = agent._detect_benchmark(us_portfolio_context.portfolio)
        
        # Should be SPY for US stocks
        assert benchmark == "SPY"
    
    @pytest.mark.asyncio
    async def test_generates_observations(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Should generate plain-language observations"""
        
        result = await agent.execute(
            query="Give me insights",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        assert len(result.observations) > 0
        
        # Each observation should have severity and text
        for obs in result.observations:
            assert obs.severity in ["info", "warning", "critical"]
            assert len(obs.text) > 0
            # Should be plain language, not jargon-heavy
            assert len(obs.text.split()) > 3  # At least a few words
    
    @pytest.mark.asyncio
    async def test_limits_observation_count(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Should not overwhelm user with too many observations"""
        
        result = await agent.execute(
            query="Analysis please",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        # Should limit to 3-4 most important observations
        assert len(result.observations) <= 4
    
    @pytest.mark.asyncio
    async def test_always_includes_disclaimer(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Every response must include regulatory disclaimer"""
        
        result = await agent.execute(
            query="Portfolio check",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        assert result.disclaimer
        assert len(result.disclaimer) > 50  # Should be substantial
        assert "not" in result.disclaimer.lower() and "advice" in result.disclaimer.lower()

    @pytest.mark.asyncio
    async def test_returns_valid_schema(self, agent, mock_user_context, basic_classification, mock_yfinance):
        """Response should match expected schema"""
        
        result = await agent.execute(
            query="Check my portfolio",
            classification=basic_classification,
            user_context=mock_user_context
        )
        
        # Check all required fields are present
        result_dict = result.model_dump()
        
        # Should have these top-level keys
        assert "observations" in result_dict
        assert "disclaimer" in result_dict
        
        # If portfolio is not empty, should have metrics
        if not result.is_empty:
            assert "concentration_risk" in result_dict
            assert "performance" in result_dict
            assert "benchmark_comparison" in result_dict
    
    @pytest.mark.asyncio
    async def test_handles_invalid_ticker_gracefully(self, agent, basic_classification):
        """Should handle invalid tickers without crashing"""
        
        bad_ticker_context = UserContext(
            user_id="test",
            session_id="test",
            profile=UserProfile(user_id="test", risk_profile="moderate"),
            portfolio=[
                PortfolioHolding(ticker="INVALID123", quantity=10, purchase_price=100)
            ]
        )
        
        # Should not crash even with invalid ticker
        try:
            result = await agent.execute(
                query="Portfolio health",
                classification=basic_classification,
                user_context=bad_ticker_context
            )
            # If it returns, should have some observations
            assert result is not None
        except Exception as e:
            # If it raises, should be a handled exception
            pytest.fail(f"Agent crashed on invalid ticker: {e}")
    
    @pytest.mark.asyncio
    async def test_handles_missing_purchase_price(self, agent, basic_classification, mock_yfinance):
        """Should handle holdings without purchase price"""
        
        no_price_context = UserContext(
            user_id="test",
            session_id="test",
            profile=UserProfile(user_id="test", risk_profile="moderate"),
            portfolio=[
                PortfolioHolding(ticker="AAPL", quantity=10, purchase_price=None)
            ]
        )
        
        result = await agent.execute(
            query="Check portfolio",
            classification=basic_classification,
            user_context=no_price_context
        )
        
        # Should still return valid response
        assert result is not None
        assert result.concentration_risk is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])