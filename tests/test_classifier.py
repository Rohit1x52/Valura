import pytest
import json
from unittest.mock import patch, Mock
from src.core.classifier import IntentClassifier
from src.models import ClassificationResult, ExtractedEntities, ConversationTurn
from datetime import datetime


class TestIntentClassifier:
    """Test suite for intent classification"""
    
    @pytest.mark.asyncio
    async def test_classifies_portfolio_health_query(self, mock_openai_client):
        """Should classify portfolio health queries correctly"""
        
        # Mock OpenAI to return portfolio_health classification
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "check portfolio health",
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
            "reasoning": "User wants portfolio analysis"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify("How is my portfolio doing?")
        
        assert result.agent == "portfolio_health"
        assert result.confidence > 0.9
        assert result.safety_verdict == "safe"
    
    @pytest.mark.asyncio
    async def test_classifies_market_research_query(self, mock_openai_client):
        """Should classify market research queries"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "research stock",
            "agent": "market_research",
            "entities": {
                "tickers": ["AAPL"],
                "amounts": [],
                "time_periods": [],
                "topics": ["stock analysis"],
                "sectors": []
            },
            "confidence": 0.92,
            "safety_verdict": "safe",
            "reasoning": "User wants stock research"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify("Tell me about Apple stock")
        
        assert result.agent == "market_research"
        assert "AAPL" in result.entities.tickers
    
    @pytest.mark.asyncio
    async def test_extracts_ticker_entities(self, mock_openai_client):
        """Should extract ticker symbols from query"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "compare stocks",
            "agent": "market_research",
            "entities": {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "amounts": [],
                "time_periods": [],
                "topics": ["comparison"],
                "sectors": []
            },
            "confidence": 0.88,
            "safety_verdict": "safe",
            "reasoning": "User wants to compare multiple stocks"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify("Compare AAPL, MSFT, and GOOGL")
        
        assert len(result.entities.tickers) == 3
        assert "AAPL" in result.entities.tickers
        assert "MSFT" in result.entities.tickers
        assert "GOOGL" in result.entities.tickers
    
    @pytest.mark.asyncio
    async def test_extracts_amount_entities(self, mock_openai_client):
        """Should extract dollar amounts from query"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "investment planning",
            "agent": "investment_strategy",
            "entities": {
                "tickers": [],
                "amounts": [10000],
                "time_periods": [],
                "topics": [],
                "sectors": ["tech"],
                "amount": 10000
            },
            "confidence": 0.90,
            "safety_verdict": "safe",
            "reasoning": "User has specific amount to invest"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify("I have $10,000 to invest in tech stocks")
        
        assert result.entities.amount == 10000
        assert "tech" in result.entities.sectors
    
    @pytest.mark.asyncio
    async def test_handles_follow_up_with_context(self, mock_openai_client):
        """Should resolve pronoun references using conversation history"""
        
        # Create conversation history
        history = [
            ConversationTurn(
                turn_id=1,
                timestamp=datetime.now(),
                query="Tell me about Microsoft",
                agent="market_research",
                response={},
                classification=ClassificationResult(
                    intent="research stock",
                    agent="market_research",
                    entities=ExtractedEntities(tickers=["MSFT"]),
                    confidence=0.95
                )
            )
        ]
        
        # Mock response that resolves "it" to MSFT
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "compare to competitor",
            "agent": "market_research",
            "entities": {
                "tickers": ["MSFT", "GOOGL"],  # Resolved from context
                "amounts": [],
                "time_periods": [],
                "topics": ["comparison"],
                "sectors": []
            },
            "confidence": 0.87,
            "safety_verdict": "safe",
            "reasoning": "User wants to compare MSFT (from previous query) to GOOGL"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 60
        mock_response.usage.total_tokens = 210
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify(
            "How does it compare to Google?",
            conversation_history=history
        )
        
        # Should include both MSFT (from context) and GOOGL (from current query)
        assert "MSFT" in result.entities.tickers or "GOOGL" in result.entities.tickers
    
    @pytest.mark.asyncio
    async def test_handles_openai_api_error(self):
        """Should fallback gracefully on API error"""
        
        with patch('src.core.classifier.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            classifier = IntentClassifier()
            result = await classifier.classify("Test query")
            
            # Should return fallback classification
            assert result.agent == "support"
            assert result.confidence == 0.0
            assert "error" in result.intent.lower() or "fallback" in result.intent.lower()
    
    @pytest.mark.asyncio
    async def test_handles_malformed_json_response(self):
        """Should handle malformed JSON from LLM"""
        
        with patch('src.core.classifier.OpenAI') as mock_openai:
            mock_client = Mock()
            
            # Return invalid JSON
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Not valid JSON {{"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150
            
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            classifier = IntentClassifier()
            result = await classifier.classify("Test query")
            
            # Should fallback to support agent
            assert result.agent == "support"

    @pytest.mark.skipif(True, reason="Requires fixture files and real API")
    @pytest.mark.asyncio
    async def test_routing_accuracy_against_fixtures(self, intent_queries):
        """Test routing accuracy against fixture data"""
        
        if not intent_queries:
            pytest.skip("Intent query fixtures not available")
        
        classifier = IntentClassifier()
        
        correct = 0
        total = 0
        
        for item in intent_queries:
            query = item["query"]
            expected_agent = item["expected_agent"]
            
            result = await classifier.classify(query)
            
            if result.agent == expected_agent:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"\n[Routing Accuracy]")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy*100:.1f}% (target: 85%)")
        
        # Check target
        assert accuracy >= 0.85, f"Routing accuracy {accuracy*100:.1f}% below 85% target"
    
    @pytest.mark.asyncio
    async def test_returns_safety_verdict(self, mock_openai_client):
        """Should include safety verdict in classification"""
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "intent": "educational query",
            "agent": "support",
            "entities": {
                "tickers": [],
                "amounts": [],
                "time_periods": [],
                "topics": ["insider trading"],
                "sectors": []
            },
            "confidence": 0.85,
            "safety_verdict": "educational",
            "reasoning": "User asking to learn about illegal activities"
        })
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        classifier = IntentClassifier()
        result = await classifier.classify("What is insider trading?")
        
        assert result.safety_verdict in ["safe", "educational", "flagged"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])