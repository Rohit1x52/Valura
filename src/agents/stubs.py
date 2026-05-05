from .base import BaseAgent
from ..models import ClassificationResult, UserContext, StubAgentResponse


class MarketResearchAgent(BaseAgent):
    """Stub for market research agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="market_research",
            intent=classification.intent,
            entities=classification.entities,
            message="Market research functionality coming soon. This agent will provide market data and company research."
        )


class InvestmentStrategyAgent(BaseAgent):
    """Stub for investment strategy agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="investment_strategy",
            intent=classification.intent,
            entities=classification.entities,
            message="Investment strategy planning functionality coming soon."
        )


class FinancialCalculatorAgent(BaseAgent):
    """Stub for financial calculator agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="financial_calculator",
            intent=classification.intent,
            entities=classification.entities,
            message="Financial calculator functionality coming soon. This will handle compound interest, retirement planning, etc."
        )


class RiskAssessmentAgent(BaseAgent):
    """Stub for risk assessment agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="risk_assessment",
            intent=classification.intent,
            entities=classification.entities,
            message="Risk assessment functionality coming soon."
        )


class RecommendationAgent(BaseAgent):
    """Stub for recommendation agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="recommendation",
            intent=classification.intent,
            entities=classification.entities,
            message="Investment recommendation functionality coming soon."
        )


class SupportAgent(BaseAgent):
    """Stub for general support agent"""
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> StubAgentResponse:
        
        return StubAgentResponse(
            agent="support",
            intent=classification.intent,
            entities=classification.entities,
            message="General support functionality coming soon. For now, please rephrase your question."
        )