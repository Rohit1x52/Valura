from typing import Dict, Any
from ..models import ClassificationResult, UserContext, PortfolioHealthResponse, StubAgentResponse
from ..config import IMPLEMENTED_AGENTS


class Router:
    """
    Routes classified queries to the appropriate agent
    Returns stub responses for unimplemented agents
    """
    
    def __init__(self):
        """Initialize router with agent registry"""
        # Import agents here to avoid circular imports
        from ..agents.portfolio_health import PortfolioHealthAgent
        
        # Registry of implemented agents
        self._agents = {
            "portfolio_health": PortfolioHealthAgent()
        }
        
        print(f"[Router] Initialized with {len(self._agents)} implemented agents")
    
    async def route(self, 
                    classification: ClassificationResult,
                    user_context: UserContext,
                    original_query: str) -> Dict[str, Any]:
        """
        Route the query to appropriate agent
        
        Args:
            classification: Result from intent classifier
            user_context: User profile and portfolio data
            original_query: The original user query text
            
        Returns:
            Agent response as dict (either full response or stub)
        """
        
        agent_name = classification.agent
        
        # Check if agent is implemented
        if agent_name in self._agents:
            # Call the implemented agent
            agent = self._agents[agent_name]
            print(f"[Router] Routing to implemented agent: {agent_name}")
            
            try:
                response = await agent.execute(
                    query=original_query,
                    classification=classification,
                    user_context=user_context
                )
                return response.model_dump()
                
            except Exception as e:
                print(f"[Router] Agent {agent_name} execution failed: {e}")
                # Return error response
                return {
                    "error": True,
                    "agent": agent_name,
                    "message": f"Agent execution failed: {str(e)}"
                }
        
        else:
            # Return stub response for unimplemented agents
            print(f"[Router] Agent {agent_name} not implemented, returning stub")
            return self._create_stub_response(classification, original_query)
    
    def _create_stub_response(self, 
                             classification: ClassificationResult,
                             original_query: str) -> Dict[str, Any]:
        """
        Create a structured stub response for unimplemented agents
        
        This meets the requirement: router must work even when destination is a stub
        """
        
        stub = StubAgentResponse(
            agent=classification.agent,
            intent=classification.intent,
            entities=classification.entities,
            message=(
                f"The {classification.agent} agent is not yet implemented in this build. "
                f"Your query has been classified correctly, but the full implementation "
                f"will be available in a future release."
            ),
            status="not_implemented"
        )
        
        return stub.model_dump()
    
    def get_available_agents(self) -> list:
        """Return list of implemented agent names"""
        return list(self._agents.keys())


# Quick test
if __name__ == "__main__":
    import asyncio
    from ..models import ExtractedEntities, UserProfile, PortfolioHolding
    
    async def test():
        router = Router()
        
        # Mock classification
        classification = ClassificationResult(
            intent="check portfolio health",
            agent="portfolio_health",
            entities=ExtractedEntities(),
            confidence=0.95,
            safety_verdict="safe"
        )
        
        # Mock user context
        user_context = UserContext(
            user_id="test_user",
            session_id="test_session",
            profile=UserProfile(
                user_id="test_user",
                risk_profile="moderate"
            ),
            portfolio=[
                PortfolioHolding(ticker="AAPL", quantity=10, purchase_price=150),
                PortfolioHolding(ticker="MSFT", quantity=5, purchase_price=300)
            ]
        )
        
        print("Router Test:")
        
        # Test implemented agent
        print("\n1. Testing implemented agent (portfolio_health):--")
        response = await router.route(classification, user_context, "How is my portfolio?")
        print(f"Response type: {type(response)}")
        print(f"Keys: {response.keys()}")
        
        # Test stub agent
        print("\n2. Testing stub agent (market_research):--")
        stub_classification = ClassificationResult(
            intent="research Apple stock",
            agent="market_research",
            entities=ExtractedEntities(tickers=["AAPL"]),
            confidence=0.92,
            safety_verdict="safe"
        )
        
        stub_response = await router.route(stub_classification, user_context, "Tell me about AAPL")
        print(f"Stub response: {stub_response}")
    
    asyncio.run(test())