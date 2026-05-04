import json
import time
from typing import Optional, List
from openai import OpenAI
from ..models import ClassificationResult, ExtractedEntities, ConversationTurn
from ..config import config, AGENT_TAXONOMY


class IntentClassifier:
    """
    Classifies user queries and extracts entities in a single LLM call
    Routes to appropriate agent based on intent
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize with OpenAI client"""
        api_key = openai_api_key or config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = config.OPENAI_MODEL
        
        # Build system prompt with agent taxonomy
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt(self) -> str:
        """Construct the classification system prompt"""
        
        agent_descriptions = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in AGENT_TAXONOMY.items()
        ])
        
        prompt = f"""You are an intent classifier for a wealth management AI platform.

Your job: Analyze the user's query and return structured classification information.

Available agents:
{agent_descriptions}

Extract these entities from the query:
- tickers: Stock/ETF symbols mentioned (e.g., AAPL, MSFT, SPY)
- amounts: Dollar amounts or quantities
- time_periods: Time references (e.g., "last quarter", "6 months")
- topics: General topics discussed
- sectors: Industry sectors mentioned
- amount: Primary dollar amount if one is referenced
- rate: Interest rate or return percentage if mentioned
- period_years: Time period in years if specified

Safety assessment:
- "safe": Normal investment query
- "educational": Asking about harmful topics to learn (not to act)
- "flagged": Potentially harmful intent (informational only - guard already ran)

Choose the most appropriate agent based on the query intent.
For follow-up queries, use the conversation context to resolve references.

Return valid JSON matching this structure:
{{
  "intent": "brief description of what user wants",
  "agent": "agent_name",
  "entities": {{
    "tickers": [],
    "amounts": [],
    "time_periods": [],
    "topics": [],
    "sectors": [],
    "amount": null,
    "rate": null,
    "period_years": null
  }},
  "confidence": 0.95,
  "safety_verdict": "safe",
  "reasoning": "why this classification"
}}"""
        
        return prompt
    
    def _build_context(self, 
                       query: str, 
                       conversation_history: List[ConversationTurn]) -> str:
        """Build context string including recent conversation turns"""
        
        if not conversation_history:
            return query
        
        # Include last N turns for context (config.SESSION_CONTEXT_TURNS)
        recent_turns = conversation_history[-config.SESSION_CONTEXT_TURNS:]
        
        context_parts = ["Recent conversation:"]
        for turn in recent_turns:
            context_parts.append(f"User: {turn.query}")
            context_parts.append(f"Agent used: {turn.agent}")
        
        context_parts.append(f"\nCurrent query: {query}")
        
        return "\n".join(context_parts)
    
    async def classify(self, 
                       query: str,
                       conversation_history: Optional[List[ConversationTurn]] = None) -> ClassificationResult:
        """
        Classify the query and extract entities
        
        Args:
            query: User's query text
            conversation_history: Previous conversation turns for context
            
        Returns:
            ClassificationResult with intent, agent, entities, etc.
        """
        start_time = time.perf_counter()
        
        try:
            # Build context with conversation history
            context_query = self._build_context(
                query, 
                conversation_history or []
            )
            
            # Make OpenAI API call with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": context_query}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # low temp for consistency
                max_tokens=500  
            )
            
            # Parse the JSON response
            result_json = json.loads(response.choices[0].message.content)
            
            # Convert to our Pydantic model
            entities = ExtractedEntities(**result_json.get("entities", {}))
            
            classification = ClassificationResult(
                intent=result_json.get("intent", "unknown"),
                agent=result_json.get("agent", "support"),
                entities=entities,
                confidence=result_json.get("confidence", 0.5),
                safety_verdict=result_json.get("safety_verdict", "safe"),
                reasoning=result_json.get("reasoning")
            )
            
            elapsed = time.perf_counter() - start_time
            
            # Track tokens and cost
            usage = response.usage
            self._log_usage(usage.prompt_tokens, usage.completion_tokens)
            
            return classification
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"[Classifier] ERROR after {elapsed:.2f}s: {e}")
            
            # Return a fallback classification
            return ClassificationResult(
                intent="error_fallback",
                agent="support",
                entities=ExtractedEntities(),
                confidence=0.0,
                safety_verdict="safe",
                reasoning=f"Classification failed: {str(e)}"
            )
    
    def _log_usage(self, input_tokens: int, output_tokens: int):
        """Log token usage and cost (for metrics tracking)"""
        
        # Calculate cost based on model
        if "gpt-4.1" in self.model.lower():
            cost = (input_tokens * config.GPT4_1_INPUT_COST + 
                   output_tokens * config.GPT4_1_OUTPUT_COST)
        else:  # gpt-4o-mini
            cost = (input_tokens * config.GPT4O_MINI_INPUT_COST + 
                   output_tokens * config.GPT4O_MINI_OUTPUT_COST)
        
        if config.DEBUG:
            print(f"[Classifier] Tokens: {input_tokens} in, {output_tokens} out | Cost: ${cost:.6f}")


# Quick test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        classifier = IntentClassifier()
        
        test_queries = [
            "How is my portfolio doing?",
            "Tell me about Apple stock",
            "Should I invest $10000 in tech stocks?",
            "Calculate compound interest on $5000 at 7% for 10 years"
        ]
        
        print("Classifier Test:--")
        
        for q in test_queries:
            print(f"\nQuery: {q}")
            result = await classifier.classify(q)
            print(f"Agent: {result.agent}")
            print(f"Intent: {result.intent}")
            print(f"Entities: {result.entities.model_dump()}")
            print(f"Confidence: {result.confidence}")
    
    # Run test
    asyncio.run(test())