import time
import json
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from ..config import config


class LLMClient:
    """
    Wrapper around OpenAI client with retry logic and metrics
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM client
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: Model name (defaults to config)
        """
        
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.backoff_factor = 2.0
        
        # Metrics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def completion(self,
                        messages: List[Dict[str, str]],
                        temperature: float = 0.1,
                        max_tokens: int = 500,
                        json_mode: bool = False) -> Dict[str, Any]:
        """
        Call OpenAI completion API with retry logic
        
        Args:
            messages: List of message dicts with role and content
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            json_mode: Use JSON mode for structured outputs
            
        Returns:
            Response dict with content, tokens, cost, etc.
        """
        
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                # Build request params
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                
                # Make API call
                start_time = time.perf_counter()
                
                response = self.client.chat.completions.create(**params)
                
                latency = time.perf_counter() - start_time
                
                # Extract response
                content = response.choices[0].message.content
                
                # Track metrics
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                cost = self._calculate_cost(input_tokens, output_tokens)
                
                self.total_calls += 1
                self.total_tokens += total_tokens
                self.total_cost += cost
                
                # Return structured response
                return {
                    "content": content,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "latency": latency,
                    "model": self.model
                }
                
            except RateLimitError as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** (retries - 1))
                    print(f"[LLMClient] Rate limit hit, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Rate limit exceeded after {self.max_retries} retries") from e
            
            except APIConnectionError as e:
                last_error = e
                retries += 1
                if retries < self.max_retries:
                    delay = self.retry_delay * (self.backoff_factor ** (retries - 1))
                    print(f"[LLMClient] Connection error, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Connection failed after {self.max_retries} retries") from e
            
            except APIError as e:
                # Don't retry on other API errors
                raise Exception(f"OpenAI API error: {str(e)}") from e
            
            except Exception as e:
                # Unexpected error
                raise Exception(f"Unexpected LLM error: {str(e)}") from e
        
        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise Exception("LLM call failed for unknown reason")
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost based on model and token counts
        """
        
        if "gpt-4.1" in self.model.lower():
            cost = (input_tokens * config.GPT4_1_INPUT_COST + 
                   output_tokens * config.GPT4_1_OUTPUT_COST)
        elif "gpt-4o-mini" in self.model.lower():
            cost = (input_tokens * config.GPT4O_MINI_INPUT_COST + 
                   output_tokens * config.GPT4O_MINI_OUTPUT_COST)
        else:
            # Default to gpt-4o-mini pricing
            cost = (input_tokens * config.GPT4O_MINI_INPUT_COST + 
                   output_tokens * config.GPT4O_MINI_OUTPUT_COST)
        
        return cost
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics
        """
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "avg_tokens_per_call": round(self.total_tokens / self.total_calls, 1) if self.total_calls > 0 else 0,
            "avg_cost_per_call": round(self.total_cost / self.total_calls, 6) if self.total_calls > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0


# Test
if __name__ == "__main__":
    import asyncio
    
    async def test():
        client = LLMClient()
        
        print("LLM Client Test:--")
        
        # Test completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in JSON format with a 'message' field."}
        ]
        
        try:
            response = await client.completion(messages, json_mode=True)
            print(f"Response: {response['content']}")
            print(f"Tokens: {response['total_tokens']}")
            print(f"Cost: ${response['cost']:.6f}")
            print(f"Latency: {response['latency']:.3f}s")
            
            # Get metrics
            metrics = client.get_metrics()
            print(f"\nMetrics: {metrics}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    asyncio.run(test())