import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL_DEV: str = os.getenv("OPENAI_MODEL_DEV", "gpt-4o-mini")  # cheaper for dev
    OPENAI_MODEL_PROD: str = os.getenv("OPENAI_MODEL_PROD", "gpt-4.1")
    
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEV)
    
    # Performance targets
    MAX_CLASSIFICATION_TIME: float = 2.0  # seconds
    MAX_AGENT_TIME: float = 6.0  # seconds
    MAX_TOTAL_TIME: float = 8.0  # seconds - bit of buffer

    SAFETY_GUARD_MAX_TIME: float = 0.01  # 10ms max
    
    DB_PATH: str = os.getenv("DB_PATH", "./data/sessions.db")

    SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "10"))
    SESSION_CONTEXT_TURNS: int = 3  # how many prev turns to include in context
    
    # Cost tracking in USD per 1M tokens for gpt-4.1
    GPT4_1_INPUT_COST: float = 0.15 / 1_000_000
    GPT4_1_OUTPUT_COST: float = 0.60 / 1_000_000
    GPT4O_MINI_INPUT_COST: float = 0.015 / 1_000_000  # way cheaper
    GPT4O_MINI_OUTPUT_COST: float = 0.06 / 1_000_000

    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Check if required config is present"""
        if not cls.OPENAI_API_KEY and cls.ENV == "production":
            raise ValueError("OPENAI_API_KEY is required in production")
        
        # TODO: add more validation as needed
        return True


# Agent taxonomy - defines all available agents
# NOTE: Only portfolio_health is fully implemented for now
AGENT_TAXONOMY = {
    "portfolio_health": "Analyzes portfolio composition, risk, and performance",
    "market_research": "Provides market data and company research",
    "investment_strategy": "Helps build investment plans and strategies",
    "financial_calculator": "Performs financial calculations",
    "risk_assessment": "Evaluates investment risks",
    "recommendation": "Suggests specific investment actions",
    "support": "General support and educational queries"
}

# Implemented agents only these will execute, rest are stubs
IMPLEMENTED_AGENTS = ["portfolio_health"]


config = Config()

if __name__ == "__main__":
    config.validate()
    print(f"Config loaded successfully")
    print(f"Model: {config.OPENAI_MODEL}")
    print(f"Environment: {config.ENV}")