"""
Portfolio Health Agent - fully implemented specialist agent

Analyzes portfolio composition, risk, performance, and provides actionable insights
This is the agent a novice investor hits when asking "how is my portfolio doing?"
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf

from .base import BaseAgent
from ..models import (
    ClassificationResult, UserContext, PortfolioHealthResponse,
    ConcentrationRisk, Performance, BenchmarkComparison, Observation,
    PortfolioHolding
)
from ..config import config


class PortfolioHealthAgent(BaseAgent):
    """
    Analyzes portfolio health across multiple dimensions:
    - Concentration risk
    - Performance metrics
    - Benchmark comparison
    - Actionable observations
    """
    
    DISCLAIMER = (
        "This analysis is for informational purposes only and does not constitute "
        "investment advice. Past performance does not guarantee future results. "
        "Please consult with a licensed financial advisor before making investment decisions."
    )
    
    def __init__(self):
        super().__init__()
        print(f"[{self.name}] Initialized")
    
    async def execute(self,
                     query: str,
                     classification: ClassificationResult,
                     user_context: UserContext) -> PortfolioHealthResponse:
        """
        Execute portfolio health analysis
        """
        
        # Validate inputs
        self._validate_inputs(query, classification, user_context)
        
        portfolio = user_context.portfolio
        
        # Handle empty portfolio case
        if not portfolio or len(portfolio) == 0:
            return self._handle_empty_portfolio(user_context)
        
        print(f"[{self.name}] Analyzing portfolio with {len(portfolio)} positions")
        
        # Fetch current market data for all holdings
        market_data = await self._fetch_market_data(portfolio)
        
        # Calculate portfolio metrics
        concentration = self._calculate_concentration(portfolio, market_data)
        performance = self._calculate_performance(portfolio, market_data)
        
        # Detect appropriate benchmark and compare
        benchmark_name = self._detect_benchmark(portfolio)
        benchmark_comp = await self._compare_to_benchmark(
            portfolio, 
            market_data, 
            benchmark_name,
            performance
        )
        
        # Generate observations using analysis
        observations = self._generate_observations(
            concentration,
            performance,
            benchmark_comp,
            portfolio,
            market_data
        )
        
        return PortfolioHealthResponse(
            concentration_risk=concentration,
            performance=performance,
            benchmark_comparison=benchmark_comp,
            observations=observations,
            disclaimer=self.DISCLAIMER,
            is_empty=False
        )
    
    def _handle_empty_portfolio(self, user_context: UserContext) -> PortfolioHealthResponse:
        """
        Handle case where user has no portfolio yet
        Guide them toward BUILD mode
        """
        
        risk_profile = user_context.profile.risk_profile
        
        message = (
            f"You don't have any holdings yet. As a {risk_profile} investor, "
            f"let's start building your portfolio. Consider starting with your "
            f"investment goals and time horizon to determine the right asset allocation."
        )
        
        observations = [
            Observation(
                severity="info",
                text="No portfolio holdings detected. Ready to start investing?"
            ),
            Observation(
                severity="info",
                text=f"Risk profile: {risk_profile}. This will guide your asset selection."
            )
        ]
        
        return PortfolioHealthResponse(
            observations=observations,
            disclaimer=self.DISCLAIMER,
            is_empty=True,
            build_mode_message=message
        )
    
    async def _fetch_market_data(self, portfolio: List[PortfolioHolding]) -> Dict:
        """
        Fetch current market data for all tickers
        Uses yfinance library
        """
        
        tickers = [holding.ticker for holding in portfolio]
        market_data = {}
        
        # Fetch data in parallel for speed
        async def fetch_ticker_data(ticker: str):
            try:
                # Run yfinance in thread pool since it's blocking
                loop = asyncio.get_event_loop()
                stock = await loop.run_in_executor(None, yf.Ticker, ticker)
                info = await loop.run_in_executor(None, lambda: stock.info)
                history = await loop.run_in_executor(
                    None, 
                    lambda: stock.history(period="1y")
                )
                
                market_data[ticker] = {
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice", 0),
                    "sector": info.get("sector", "Unknown"),
                    "market": info.get("exchange", "Unknown"),
                    "history": history,
                    "info": info
                }
                
            except Exception as e:
                print(f"[{self.name}] Error fetching {ticker}: {e}")
                # Fallback data
                market_data[ticker] = {
                    "current_price": 0,
                    "sector": "Unknown",
                    "market": "Unknown",
                    "history": None,
                    "info": {}
                }
        
        # Fetch all tickers concurrently
        await asyncio.gather(*[fetch_ticker_data(t) for t in tickers])
        
        return market_data
    
    def _calculate_concentration(self, 
                                 portfolio: List[PortfolioHolding],
                                 market_data: Dict) -> ConcentrationRisk:
        """
        Calculate concentration risk metrics
        """
        
        # Calculate position values
        position_values = []
        for holding in portfolio:
            current_price = market_data.get(holding.ticker, {}).get("current_price", 0)
            value = holding.quantity * current_price
            position_values.append((holding.ticker, value))
        
        # Sort by value descending
        position_values.sort(key=lambda x: x[1], reverse=True)
        
        total_value = sum(v for _, v in position_values)
        
        if total_value == 0:
            # Avoid division by zero
            return ConcentrationRisk(
                top_position_pct=0,
                top_3_positions_pct=0,
                flag="low"
            )
        
        # Top position percentage
        top_pct = (position_values[0][1] / total_value) * 100
        
        # Top 3 positions percentage
        top_3_value = sum(v for _, v in position_values[:3])
        top_3_pct = (top_3_value / total_value) * 100
        
        # Determine flag
        if top_pct >= 50:
            flag = "extreme"
        elif top_pct >= 30:
            flag = "high"
        elif top_pct >= 20:
            flag = "medium"
        else:
            flag = "low"
        
        return ConcentrationRisk(
            top_position_pct=round(top_pct, 2),
            top_3_positions_pct=round(top_3_pct, 2),
            flag=flag
        )
    
    def _calculate_performance(self,
                              portfolio: List[PortfolioHolding],
                              market_data: Dict) -> Performance:
        """
        Calculate portfolio performance metrics
        """
        
        total_cost = 0
        total_current_value = 0
        
        for holding in portfolio:
            current_price = market_data.get(holding.ticker, {}).get("current_price", 0)
            purchase_price = holding.purchase_price or current_price  # fallback
            
            cost = holding.quantity * purchase_price
            current_value = holding.quantity * current_price
            
            total_cost += cost
            total_current_value += current_value
        
        if total_cost == 0:
            return Performance(
                total_return_pct=0,
                annualized_return_pct=0,
                time_period="N/A"
            )
        
        # Calculate total return
        total_return_pct = ((total_current_value - total_cost) / total_cost) * 100
        
        # TODO: calculate annualized return properly using purchase dates
        # For now, just return total return
        # In production, would use actual time-weighted returns
        
        return Performance(
            total_return_pct=round(total_return_pct, 2),
            annualized_return_pct=None,  # would need purchase dates
            time_period="Since purchase"
        )
    
    def _detect_benchmark(self, portfolio: List[PortfolioHolding]) -> str:
        """
        Detect appropriate benchmark based on portfolio composition
        
        This is a heuristic - in production would be more sophisticated
        """
        
        # Count US vs international holdings based on ticker patterns
        us_count = 0
        total = len(portfolio)
        
        for holding in portfolio:
            ticker = holding.ticker.upper()
            # Heuristic: US tickers typically don't have exchange suffix
            # International often have .L, .AS, .TO, etc.
            if '.' not in ticker:
                us_count += 1
        
        us_ratio = us_count / total if total > 0 else 0
        
        # Decision logic
        if us_ratio > 0.7:
            return "SPY"  # S&P 500 ETF
        elif us_ratio < 0.3:
            return "EFA"  # MSCI EAFE (international developed markets)
        else:
            return "VT"  # Total world stock market
    
    async def _compare_to_benchmark(self,
                                    portfolio: List[PortfolioHolding],
                                    market_data: Dict,
                                    benchmark_ticker: str,
                                    portfolio_performance: Performance) -> BenchmarkComparison:
        """
        Compare portfolio performance to benchmark
        """
        
        try:
            # Fetch benchmark data
            loop = asyncio.get_event_loop()
            benchmark = await loop.run_in_executor(None, yf.Ticker, benchmark_ticker)
            bench_history = await loop.run_in_executor(
                None,
                lambda: benchmark.history(period="1y")
            )
            
            if bench_history is None or len(bench_history) == 0:
                raise ValueError("No benchmark data")
            
            # Calculate benchmark return
            start_price = bench_history['Close'].iloc[0]
            end_price = bench_history['Close'].iloc[-1]
            bench_return_pct = ((end_price - start_price) / start_price) * 100
            
            # Calculate alpha (outperformance)
            alpha = portfolio_performance.total_return_pct - bench_return_pct
            
            # Map ticker to friendly name
            benchmark_names = {
                "SPY": "S&P 500",
                "EFA": "MSCI EAFE",
                "VT": "Total World Stock Market"
            }
            
            return BenchmarkComparison(
                benchmark=benchmark_names.get(benchmark_ticker, benchmark_ticker),
                portfolio_return_pct=portfolio_performance.total_return_pct,
                benchmark_return_pct=round(bench_return_pct, 2),
                alpha_pct=round(alpha, 2)
            )
            
        except Exception as e:
            print(f"[{self.name}] Benchmark comparison failed: {e}")
            # Return fallback comparison
            return BenchmarkComparison(
                benchmark="S&P 500",
                portfolio_return_pct=portfolio_performance.total_return_pct,
                benchmark_return_pct=0,
                alpha_pct=0
            )
    
    def _generate_observations(self,
                              concentration: ConcentrationRisk,
                              performance: Performance,
                              benchmark_comp: BenchmarkComparison,
                              portfolio: List[PortfolioHolding],
                              market_data: Dict) -> List[Observation]:
        """
        Generate plain-language observations for the user
        
        Focus on 2-3 most important insights
        """
        
        observations = []
        
        # Concentration observations
        if concentration.flag in ["extreme", "high"]:
            top_ticker = max(
                portfolio,
                key=lambda h: h.quantity * market_data.get(h.ticker, {}).get("current_price", 0)
            ).ticker
            
            observations.append(Observation(
                severity="warning" if concentration.flag == "high" else "critical",
                text=f"{concentration.top_position_pct}% of your portfolio is in {top_ticker}. "
                     f"This high concentration increases risk significantly."
            ))
        
        # Performance observations
        if performance.total_return_pct > 0:
            observations.append(Observation(
                severity="info",
                text=f"Portfolio is up {performance.total_return_pct}% since purchase."
            ))
        else:
            observations.append(Observation(
                severity="warning",
                text=f"Portfolio is down {abs(performance.total_return_pct)}% since purchase."
            ))
        
        # Benchmark comparison
        if benchmark_comp.alpha_pct > 5:
            observations.append(Observation(
                severity="info",
                text=f"Outperforming {benchmark_comp.benchmark} by {benchmark_comp.alpha_pct}%."
            ))
        elif benchmark_comp.alpha_pct < -5:
            observations.append(Observation(
                severity="warning",
                text=f"Underperforming {benchmark_comp.benchmark} by {abs(benchmark_comp.alpha_pct)}%."
            ))
        
        # Sector concentration check
        sectors = {}
        for holding in portfolio:
            sector = market_data.get(holding.ticker, {}).get("sector", "Unknown")
            if sector not in sectors:
                sectors[sector] = 0
            current_price = market_data.get(holding.ticker, {}).get("current_price", 0)
            sectors[sector] += holding.quantity * current_price
        
        if len(sectors) <= 2 and len(portfolio) > 2:
            observations.append(Observation(
                severity="warning",
                text="Portfolio is concentrated in just a few sectors. Consider diversifying."
            ))
        
        # Limit to top 3-4 most important observations
        return observations[:4]


# Quick test
if __name__ == "__main__":
    import asyncio
    from ..models import UserProfile
    
    async def test():
        agent = PortfolioHealthAgent()
        
        # Mock data
        user_context = UserContext(
            user_id="test",
            session_id="test",
            profile=UserProfile(
                user_id="test",
                risk_profile="moderate"
            ),
            portfolio=[
                PortfolioHolding(ticker="AAPL", quantity=10, purchase_price=150),
                PortfolioHolding(ticker="MSFT", quantity=5, purchase_price=300)
            ]
        )
        
        classification = ClassificationResult(
            intent="portfolio health",
            agent="portfolio_health",
            entities={},
            confidence=0.95
        )
        
        print("Testing Portfolio Health Agent:")
        print("=" * 60)
        
        result = await agent.execute(
            query="How is my portfolio doing?",
            classification=classification,
            user_context=user_context
        )
        
        print(f"Concentration: {result.concentration_risk}")
        print(f"Performance: {result.performance}")
        print(f"Observations: {len(result.observations)}")
        
        for obs in result.observations:
            print(f"  [{obs.severity}] {obs.text}")
    
    asyncio.run(test())