import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    query_id: str
    timestamp: str
    
    # Latencies
    safety_guard_time: float
    classification_time: float
    agent_time: float
    total_time: float
    
    # Tokens
    input_tokens: int
    output_tokens: int
    total_tokens: int
    
    # Cost
    cost: float
    
    # Metadata
    agent: str
    intent: str
    safety_blocked: bool = False


class MetricsTracker:
    """
    Tracks and aggregates metrics across queries
    Provides methods to calculate p95, averages, etc.
    """
    
    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize metrics tracker
        
        Args:
            output_file: Path to save metrics (optional)
        """
        
        self.metrics: List[QueryMetrics] = []
        self.output_file = output_file or "./data/metrics.json"
        
        # Create output directory if needed
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"[MetricsTracker] Initialized, output: {self.output_file}")
    
    def record(self, metrics: QueryMetrics):
        """Record metrics for a query"""
        self.metrics.append(metrics)
        
        # Auto-save periodically (every 10 queries)
        if len(self.metrics) % 10 == 0:
            self._save_to_file()
    
    def calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        return float(np.percentile(values, percentile))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics
        
        Returns dict with p50, p95, p99 latencies, costs, etc.
        """
        
        if not self.metrics:
            return {
                "error": "No metrics recorded yet"
            }
        
        # Extract latency values
        total_times = [m.total_time for m in self.metrics]
        classification_times = [m.classification_time for m in self.metrics]
        agent_times = [m.agent_time for m in self.metrics]
        
        # Extract cost values
        costs = [m.cost for m in self.metrics]
        
        # Extract token values
        total_tokens = [m.total_tokens for m in self.metrics]
        
        # Calculate statistics
        summary = {
            "total_queries": len(self.metrics),
            
            # Latency stats
            "latency": {
                "total_time": {
                    "p50": round(self.calculate_percentile(total_times, 50), 3),
                    "p95": round(self.calculate_percentile(total_times, 95), 3),
                    "p99": round(self.calculate_percentile(total_times, 99), 3),
                    "mean": round(np.mean(total_times), 3),
                    "max": round(max(total_times), 3)
                },
                "classification_time": {
                    "p50": round(self.calculate_percentile(classification_times, 50), 3),
                    "p95": round(self.calculate_percentile(classification_times, 95), 3),
                    "mean": round(np.mean(classification_times), 3)
                },
                "agent_time": {
                    "p50": round(self.calculate_percentile(agent_times, 50), 3),
                    "p95": round(self.calculate_percentile(agent_times, 95), 3),
                    "mean": round(np.mean(agent_times), 3)
                }
            },
            
            # Cost stats
            "cost": {
                "p50": round(self.calculate_percentile(costs, 50), 6),
                "p95": round(self.calculate_percentile(costs, 95), 6),
                "mean": round(np.mean(costs), 6),
                "total": round(sum(costs), 6)
            },
            
            # Token stats
            "tokens": {
                "mean_per_query": round(np.mean(total_tokens), 1),
                "total": sum(total_tokens)
            },
            
            # Agent breakdown
            "by_agent": self._get_agent_breakdown()
        }
        
        return summary
    
    def _get_agent_breakdown(self) -> Dict[str, Any]:
        """Get metrics broken down by agent"""
        
        by_agent = {}
        
        for metric in self.metrics:
            agent = metric.agent
            if agent not in by_agent:
                by_agent[agent] = {
                    "count": 0,
                    "total_time": [],
                    "cost": []
                }
            
            by_agent[agent]["count"] += 1
            by_agent[agent]["total_time"].append(metric.total_time)
            by_agent[agent]["cost"].append(metric.cost)
        
        # Calculate stats for each agent
        for agent in by_agent:
            times = by_agent[agent]["total_time"]
            costs = by_agent[agent]["cost"]
            
            by_agent[agent] = {
                "count": by_agent[agent]["count"],
                "avg_time": round(np.mean(times), 3),
                "p95_time": round(self.calculate_percentile(times, 95), 3),
                "avg_cost": round(np.mean(costs), 6)
            }
        
        return by_agent
    
    def _save_to_file(self):
        """Save metrics to JSON file"""
        
        try:
            data = {
                "generated_at": datetime.now().isoformat(),
                "total_queries": len(self.metrics),
                "summary": self.get_summary(),
                "raw_metrics": [asdict(m) for m in self.metrics]
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Don't spam logs
            # print(f"[MetricsTracker] Saved {len(self.metrics)} metrics to {self.output_file}")
            
        except Exception as e:
            print(f"[MetricsTracker] Error saving metrics: {e}")
    
    def print_summary(self):
        """Print summary to console"""
        
        summary = self.get_summary()
        
        print("METRICS SUMMARY")
        
        print(f"\nTotal Queries: {summary['total_queries']}")
        
        print("\nLatency (seconds):")
        print(f"  Total Time:")
        print(f"    p50: {summary['latency']['total_time']['p50']}s")
        print(f"    p95: {summary['latency']['total_time']['p95']}s  {'ok' if summary['latency']['total_time']['p95'] < 6.0 else ' TARGET MISSED'}")
        print(f"    p99: {summary['latency']['total_time']['p99']}s")
        
        print(f"\n  Classification:")
        print(f"    p95: {summary['latency']['classification_time']['p95']}s  {'ok' if summary['latency']['classification_time']['p95'] < 2.0 else ' TARGET MISSED'}")
        
        print(f"\nCost (USD):")
        print(f"  Mean per query: ${summary['cost']['mean']:.6f}  {'ok' if summary['cost']['mean'] < 0.05 else ' TARGET MISSED'}")
        print(f"  p95 per query:  ${summary['cost']['p95']:.6f}")
        print(f"  Total:          ${summary['cost']['total']:.6f}")
        
        print(f"\nTokens:")
        print(f"  Mean per query: {summary['tokens']['mean_per_query']:.0f}")
        print(f"  Total:          {summary['tokens']['total']}")
        
        print("\nBy Agent:")
        for agent, stats in summary['by_agent'].items():
            print(f"  {agent}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg time: {stats['avg_time']}s")
            print(f"    Avg cost: ${stats['avg_cost']:.6f}")
        
    
    def check_targets(self) -> Dict[str, bool]:
        """
        Check if we're meeting performance targets
        
        Returns:
            Dict with target names and pass/fail
        """
        
        summary = self.get_summary()
        
        if not summary or "error" in summary:
            return {
                "error": "No metrics available"
            }
        
        return {
            "p95_classification_under_2s": summary['latency']['classification_time']['p95'] < 2.0,
            "p95_total_under_6s": summary['latency']['total_time']['p95'] < 6.0,
            "mean_cost_under_5cents": summary['cost']['mean'] < 0.05
        }


# Quick test
if __name__ == "__main__":
    print("Metrics Tracker Test:--")
    
    tracker = MetricsTracker(output_file="./test_metrics.json")
    
    # Simulate some queries
    for i in range(20):
        metrics = QueryMetrics(
            query_id=f"test_{i}",
            timestamp=datetime.now().isoformat(),
            safety_guard_time=0.002,
            classification_time=0.5 + np.random.random() * 0.5,
            agent_time=2.0 + np.random.random() * 2.0,
            total_time=3.0 + np.random.random() * 2.0,
            input_tokens=100 + int(np.random.random() * 50),
            output_tokens=200 + int(np.random.random() * 100),
            total_tokens=300,
            cost=0.02 + np.random.random() * 0.02,
            agent="portfolio_health",
            intent="test",
            safety_blocked=False
        )
        
        tracker.record(metrics)
    
    # Print summary
    tracker.print_summary()
    
    # Check targets
    targets = tracker.check_targets()
    print("Target Check:")
    for target, passed in targets.items():
        print(f"  {target}: {'✓ PASS' if passed else '✗ FAIL'}")
    
    # Cleanup
    import os
    if os.path.exists("./test_metrics.json"):
        os.remove("./test_metrics.json")
    print("\nTest completed")