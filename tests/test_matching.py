import pytest
from src.models import ExtractedEntities, normalize_ticker, compare_entities


class TestEntityMatching:
    """Test suite for entity matching and normalization"""
    
    def test_normalize_ticker_uppercase(self):
        """Should convert tickers to uppercase"""
        assert normalize_ticker("aapl") == "AAPL"
        assert normalize_ticker("msft") == "MSFT"
        assert normalize_ticker("GoOgL") == "GOOGL"
    
    def test_normalize_ticker_removes_exchange_suffix(self):
        """Should remove exchange suffix (.AS, .L, etc.)"""
        assert normalize_ticker("ASML.AS") == "ASML"
        assert normalize_ticker("HSBC.L") == "HSBC"
        assert normalize_ticker("RY.TO") == "RY"
    
    def test_normalize_ticker_handles_no_suffix(self):
        """Should handle tickers without exchange suffix"""
        assert normalize_ticker("AAPL") == "AAPL"
        assert normalize_ticker("MSFT") == "MSFT"

    def test_ticker_matching_case_insensitive(self):
        """Ticker matching should be case-insensitive"""
        expected = ExtractedEntities(tickers=["AAPL", "MSFT"])
        actual = ExtractedEntities(tickers=["aapl", "msft"])
        
        assert compare_entities(expected, actual)
    
    def test_ticker_matching_with_exchange_suffix(self):
        """Should match tickers with/without exchange suffix"""
        expected = ExtractedEntities(tickers=["ASML"])
        actual = ExtractedEntities(tickers=["ASML.AS"])
        
        assert compare_entities(expected, actual)
    
    def test_ticker_matching_subset(self):
        """Should pass if expected tickers are subset of actual"""
        expected = ExtractedEntities(tickers=["AAPL", "MSFT"])
        actual = ExtractedEntities(tickers=["AAPL", "MSFT", "GOOGL"])
        
        assert compare_entities(expected, actual)
    
    def test_ticker_matching_fails_if_missing(self):
        """Should fail if expected ticker is missing"""
        expected = ExtractedEntities(tickers=["AAPL", "MSFT", "GOOGL"])
        actual = ExtractedEntities(tickers=["AAPL", "MSFT"])
        
        assert not compare_entities(expected, actual)
    
    def test_amount_matching_exact(self):
        """Should match exact amounts"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=10000)
        
        assert compare_entities(expected, actual)
    
    def test_amount_matching_within_tolerance(self):
        """Should match amounts within ±5% tolerance"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=10400)  # 4% difference
        
        assert compare_entities(expected, actual)
    
    def test_amount_matching_fails_outside_tolerance(self):
        """Should fail if amount differs by more than 5%"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=11000)  # 10% difference
        
        assert not compare_entities(expected, actual)
    
    def test_amount_matching_edge_case_5_percent(self):
        """Should pass at exactly 5% tolerance boundary"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=10500)  # Exactly 5%
        
        # At boundary - implementation dependent
        # Our implementation uses <= tolerance, so this should pass
        assert compare_entities(expected, actual)
    
    def test_rate_matching_within_tolerance(self):
        """Should match rate field within tolerance"""
        expected = ExtractedEntities(rate=7.0)
        actual = ExtractedEntities(rate=7.3)  # ~4.3% difference
        
        assert compare_entities(expected, actual)
    
    def test_period_years_matching(self):
        """Should match period_years field within tolerance"""
        expected = ExtractedEntities(period_years=10)
        actual = ExtractedEntities(period_years=10)
        
        assert compare_entities(expected, actual)
    
    def test_combined_entity_matching(self):
        """Should match entities with multiple fields"""
        expected = ExtractedEntities(
            tickers=["AAPL"],
            amount=10000,
            rate=7.0
        )
        actual = ExtractedEntities(
            tickers=["aapl"],
            amount=10200,  # Within tolerance
            rate=7.2       # Within tolerance
        )
        
        assert compare_entities(expected, actual)
    
    def test_combined_entity_fails_if_one_field_wrong(self):
        """Should fail if any required field doesn't match"""
        expected = ExtractedEntities(
            tickers=["AAPL"],
            amount=10000
        )
        actual = ExtractedEntities(
            tickers=["MSFT"],  # Wrong ticker
            amount=10000
        )
        
        assert not compare_entities(expected, actual)
    
    def test_empty_entities_match(self):
        """Empty entity sets should match"""
        expected = ExtractedEntities()
        actual = ExtractedEntities()
        
        assert compare_entities(expected, actual)
    
    def test_none_values_handling(self):
        """Should handle None values correctly"""
        expected = ExtractedEntities(amount=None)
        actual = ExtractedEntities(amount=10000)
        
        # If expected is None, we don't check that field
        assert compare_entities(expected, actual)
    
    def test_actual_none_when_expected_has_value(self):
        """Should fail if expected has value but actual is None"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=None)
        
        assert not compare_entities(expected, actual)
    
    def test_custom_tolerance_parameter(self):
        """Should respect custom tolerance parameter"""
        expected = ExtractedEntities(amount=10000)
        actual = ExtractedEntities(amount=11000)  # 10% difference
        
        # Default 5% tolerance - should fail
        assert not compare_entities(expected, actual, tolerance=0.05)
        
        # 15% tolerance - should pass
        assert compare_entities(expected, actual, tolerance=0.15)
    
    def test_fixture_like_example_1(self):
        """Test realistic fixture-like scenario"""
        # Query: "Should I invest $10,000 in AAPL and MSFT?"
        expected = ExtractedEntities(
            tickers=["AAPL", "MSFT"],
            amount=10000
        )
        
        # Classifier returns:
        actual = ExtractedEntities(
            tickers=["AAPL", "MSFT", "tech"],  # Extra items OK
            amounts=[5000, 5000],  # Split amounts
            amount=10000
        )
        
        assert compare_entities(expected, actual)
    
    def test_fixture_like_example_2(self):
        """Test with international ticker"""
        # Query: "What do you think about ASML?"
        expected = ExtractedEntities(
            tickers=["ASML"]
        )
        
        # Classifier might return with exchange suffix
        actual = ExtractedEntities(
            tickers=["ASML.AS"]
        )
        
        assert compare_entities(expected, actual)
    
    def test_fixture_like_example_3(self):
        """Test compound interest calculation query"""
        # Query: "Calculate interest on $5000 at 7% for 10 years"
        expected = ExtractedEntities(
            amount=5000,
            rate=7.0,
            period_years=10
        )
        
        # Classifier extracts:
        actual = ExtractedEntities(
            amount=5000,
            rate=7.0,
            period_years=10
        )
        
        assert compare_entities(expected, actual)


class TestEntityMatchingHelpers:
    """Test helper functions for entity matching"""
    
    def test_normalize_ticker_list(self, normalize_ticker):
        """Test normalizing a list of tickers"""
        tickers = ["aapl", "MSFT", "GOOGL.L", "asml.as"]
        normalized = [normalize_ticker(t) for t in tickers]
        
        assert normalized == ["AAPL", "MSFT", "GOOGL", "ASML"]
    
    def test_entity_matcher_fixture(self, entity_matcher):
        """Test entity matcher from conftest fixture"""
        expected = {
            "tickers": ["AAPL", "MSFT"],
            "amount": 10000
        }
        
        actual = {
            "tickers": ["aapl", "msft", "googl"],
            "amount": 10200
        }
        
        assert entity_matcher(expected, actual)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])