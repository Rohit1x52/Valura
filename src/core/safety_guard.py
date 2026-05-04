import re
import time
from typing import Optional, Tuple
from ..models import SafetyResult


class SafetyGuard:
    
    def __init__(self):
        # Compile regex patterns once for performance
        self._patterns = self._build_patterns()
        
        # Category-specific professional responses
        self._category_responses = {
            "insider_trading": (
                "I cannot assist with insider trading or the use of material non-public information. "
                "These activities violate securities laws and can result in serious legal consequences."
            ),
            "market_manipulation": (
                "I cannot help with market manipulation schemes such as pump-and-dump operations or "
                "coordinated trading designed to mislead other investors. These activities are illegal."
            ),
            "guaranteed_returns": (
                "I cannot make claims about guaranteed returns or risk-free investments. "
                "All investments carry risk, and past performance does not guarantee future results."
            ),
            "money_laundering": (
                "I cannot provide assistance with money laundering, tax evasion, or hiding funds from "
                "regulatory authorities. These are serious financial crimes."
            ),
            "reckless_advice": (
                "I cannot recommend highly risky strategies without proper risk assessment and disclosure. "
                "Please consult with a licensed financial advisor for personalized investment advice."
            )
        }
    
    def _build_patterns(self) -> dict:
        """Build compiled regex patterns for each harmful category"""
        
        patterns = {
            "insider_trading": [
                r'\binsider\s+(?:info|information|tip|knowledge)',
                r'\bnon[- ]?public\s+(?:info|information)',
                r'\bmaterial\s+non[- ]?public',
                r'\btrade\s+(?:on|using)\s+(?:inside|insider)',
                r'\bmnpi\b',  # material non-public information acronym
            ],
            
            "market_manipulation": [
                r'\bpump\s+and\s+dump\b',
                r'\bpump[- ]?and[- ]?dump\b',
                r'\bmanipulate\s+(?:the\s+)?(?:market|price|stock)',
                r'\bcoordinat(?:e|ed|ing)\s+(?:buying|selling|trades)',
                r'\bpenny\s+stock\s+(?:scheme|scam)',
                r'\bartificially\s+(?:inflate|deflate)',
            ],
            
            "guaranteed_returns": [
                r'\bguaranteed?\s+(?:return|profit|gain)',
                r'\brisk[- ]?free\s+(?:return|profit|investment)',
                r'\bcan[\'']?t\s+lose\b',
                r'\bno\s+risk\b',
                r'\b100%\s+(?:safe|guaranteed|certain)',
                r'\bzero\s+risk\b',
            ],
            
            "money_laundering": [
                r'\blaunder(?:ing)?\s+(?:money|funds)',
                r'\bhide\s+(?:money|funds|assets).*(?:from|avoid).*(?:tax|irs|authorities)',
                r'\boffshore.*(?:hide|avoid).*tax',
                r'\bevade\s+tax(?:es)?',
                r'\bunreported\s+(?:income|gains)',
            ],
            
            "reckless_advice": [
                r'\ball[- ]?in\s+on\s+\w+',  # "go all-in on TSLA"
                r'\b(?:max|maximum)\s+leverage\b',
                r'\bborrow\s+(?:as\s+much|everything|all)',
                r'\bmortgage.*house.*invest',  # extremely risky
                r'\bcrypto.*leverage.*\d+x',
            ]
        }
        
        # Compile all patterns for performance
        compiled = {}
        for category, pattern_list in patterns.items():
            compiled[category] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in pattern_list
            ]
        
        return compiled
    
    def _is_educational_query(self, query: str) -> bool:
        """
        Detect if this is an educational question vs intent to act
        Educational queries should PASS even if they mention harmful topics
        """
        
        # Common educational question patterns
        educational_indicators = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+are\b',
            r'\bexplain\b',
            r'\bdefine\b',
            r'\bdefin(?:e|ition)\s+(?:of\s+)?',
            r'\bhow\s+does.*work\b',
            r'\bwhy\s+is\b',
            r'\bcan\s+you\s+(?:explain|tell|teach)',
            r'\blearn\s+about\b',
            r'\bunders(?:tand|tanding)\b',
        ]
        
        for pattern in educational_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        # Check for negation/prohibition context
        # "Why shouldn't I__", "Don't do X because__"
        negation_patterns = [
            r'\bshould(?:n[\'']t|n\'t| not)\b',
            r'\bdo(?:n[\'']t|n\'t| not)\b',
            r'\bwhy.*(?:illegal|wrong|bad)\b',
            r'\bconsequences?\s+of\b',
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def check(self, query: str) -> SafetyResult:
        """
        Check if query is safe to proceed
        Returns:
            SafetyResult with is_safe=True if query passes
            SafetyResult with is_safe=False and category-specific message if blocked
        """
        start_time = time.perf_counter()
        
        # First check: is this educational?
        if self._is_educational_query(query):
            elapsed = time.perf_counter() - start_time
            return SafetyResult(is_safe=True)
        
        for category, patterns in self._patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    elapsed = time.perf_counter() - start_time
                    
                    return SafetyResult(
                        is_safe=False,
                        blocked_category=category,
                        response_message=self._category_responses[category]
                    )
        
        elapsed = time.perf_counter() - start_time
        
        return SafetyResult(is_safe=True)


if __name__ == "__main__":
    guard = SafetyGuard()
    
    test_queries = [
        ("What is insider trading?", True),  
        ("How can I use insider information to make money?", False),  
        ("Tell me about pump and dump schemes", True),  
        ("Help me pump this penny stock", False),  
        ("This investment has guaranteed returns", False),  
        ("What are the risks of investing?", True),  
        ("How do I launder money?", False),  
        ("What is money laundering?", True),  
        ("Go all-in on TSLA with max leverage", False),  
        ("What does 'going all-in' mean?", True),
    ]
    
    print("Safety Guard Test:--")
    
    for query, expected_safe in test_queries:
        result = guard.check(query)
        status = " PASS" if result.is_safe == expected_safe else "✗ FAIL"
        print(f"{status} | Safe={result.is_safe} | {query[:50]}")
        if not result.is_safe:
            print(f"       Category: {result.blocked_category}")