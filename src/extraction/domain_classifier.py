"""Domain classification for terminology extraction."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger
import asyncio # Added for classify_batch example

# FIXED: Removed termextractor. prefix
from api.api_manager import APIManager
# FIXED: Removed termextractor. prefix
from utils.constants import DOMAIN_CATEGORIES
# FIXED: Removed termextractor. prefix
from utils.helpers import parse_domain_path, chunk_list # Added chunk_list


@dataclass
class DomainResult:
    """Domain classification result."""
    # ... (class content seems okay) ...
    hierarchy: List[str]
    confidence: float
    alternative_domains: List[Dict[str, Any]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hierarchy": self.hierarchy,
            "confidence": self.confidence,
            "alternative_domains": self.alternative_domains,
            "keywords": self.keywords,
            "reasoning": self.reasoning,
        }

    @property
    def primary_domain(self) -> str:
        """Get primary domain."""
        return self.hierarchy[0] if self.hierarchy else "General"

    @property
    def full_path(self) -> str:
        """Get full domain path as string."""
        return " → ".join(self.hierarchy)


class DomainClassifier:
    """Hierarchical domain classifier."""
    def __init__(
        self,
        api_client: APIManager,
        max_depth: int = 5,
        confidence_threshold: float = 0.7,
    ):
        """Initialize DomainClassifier."""
        self.api_client = api_client
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        logger.info("DomainClassifier initialized")

    async def classify(
        self,
        text: str,
        suggested_domains: Optional[List[str]] = None,
        user_domain_path: Optional[str] = None,
    ) -> DomainResult:
        """Classify text into domain hierarchy."""
        if user_domain_path:
            hierarchy = parse_domain_path(user_domain_path)
            logger.info(f"Using user-specified domain: {hierarchy}")
            return DomainResult(hierarchy=hierarchy, confidence=1.0, reasoning="User-specified domain")

        logger.info("Classifying domain using AI")
        try:
             # Assuming classify_domain exists on the client (needs implementation there)
            result = await self.api_client.client.classify_domain(
                text=text,
                suggested_domains=suggested_domains or self._get_suggested_domains(),
            )
            hierarchy = result.get("domain_hierarchy", ["General"])
            confidence = result.get("confidence", 0.0) / 100.0
        except AttributeError:
             logger.error("API client does not have a 'classify_domain' method. Returning default.")
             hierarchy = ["General"]
             confidence = 0.0
             result = {} # Ensure result is defined
        except Exception as e:
            logger.error(f"Error during AI domain classification: {e}")
            hierarchy = ["General"]
            confidence = 0.0
            result = {} # Ensure result is defined


        if len(hierarchy) > self.max_depth:
            hierarchy = hierarchy[: self.max_depth]

        domain_result = DomainResult(
            hierarchy=hierarchy,
            confidence=confidence,
            alternative_domains=result.get("alternative_domains", []),
            keywords=result.get("keywords", []),
            reasoning=result.get("reasoning", ""),
        )
        logger.info(f"Domain classified: {domain_result.full_path} (confidence: {confidence:.2f})")
        return domain_result

    def _get_suggested_domains(self) -> List[str]:
        """Get list of suggested domains from constants."""
        return list(DOMAIN_CATEGORIES.keys())

    # ... (rest of the methods like validate_domain_path, suggest_subdomains) ...

    async def classify_batch(self, texts: List[str], batch_size: int = 10) -> List[DomainResult]:
        """Classify multiple texts."""
        # FIXED: Removed internal import of chunk_list, already imported at top
        logger.info(f"Batch classifying {len(texts)} texts")
        results = []
        text_chunks = chunk_list(texts, batch_size) # Uses imported chunk_list
        for chunk in text_chunks:
            chunk_results = await asyncio.gather(*[self.classify(text) for text in chunk])
            results.extend(chunk_results)
        return results

    # ... (filter_by_domain method) ...
