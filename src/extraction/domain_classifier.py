"""Domain classification for terminology extraction."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from api.api_manager import APIManager
from utils.constants import DOMAIN_CATEGORIES
from utils.helpers import parse_domain_path


@dataclass
class DomainResult:
    """Domain classification result."""

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
    """
    Hierarchical domain classifier.

    Features:
    - AI-powered domain detection
    - Multi-level hierarchy (3-5 levels)
    - Confidence scoring
    - Alternative domain suggestions
    - Manual domain override
    """

    def __init__(
        self,
        api_client: APIManager,
        max_depth: int = 5,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize DomainClassifier.

        Args:
            api_client: API manager instance
            max_depth: Maximum domain hierarchy depth
            confidence_threshold: Minimum confidence threshold
        """
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
        """
        Classify text into domain hierarchy.

        Args:
            text: Text to classify
            suggested_domains: Optional suggested domains
            user_domain_path: User-specified domain path (overrides AI)

        Returns:
            DomainResult with classification
        """
        # If user provided domain path, use it directly
        if user_domain_path:
            hierarchy = parse_domain_path(user_domain_path)
            logger.info(f"Using user-specified domain: {hierarchy}")
            return DomainResult(
                hierarchy=hierarchy,
                confidence=1.0,
                reasoning="User-specified domain",
            )

        # Use AI classification
        logger.info("Classifying domain using AI")

        result = await self.api_client.client.classify_domain(
            text=text,
            suggested_domains=suggested_domains or self._get_suggested_domains(),
        )

        # Parse result
        hierarchy = result.get("domain_hierarchy", ["General"])
        confidence = result.get("confidence", 0.0) / 100.0  # Convert to 0-1

        # Truncate to max depth
        if len(hierarchy) > self.max_depth:
            hierarchy = hierarchy[: self.max_depth]

        domain_result = DomainResult(
            hierarchy=hierarchy,
            confidence=confidence,
            alternative_domains=result.get("alternative_domains", []),
            keywords=result.get("keywords", []),
            reasoning=result.get("reasoning", ""),
        )

        logger.info(
            f"Domain classified: {domain_result.full_path} "
            f"(confidence: {confidence:.2f})"
        )

        return domain_result

    def _get_suggested_domains(self) -> List[str]:
        """
        Get list of suggested domains from constants.

        Returns:
            List of domain names
        """
        return list(DOMAIN_CATEGORIES.keys())

    def validate_domain_path(self, domain_path: str) -> bool:
        """
        Validate a domain path.

        Args:
            domain_path: Domain path string

        Returns:
            True if valid
        """
        try:
            hierarchy = parse_domain_path(domain_path)
            return len(hierarchy) > 0 and len(hierarchy) <= self.max_depth
        except Exception as e:
            logger.warning(f"Invalid domain path: {e}")
            return False

    def suggest_subdomains(self, primary_domain: str) -> List[str]:
        """
        Suggest subdomains for a primary domain.

        Args:
            primary_domain: Primary domain name

        Returns:
            List of suggested subdomains
        """
        return DOMAIN_CATEGORIES.get(primary_domain, [])

    async def classify_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
    ) -> List[DomainResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts
            batch_size: Batch size for processing

        Returns:
            List of DomainResults
        """
        import asyncio
        from termextractor.utils.helpers import chunk_list

        logger.info(f"Batch classifying {len(texts)} texts")

        results = []
        text_chunks = chunk_list(texts, batch_size)

        for chunk in text_chunks:
            chunk_results = await asyncio.gather(
                *[self.classify(text) for text in chunk]
            )
            results.extend(chunk_results)

        return results

    def filter_by_domain(
        self,
        terms: List[Any],
        target_domain: str,
        exact_match: bool = False,
    ) -> List[Any]:
        """
        Filter terms by domain.

        Args:
            terms: List of terms (must have 'domain' attribute)
            target_domain: Target domain to filter by
            exact_match: Require exact match vs. contains

        Returns:
            Filtered list of terms
        """
        filtered = []

        for term in terms:
            if not hasattr(term, "domain"):
                continue

            if exact_match:
                if term.domain == target_domain:
                    filtered.append(term)
            else:
                if target_domain.lower() in term.domain.lower():
                    filtered.append(term)

        logger.info(
            f"Filtered {len(filtered)}/{len(terms)} terms for domain '{target_domain}'"
        )

        return filtered
