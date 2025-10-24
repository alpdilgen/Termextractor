# src/extraction/data_models.py
"""Data models for Term and ExtractionResult."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time # Needed for _calculate_statistics if using start/end times

@dataclass
class Term:
    """Represents an extracted term."""

    term: str
    translation: Optional[str] = None
    domain: str = "General"
    subdomain: Optional[str] = None
    pos: str = "NOUN" # Part of Speech
    definition: str = ""
    context: str = ""
    relevance_score: float = 0.0 # Score 0-100
    confidence_score: float = 0.0 # Score 0-100
    frequency: int = 1
    is_compound: bool = False
    is_abbreviation: bool = False
    variants: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict) # For additional info

    def to_dict(self) -> Dict[str, Any]:
        """Convert Term object to a dictionary."""
        return {
            "term": self.term,
            "translation": self.translation,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "pos": self.pos,
            "definition": self.definition,
            "context": self.context,
            "relevance_score": self.relevance_score,
            "confidence_score": self.confidence_score,
            "frequency": self.frequency,
            "is_compound": self.is_compound,
            "is_abbreviation": self.is_abbreviation,
            "variants": self.variants,
            "related_terms": self.related_terms,
            "metadata": self.metadata,
        }

@dataclass
class ExtractionResult:
    """Results of terminology extraction."""

    terms: List[Term]
    domain_hierarchy: List[str] = field(default_factory=list)
    language_pair: str = ""
    source_language: str = ""
    target_language: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., processing time, errors

    def __post_init__(self):
        """Calculate statistics after initialization if not provided."""
        if not self.statistics and self.terms:
            self.statistics = self._calculate_statistics(self.terms)
        elif not self.statistics:
             self.statistics = self._calculate_statistics([]) # Ensure stats dict exists

    def filter_by_relevance(self, threshold: float) -> "ExtractionResult":
        """
        Filter terms by relevance threshold.

        Args:
            threshold: Minimum relevance score (0-100)

        Returns:
            New ExtractionResult with filtered terms (original is unchanged)
        """
        filtered_terms = [
            term for term in self.terms if term.relevance_score >= threshold
        ]
        # Create a new instance with filtered terms and recalculated stats
        return ExtractionResult(
            terms=filtered_terms,
            domain_hierarchy=self.domain_hierarchy,
            language_pair=self.language_pair,
            source_language=self.source_language,
            target_language=self.target_language,
            # statistics=self._calculate_statistics(filtered_terms), # Recalculate stats
            metadata=self.metadata, # Keep original metadata
        )

    def get_high_relevance_terms(self, threshold: float = 80) -> List[Term]:
        """Get terms with relevance score >= threshold."""
        return [t for t in self.terms if t.relevance_score >= threshold]

    def get_medium_relevance_terms(
        self, low_threshold: float = 60, high_threshold: float = 80
    ) -> List[Term]:
        """Get terms with relevance score between low_threshold (inclusive) and high_threshold (exclusive)."""
        return [
            t
            for t in self.terms
            if low_threshold <= t.relevance_score < high_threshold
        ]

    def get_low_relevance_terms(self, threshold: float = 60) -> List[Term]:
        """Get terms with relevance score < threshold."""
        return [t for t in self.terms if t.relevance_score < threshold]

    def _calculate_statistics(self, terms: List[Term]) -> Dict[str, Any]:
        """Calculate basic statistics for a list of terms."""
        if not terms:
            return {"total_terms": 0, "high_relevance": 0, "medium_relevance": 0, "low_relevance": 0}

        total = len(terms)
        high = len(self.get_high_relevance_terms()) # Use methods with default thresholds
        medium = len(self.get_medium_relevance_terms())
        low = len(self.get_low_relevance_terms())
        avg_relevance = sum(t.relevance_score for t in terms) / total if total > 0 else 0
        avg_confidence = sum(t.confidence_score for t in terms) / total if total > 0 else 0

        return {
            "total_terms": total,
            "high_relevance": high,
            "medium_relevance": medium,
            "low_relevance": low,
            "average_relevance": round(avg_relevance, 2),
            "average_confidence": round(avg_confidence, 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExtractionResult object to a dictionary."""
        return {
            "terms": [term.to_dict() for term in self.terms],
            "domain_hierarchy": self.domain_hierarchy,
            "language_pair": self.language_pair,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "statistics": self.statistics,
            "metadata": self.metadata,
        }
