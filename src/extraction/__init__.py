"""Extraction modules for terminology extraction."""

# FIXED: Ensure relative imports start with '.'
from .term_extractor import TermExtractor, ExtractionResult
from .domain_classifier import DomainClassifier, DomainResult
from .language_processor import LanguageProcessor

__all__ = [
    "TermExtractor",
    "ExtractionResult",
    "DomainClassifier",
    "DomainResult",
    "LanguageProcessor",
]
