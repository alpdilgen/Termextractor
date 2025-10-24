"""Extraction modules for terminology extraction."""

from termextractor.extraction.term_extractor import TermExtractor, ExtractionResult
from termextractor.extraction.domain_classifier import DomainClassifier, DomainResult
from termextractor.extraction.language_processor import LanguageProcessor

__all__ = [
    "TermExtractor",
    "ExtractionResult",
    "DomainClassifier",
    "DomainResult",
    "LanguageProcessor",
]
