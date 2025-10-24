"""
TermExtractor - Advanced Terminology Extraction System

A state-of-the-art terminology extraction system powered by Anthropic's Claude AI.
"""

__version__ = "1.0.0"
__author__ = "TermExtractor Team"
__license__ = "MIT"

from termextractor.extraction.term_extractor import TermExtractor
from termextractor.extraction.domain_classifier import DomainClassifier
from termextractor.api.anthropic_client import AnthropicClient

__all__ = [
    "TermExtractor",
    "DomainClassifier",
    "AnthropicClient",
]
