"""Main terminology extraction module."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# FIXED: Corrected imports relative to src/ or same package
from api.api_manager import APIManager
from .domain_classifier import DomainClassifier
from .language_processor import LanguageProcessor
from core.progress_tracker import ProgressTracker
from utils.helpers import chunk_list
# FIXED: Corrected import relative to src/
from file_io.file_parser import FileParser # Changed 'io' to 'file_io'


@dataclass
class Term:
    """Represents an extracted term."""
    # ... (class content seems okay) ...
    term: str
    translation: Optional[str] = None
    domain: str = "General"
    subdomain: Optional[str] = None
    pos: str = "NOUN"
    definition: str = ""
    context: str = ""
    relevance_score: float = 0.0
    confidence_score: float = 0.0
    frequency: int = 1
    is_compound: bool = False
    is_abbreviation: bool = False
    variants: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
         """Convert to dictionary."""
         # ... (implementation seems okay) ...
         return {} # Placeholder


@dataclass
class ExtractionResult:
    """Results of terminology extraction."""
    # ... (class content seems okay) ...
    terms: List[Term]
    domain_hierarchy: List[str] = field(default_factory=list)
    language_pair: str = ""
    source_language: str = ""
    target_language: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def filter_by_relevance(self, threshold: float) -> "ExtractionResult":
         """Filter terms by relevance threshold."""
         # ... (implementation seems okay) ...
         return self # Placeholder

    def get_high_relevance_terms(self, threshold: float = 80) -> List[Term]:
         """Get terms with high relevance."""
         # ... (implementation seems okay) ...
         return [] # Placeholder

    def get_medium_relevance_terms(self, low_threshold: float = 60, high_threshold: float = 80) -> List[Term]:
         """Get terms with medium relevance."""
         # ... (implementation seems okay) ...
         return [] # Placeholder

    def get_low_relevance_terms(self, threshold: float = 60) -> List[Term]:
         """Get terms with low relevance."""
         # ... (implementation seems okay) ...
         return [] # Placeholder

    def _calculate_statistics(self, terms: List[Term]) -> Dict[str, Any]:
         """Calculate statistics for terms."""
         # ... (implementation seems okay) ...
         return {} # Placeholder

    def to_dict(self) -> Dict[str, Any]:
         """Convert to dictionary."""
         # ... (implementation seems okay) ...
         return {} # Placeholder


class TermExtractor:
    """Main terminology extraction system."""
    def __init__(
        self,
        api_client: APIManager,
        domain_classifier: Optional[DomainClassifier] = None,
        language_processor: Optional[LanguageProcessor] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        default_relevance_threshold: float = 70,
    ):
        """Initialize TermExtractor."""
        self.api_client = api_client
        self.domain_classifier = domain_classifier or DomainClassifier(api_client)
        self.language_processor = language_processor or LanguageProcessor()
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.default_relevance_threshold = default_relevance_threshold
        logger.info("TermExtractor initialized")

    async def extract_from_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None,
        context: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract terms from text."""
        # ... (implementation seems okay, relies on api_client.extract_terms) ...
        logger.info(f"Extracting terms from text ({len(text)} chars, {source_lang}->{target_lang or 'mono'})")
        domain = domain_path
        if not domain and self.domain_classifier: # Check if classifier exists
            try:
                domain_result = await self.domain_classifier.classify(text)
                domain = " / ".join(domain_result.hierarchy)
            except Exception as e:
                logger.warning(f"Domain classification failed: {e}. Using 'General'.")
                domain = "General"
        elif not domain:
             domain = "General"

        try:
            # Assuming api_client.extract_terms is implemented and returns dict
            result = await self.api_client.extract_terms(
                text=text, source_lang=source_lang, target_lang=target_lang,
                domain=domain, context=context,
            )
            terms = self._parse_terms(result.get("terms", []))
            domain_hierarchy = result.get("domain_hierarchy", domain.split(' / ')) # Use classified/provided domain
            stats = result.get("statistics", {})
            metadata = result.get("_metadata", {})
        except Exception as e:
             logger.error(f"API call to extract_terms failed: {e}")
             terms = []
             domain_hierarchy = domain.split(' / ')
             stats = {}
             metadata = {"error": str(e)}

        extraction_result = ExtractionResult(
            terms=terms, domain_hierarchy=domain_hierarchy,
            language_pair=f"{source_lang}-{target_lang or source_lang}",
            source_language=source_lang, target_language=target_lang,
            statistics=stats, metadata=metadata,
        )
        threshold = relevance_threshold or self.default_relevance_threshold
        if threshold > 0:
            extraction_result = extraction_result.filter_by_relevance(threshold)
        logger.info(f"Extracted {len(extraction_result.terms)} terms after filtering")
        return extraction_result


    async def extract_from_file(
        self,
        file_path: Union[str, Path],
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract terms from file."""
        # FIXED: Removed internal import, corrected path below
        logger.info(f"Extracting terms from file: {file_path}")
        try:
            parser = FileParser() # Uses the imported class
            parsed_content = await parser.parse_file(Path(file_path))
            if not parsed_content or "text" not in parsed_content or len(parsed_content["text"]) == 0:
                 logger.warning(f"File parser returned no text content for {file_path}")
                 # Return an empty result instead of proceeding
                 return ExtractionResult(
                      terms=[], source_language=source_lang, target_language=target_lang,
                      metadata={"error": "File parsing yielded no text."}
                 )

        except Exception as e:
             logger.error(f"Failed to parse file {file_path}: {e}")
             # Return an empty result with error metadata
             return ExtractionResult(
                  terms=[], source_language=source_lang, target_language=target_lang,
                  metadata={"error": f"File parsing failed: {e}"}
             )

        # Proceed to extract from the parsed text
        return await self.extract_from_text(
            text=parsed_content["text"],
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain_path,
            relevance_threshold=relevance_threshold,
        )

    # ... (rest of methods: extract_batch, _parse_terms, merge_results seem okay conceptually) ...
    # Ensure _parse_terms handles potential missing keys gracefully

    def _parse_terms(self, terms_data: List[Dict[str, Any]]) -> List[Term]:
        """Parse terms from API response."""
        # ... (implementation seems okay, ensure robustness to missing keys) ...
        return [] # Placeholder

    async def extract_batch(self, texts: List[str], source_lang: str, target_lang: Optional[str] = None, domain_path: Optional[str] = None, batch_size: int = 10, show_progress: bool = True) -> List[ExtractionResult]:
         """Extract terms from multiple texts in batches."""
         # ... (implementation seems okay) ...
         return [] # Placeholder

    async def merge_results(self, results: List[ExtractionResult], deduplicate: bool = True) -> ExtractionResult:
         """Merge multiple extraction results."""
         # ... (implementation seems okay) ...
         return ExtractionResult(terms=[]) # Placeholder
