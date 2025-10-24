"""Main terminology extraction module."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field # Keep dataclass if needed elsewhere
from pathlib import Path
from loguru import logger

# Corrected imports relative to src/ or same package
from api.api_manager import APIManager
from .domain_classifier import DomainClassifier, DomainResult # Added DomainResult
from .language_processor import LanguageProcessor
from core.progress_tracker import ProgressTracker
from utils.helpers import chunk_list, parse_domain_path # Added parse_domain_path
from file_io.file_parser import FileParser

# Import data models from the new file
from .data_models import Term, ExtractionResult


class TermExtractor:
    """
    Main terminology extraction system.
    Features: AI-powered extraction, bilingual/monolingual, domain context, etc.
    """
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
        # Ensure components are initialized if not provided
        self.domain_classifier = domain_classifier or DomainClassifier(api_client)
        self.language_processor = language_processor or LanguageProcessor()
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.default_relevance_threshold = default_relevance_threshold
        logger.info(f"TermExtractor initialized with threshold {default_relevance_threshold}")

    async def extract_from_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None, # User-provided path string
        context: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract terms from text."""
        logger.info(f"Extracting terms from text ({len(text)} chars, {source_lang}->{target_lang or 'mono'})")

        domain_hierarchy_list: List[str] = ["General"] # Default
        domain_to_pass_api: Optional[str] = None # Domain string for API prompt

        # Determine domain
        if domain_path: # User provided a path
            domain_hierarchy_list = parse_domain_path(domain_path)
            domain_to_pass_api = " / ".join(domain_hierarchy_list) # Format for API
            logger.info(f"Using user-specified domain: {domain_to_pass_api}")
        elif self.domain_classifier: # Try automatic classification
            logger.info("Attempting automatic domain classification...")
            try:
                domain_result: DomainResult = await self.domain_classifier.classify(text)
                if domain_result and domain_result.hierarchy:
                     domain_hierarchy_list = domain_result.hierarchy
                     domain_to_pass_api = " / ".join(domain_hierarchy_list)
                     logger.info(f"AI classified domain: {domain_to_pass_api} (Conf: {domain_result.confidence:.2f})")
                else:
                     logger.warning("Automatic domain classification returned no hierarchy. Using 'General'.")
            except Exception as e:
                logger.error(f"Domain classification failed: {e}. Using 'General'.")
        else:
             logger.info("No domain path provided and no classifier available. Using 'General'.")
             domain_to_pass_api = "General"


        # Prepare for API call
        terms_list: List[Term] = []
        api_result_metadata: Dict[str, Any] = {}
        api_stats: Dict[str, Any] = {}

        try:
            # Extract terms using API client (via manager)
            # Assumes api_client.extract_terms handles API calls, retries, etc.
            # and returns a dictionary matching the expected structure
            api_response: Dict[str, Any] = await self.api_client.extract_terms(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain_to_pass_api, # Pass the string version
                context=context,
            )

            # Parse results safely
            terms_data = api_response.get("terms", [])
            if isinstance(terms_data, list):
                 terms_list = self._parse_terms(terms_data)
            else:
                 logger.warning(f"API response 'terms' field was not a list: {type(terms_data)}")

            # Use domain from API if returned and valid, otherwise keep classified/user one
            api_domain_hierarchy = api_response.get("domain_hierarchy")
            if isinstance(api_domain_hierarchy, list) and api_domain_hierarchy:
                 domain_hierarchy_list = api_domain_hierarchy

            api_stats = api_response.get("statistics", {})
            api_result_metadata = api_response.get("_metadata", {}) # Include API cost, tokens etc.

        except Exception as e:
             logger.error(f"API call via manager failed during term extraction: {e}", exc_info=True)
             api_result_metadata["error"] = f"API call failed: {e}"


        # Create extraction result object
        extraction_result = ExtractionResult(
            terms=terms_list,
            domain_hierarchy=domain_hierarchy_list,
            language_pair=f"{source_lang}-{target_lang or source_lang}",
            source_language=source_lang,
            target_language=target_lang,
            statistics=api_stats, # Use stats from API if available
            metadata=api_result_metadata,
        )

        # Apply final relevance threshold filter (uses default if None provided)
        threshold_to_apply = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
        if threshold_to_apply >= 0: # Allow 0 threshold to skip filtering
            final_result = extraction_result.filter_by_relevance(threshold_to_apply)
            logger.info(f"Extracted {len(terms_list)} terms before filter, {len(final_result.terms)} terms after applying threshold >= {threshold_to_apply}")
            # Note: filter_by_relevance recalculates stats based on filtered list now
            final_result.statistics = final_result._calculate_statistics(final_result.terms)
            return final_result
        else:
            logger.info(f"Extracted {len(extraction_result.terms)} terms (no threshold filter applied)")
            # Ensure stats are calculated if not returned by API
            if not extraction_result.statistics and extraction_result.terms:
                 extraction_result.statistics = extraction_result._calculate_statistics(extraction_result.terms)
            return extraction_result


    async def extract_from_file(
        self,
        file_path: Union[str, Path],
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract terms from a file using the FileParser."""
        logger.info(f"Extracting terms from file: {file_path}")
        file_path = Path(file_path) # Ensure it's a Path object
        parser = FileParser() # Uses the correctly imported class
        parsed_content: Dict[str, Any] = {}

        try:
            parsed_content = await parser.parse_file(file_path)

            # Check if parsing was successful and returned text
            if not parsed_content or "text" not in parsed_content:
                error_msg = parsed_content.get("metadata", {}).get("error", "File parser returned invalid data.")
                logger.error(f"File parsing failed or returned no text for {file_path.name}: {error_msg}")
                return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                        metadata={"error": f"File parsing failed: {error_msg}"})
            elif len(parsed_content["text"]) == 0:
                 logger.warning(f"File parser extracted 0 characters from {file_path.name}. No terms can be extracted.")
                 return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                         metadata={"warning": "File parsing yielded empty text."})


        except Exception as e:
             logger.error(f"Failed to parse file {file_path}: {e}", exc_info=True)
             return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                     metadata={"error": f"File parsing failed: {e}"})

        # Proceed to extract from the parsed text
        # Pass file metadata along if needed (e.g., in context for API)
        return await self.extract_from_text(
            text=parsed_content["text"],
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain_path,
            relevance_threshold=relevance_threshold,
            # context=f"File: {file_path.name}" # Example context
        )


    def _parse_terms(self, terms_data: List[Dict[str, Any]]) -> List[Term]:
        """Safely parse term dictionaries from API response into Term objects."""
        parsed_terms = []
        if not isinstance(terms_data, list):
            logger.warning("Terms data from API is not a list, cannot parse.")
            return []

        for i, term_dict in enumerate(terms_data):
            if not isinstance(term_dict, dict):
                 logger.warning(f"Item {i} in terms data is not a dictionary, skipping.")
                 continue
            try:
                # Extract fields safely using .get() with defaults
                term = Term(
                    term=str(term_dict.get("term", "")), # Ensure string
                    translation=term_dict.get("translation"), # Optional[str] is fine
                    domain=str(term_dict.get("domain", "General")),
                    subdomain=term_dict.get("subdomain"),
                    pos=str(term_dict.get("pos", "NOUN")),
                    definition=str(term_dict.get("definition", "")),
                    context=str(term_dict.get("context", "")),
                    relevance_score=float(term_dict.get("relevance_score", 0.0)),
                    confidence_score=float(term_dict.get("confidence_score", 0.0)),
                    frequency=int(term_dict.get("frequency", 1)),
                    is_compound=bool(term_dict.get("is_compound", False)),
                    is_abbreviation=bool(term_dict.get("is_abbreviation", False)),
                    variants=list(term_dict.get("variants", [])),
                    related_terms=list(term_dict.get("related_terms", [])),
                    # metadata field might itself contain complex objects
                    metadata=dict(term_dict.get("metadata", {})),
                )
                # Basic validation: term text should not be empty
                if not term.term:
                     logger.warning(f"Parsed term at index {i} has empty 'term' field, skipping.")
                     continue
                parsed_terms.append(term)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to parse term at index {i} due to type error: {e}. Data: {term_dict}")
            except Exception as e:
                logger.error(f"Unexpected error parsing term at index {i}: {e}. Data: {term_dict}", exc_info=True)

        return parsed_terms

    # --- Batch Processing & Merging ---
    # (Keep implementations, ensure they use ExtractionResult correctly)
    async def extract_batch(
         self, texts: List[str], source_lang: str, target_lang: Optional[str] = None,
         domain_path: Optional[str] = None, batch_size: int = 10, show_progress: bool = True
     ) -> List[ExtractionResult]:
         """Extract terms from multiple texts in batches."""
         # ... (Implementation using asyncio.gather and extract_from_text) ...
         # Remember to handle progress tracking if needed (using self.progress_tracker)
         return [] # Placeholder

    async def merge_results(
        self, results: List[ExtractionResult], deduplicate: bool = True
    ) -> ExtractionResult:
        """Merge multiple extraction results."""
        # ... (Implementation seems okay, relies on ExtractionResult methods) ...
        return ExtractionResult(terms=[]) # Placeholder
