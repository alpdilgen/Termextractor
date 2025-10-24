"""Main terminology extraction module."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Corrected imports relative to src/ or same package
from api.api_manager import APIManager
from .domain_classifier import DomainClassifier, DomainResult
from .language_processor import LanguageProcessor
from core.progress_tracker import ProgressTracker
from utils.helpers import chunk_list, parse_domain_path
from file_io.file_parser import FileParser

# Import data models from the new file
from .data_models import Term, ExtractionResult


class TermExtractor:
    """
    Main terminology extraction system. Orchestrates parsing, API calls, and result formatting.
    """
    def __init__(
        self,
        api_client: APIManager,
        domain_classifier: Optional[DomainClassifier] = None,
        language_processor: Optional[LanguageProcessor] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        default_relevance_threshold: float = 70.0, # Use float
    ):
        """Initialize TermExtractor."""
        if not isinstance(api_client, APIManager):
             # Ensure correct type is passed
             raise TypeError("api_client must be an instance of APIManager")

        self.api_client = api_client
        # Lazy initialization or pass dependencies correctly
        self.domain_classifier = domain_classifier
        self.language_processor = language_processor or LanguageProcessor() # Default if None
        self.progress_tracker = progress_tracker or ProgressTracker() # Default if None
        # Ensure threshold is a float
        self.default_relevance_threshold = float(default_relevance_threshold)
        logger.info(f"TermExtractor initialized with default threshold {self.default_relevance_threshold}")

    async def extract_from_text(
        self,
        text: str,
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None, # User-provided path string
        context: Optional[str] = None,
        relevance_threshold: Optional[float] = None,
    ) -> ExtractionResult:
        """Extract terms from a string of text."""
        logger.info(f"Extracting terms from text ({len(text)} chars, {source_lang}->{target_lang or 'mono'})")

        domain_hierarchy_list: List[str] = ["General"] # Default
        domain_to_pass_api: Optional[str] = None

        # --- Domain Determination ---
        if domain_path and domain_path.strip(): # User provided a path
            domain_path = domain_path.strip()
            domain_hierarchy_list = parse_domain_path(domain_path) # Use helper
            domain_to_pass_api = " / ".join(domain_hierarchy_list)
            logger.info(f"Using user-specified domain: {domain_to_pass_api}")
        elif self.domain_classifier: # Try automatic classification
            logger.info("Attempting automatic domain classification...")
            try:
                # Initialize domain classifier if not already done (dependency injection preferred)
                if not isinstance(self.domain_classifier, DomainClassifier):
                     self.domain_classifier = DomainClassifier(self.api_client)

                domain_result = await self.domain_classifier.classify(text)
                if domain_result and domain_result.hierarchy:
                     domain_hierarchy_list = domain_result.hierarchy
                     domain_to_pass_api = " / ".join(domain_hierarchy_list)
                     logger.info(f"AI classified domain: {domain_to_pass_api} (Conf: {getattr(domain_result, 'confidence', 0.0):.2f})")
                else:
                     logger.warning("Automatic domain classification returned no valid hierarchy. Using 'General'.")
                     domain_to_pass_api = "General" # Fallback
            except Exception as e:
                logger.error(f"Domain classification process failed: {e}. Using 'General'.", exc_info=True)
                domain_to_pass_api = "General" # Fallback
        else:
             logger.info("No domain path provided and no classifier available. Using 'General'.")
             domain_to_pass_api = "General"

        # --- API Call for Term Extraction ---
        terms_list: List[Term] = []
        api_result_metadata: Dict[str, Any] = {}
        api_stats: Dict[str, Any] = {}
        api_response_error: Optional[str] = None

        try:
            # Use the API manager which handles caching, rate limiting etc.
            # The actual API call happens within api_client.extract_terms
            api_response = await self.api_client.extract_terms(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain_to_pass_api,
                context=context,
                use_cache=True # Default to using cache
            )

            # Check if API call itself returned an error structure
            if isinstance(api_response, dict) and "error" in api_response:
                 api_response_error = api_response["error"]
                 logger.error(f"API call returned an error: {api_response_error}")
                 api_result_metadata = api_response.get("_metadata", {}) # Still get metadata if available
            elif isinstance(api_response, dict):
                 # --- Parse Successful Response ---
                 terms_data = api_response.get("terms", [])
                 logger.info(f"Parsing {len(terms_data) if isinstance(terms_data, list) else 'invalid'} term entries received from API.")
                 terms_list = self._parse_terms(terms_data) # Safely parse
                 logger.info(f"Successfully parsed {len(terms_list)} Term objects.")

                 # Use domain from API if valid, otherwise keep classified/user one
                 api_domain_hierarchy = api_response.get("domain_hierarchy")
                 if isinstance(api_domain_hierarchy, list) and api_domain_hierarchy:
                      domain_hierarchy_list = api_domain_hierarchy

                 api_stats = api_response.get("statistics", {})
                 api_result_metadata = api_response.get("_metadata", {})
            else:
                 # Handle unexpected response type from API manager
                 api_response_error = f"Unexpected response type from API manager: {type(api_response)}"
                 logger.error(api_response_error)


        except Exception as e:
             logger.error(f"Error during API call via manager for term extraction: {e}", exc_info=True)
             api_response_error = f"Term extraction call failed: {e}"

        # Store error in metadata if one occurred
        if api_response_error:
             api_result_metadata["error"] = api_response_error

        # --- Create ExtractionResult Object ---
        extraction_result = ExtractionResult(
            terms=terms_list,
            domain_hierarchy=domain_hierarchy_list,
            language_pair=f"{source_lang}-{target_lang or source_lang}",
            source_language=source_lang,
            target_language=target_lang,
            statistics=api_stats, # Store stats from API (might be empty)
            metadata=api_result_metadata, # Store API call metadata (tokens, cost, errors)
        )

        # Apply final relevance threshold filter
        threshold_to_apply = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
        if threshold_to_apply >= 0: # Allows 0 threshold to skip filtering
             # filter_by_relevance returns a NEW ExtractionResult object
             final_result = extraction_result.filter_by_relevance(threshold_to_apply)
             logger.info(f"Returning {len(final_result.terms)} terms after applying threshold >= {threshold_to_apply:.1f} (Initial valid parsed count: {len(terms_list)})")
             # Recalculate stats on the filtered list
             final_result.statistics = final_result._calculate_statistics(final_result.terms)
             # Ensure metadata (like errors) is preserved
             final_result.metadata = extraction_result.metadata
             return final_result
        else:
             logger.info(f"Returning {len(extraction_result.terms)} terms (no threshold filter applied)")
             # Calculate stats if API didn't provide them and terms exist
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
        logger.info(f"Starting file extraction process for: {file_path}")
        file_path = Path(file_path) # Ensure Path object
        parser = FileParser() # Instantiate the parser
        parsed_content: Optional[Dict[str, Any]] = None
        error_metadata = {}

        try:
            parsed_content = await parser.parse_file(file_path)

            # Check for parsing errors or empty content
            if not isinstance(parsed_content, dict):
                 error_msg = "File parser returned invalid data type."
                 logger.error(f"{error_msg} for {file_path.name}")
                 error_metadata["error"] = error_msg
            elif "error" in parsed_content.get("metadata", {}):
                 error_msg = parsed_content["metadata"]["error"]
                 logger.error(f"File parsing failed for {file_path.name}: {error_msg}")
                 error_metadata["error"] = f"File parsing failed: {error_msg}"
            elif "text" not in parsed_content or len(parsed_content.get("text", "")) == 0:
                 warn_msg = f"File parser extracted 0 characters from {file_path.name}."
                 logger.warning(warn_msg)
                 error_metadata["warning"] = warn_msg # Use warning key
                 # Allow proceeding to extract_from_text, which will handle empty text
            else:
                 logger.info(f"Successfully parsed {file_path.name}. Text length: {len(parsed_content['text'])}")

        except Exception as e:
             logger.error(f"Unexpected error during file parsing stage for {file_path}: {e}", exc_info=True)
             error_metadata["error"] = f"File parsing stage failed: {e}"

        # If parsing failed critically or returned unusable data, return empty result with error
        if "error" in error_metadata:
             return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                     metadata=error_metadata)

        # Proceed to extract from the parsed text (even if empty, let extract_from_text handle it)
        text_content = parsed_content.get("text", "") if parsed_content else ""
        extraction_result = await self.extract_from_text(
            text=text_content,
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain_path,
            relevance_threshold=relevance_threshold,
            # context=f"File: {file_path.name}" # Example context
        )

        # Combine parsing warnings/errors with extraction metadata
        if "warning" in error_metadata: # Keep parsing warnings
             extraction_result.metadata.update(error_metadata)

        return extraction_result


    def _parse_terms(self, terms_data: List[Dict[str, Any]]) -> List[Term]:
        """Safely parse a list of dictionaries from API into Term objects."""
        parsed_terms = []
        if not isinstance(terms_data, list):
            logger.warning(f"Invalid terms data received from API (not a list): {type(terms_data)}")
            return []

        for i, term_dict in enumerate(terms_data):
            if not isinstance(term_dict, dict):
                 logger.warning(f"Skipping item #{i+1} in terms data: not a dictionary.")
                 continue
            try:
                # Use .get with defaults and type casting for robustness
                term_text = str(term_dict.get("term", "")).strip()
                if not term_text: # Skip if term itself is empty
                     logger.warning(f"Skipping item #{i+1}: 'term' field is missing or empty.")
                     continue

                term = Term(
                    term=term_text,
                    translation=term_dict.get("translation"), # Allow None
                    domain=str(term_dict.get("domain", "General")),
                    subdomain=term_dict.get("subdomain"), # Allow None
                    pos=str(term_dict.get("pos", "UNKNOWN")),
                    definition=str(term_dict.get("definition", "")),
                    context=str(term_dict.get("context", "")),
                    relevance_score=float(term_dict.get("relevance_score", 0.0)),
                    confidence_score=float(term_dict.get("confidence_score", 0.0)),
                    frequency=int(term_dict.get("frequency", 1)),
                    is_compound=bool(term_dict.get("is_compound", False)),
                    is_abbreviation=bool(term_dict.get("is_abbreviation", False)),
                    variants=list(term_dict.get("variants", []) if isinstance(term_dict.get("variants"), list) else []),
                    related_terms=list(term_dict.get("related_terms", []) if isinstance(term_dict.get("related_terms"), list) else []),
                    metadata=dict(term_dict.get("metadata", {}) if isinstance(term_dict.get("metadata"), dict) else {}),
                )
                parsed_terms.append(term)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to parse term #{i+1} due to type/value error: {e}. Data: {term_dict}")
            except Exception as e:
                logger.error(f"Unexpected error parsing term #{i+1}: {e}. Data: {term_dict}", exc_info=True)

        return parsed_terms

    # --- Batch Processing & Merging ---
    async def extract_batch(
         self, texts: List[str], source_lang: str, target_lang: Optional[str] = None,
         domain_path: Optional[str] = None, batch_size: int = 10, show_progress: bool = True,
         relevance_threshold: Optional[float] = None # Added threshold parameter
     ) -> List[ExtractionResult]:
         """Extract terms from multiple texts in batches."""
         if not texts: return []
         logger.info(f"Starting batch extraction for {len(texts)} texts with batch size {batch_size}.")

         task_id = "batch_extraction"
         if show_progress and self.progress_tracker:
             self.progress_tracker.create_task(task_id, "Batch Term Extraction", len(texts))
             self.progress_tracker.start_task(task_id)

         results: List[ExtractionResult] = []
         text_chunks = chunk_list(texts, batch_size)
         processed_count = 0

         for batch_idx, batch in enumerate(text_chunks):
             batch_tasks = [
                 self.extract_from_text(
                     text=text, source_lang=source_lang, target_lang=target_lang,
                     domain_path=domain_path, relevance_threshold=relevance_threshold # Pass threshold
                 ) for text in batch
             ]
             # Run tasks in the current batch concurrently
             batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

             # Process results and handle potential exceptions from gather
             for result_or_exc in batch_results:
                  processed_count += 1
                  if isinstance(result_or_exc, Exception):
                       logger.error(f"Error in batch item {processed_count}: {result_or_exc}")
                       # Append an empty result with error info
                       results.append(ExtractionResult(terms=[], metadata={"error": str(result_or_exc)}))
                  elif isinstance(result_or_exc, ExtractionResult):
                       results.append(result_or_exc)
                  else:
                       logger.warning(f"Unexpected result type in batch item {processed_count}: {type(result_or_exc)}")
                       results.append(ExtractionResult(terms=[], metadata={"error": "Unexpected result type"}))

             if show_progress and self.progress_tracker:
                  self.progress_tracker.update_progress(task_id, completed=processed_count, message=f"Processed {processed_count}/{len(texts)}")

         if show_progress and self.progress_tracker:
             self.progress_tracker.complete_task(task_id)

         logger.info(f"Batch extraction finished. Processed {len(results)} results.")
         return results


    async def merge_results(
        self, results: List[ExtractionResult], deduplicate: bool = True
    ) -> ExtractionResult:
        """Merge multiple extraction results, optionally deduplicating terms."""
        if not results:
            return ExtractionResult(terms=[])

        all_terms: List[Term] = []
        merged_metadata = {"merged_from": len(results), "deduplicated": deduplicate, "errors": []}
        base_result = results[0] # Use first result for base info

        for i, result in enumerate(results):
             if isinstance(result, ExtractionResult):
                  all_terms.extend(result.terms)
                  if "error" in result.metadata:
                       merged_metadata["errors"].append(f"Result {i}: {result.metadata['error']}")
             else:
                  logger.warning(f"Item {i} in results list is not an ExtractionResult: {type(result)}")
                  merged_metadata["errors"].append(f"Result {i}: Invalid type {type(result)}")


        final_terms = all_terms
        if deduplicate:
            seen = set()
            unique_terms = []
            for term in all_terms:
                # Deduplicate based on term text and target language (if exists)
                key = (term.term.lower(), term.translation.lower() if term.translation else None)
                if key not in seen:
                    seen.add(key)
                    unique_terms.append(term)
                else:
                    # Optional: Could merge info from duplicates here (e.g., aggregate frequency)
                    pass
            final_terms = unique_terms
            logger.info(f"Merged {len(all_terms)} terms into {len(final_terms)} unique terms.")
        else:
            logger.info(f"Merged {len(all_terms)} terms without deduplication.")


        # Create the final merged result
        merged_result = ExtractionResult(
            terms=final_terms,
            # Use base info, maybe aggregate domains? For now, use first non-empty.
            domain_hierarchy=next((r.domain_hierarchy for r in results if r.domain_hierarchy), ["General"]),
            language_pair=base_result.language_pair,
            source_language=base_result.source_language,
            target_language=base_result.target_language,
            metadata=merged_metadata,
            # Statistics need recalculation based on final_terms
        )
        # Recalculate stats for the final merged list
        merged_result.statistics = merged_result._calculate_statistics(final_terms)

        return merged_result
