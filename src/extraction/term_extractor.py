# src/extraction/term_extractor.py

"""Main terminology extraction module."""

import asyncio
import re # <-- HATA DÜZELTMESİ: 're' modülü buraya eklendi
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


# --- METİN PARÇALAMA LİMİTİ ---
# AI'nın JSON hatası vermemesi için daha küçük parçalar kullanıyoruz.
TEXT_CHUNK_SIZE = 2000 # Karakter cinsinden


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
             raise TypeError("api_client must be an instance of APIManager")

        self.api_client = api_client
        self.domain_classifier = domain_classifier
        self.language_processor = language_processor or LanguageProcessor()
        self.progress_tracker = progress_tracker or ProgressTracker()
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
        """Extract terms from a single string of text (now assumes text is reasonably sized)."""
        logger.info(f"Extracting terms from text chunk ({len(text)} chars, {source_lang}->{target_lang or 'mono'})")

        domain_hierarchy_list: List[str] = ["General"] # Default
        domain_to_pass_api: Optional[str] = None

        # --- Domain Determination ---
        if domain_path and domain_path.strip():
            domain_path = domain_path.strip()
            domain_hierarchy_list = parse_domain_path(domain_path)
            domain_to_pass_api = " / ".join(domain_hierarchy_list)
            logger.info(f"Using user-specified domain: {domain_to_pass_api}")
        elif self.domain_classifier:
            logger.info("Attempting automatic domain classification...")
            try:
                if not isinstance(self.domain_classifier, DomainClassifier):
                     self.domain_classifier = DomainClassifier(self.api_client)
                domain_result = await self.domain_classifier.classify(text)
                if domain_result and domain_result.hierarchy:
                     domain_hierarchy_list = domain_result.hierarchy
                     domain_to_pass_api = " / ".join(domain_hierarchy_list)
                     logger.info(f"AI classified domain: {domain_to_pass_api} (Conf: {getattr(domain_result, 'confidence', 0.0):.2f})")
                else:
                     logger.warning("Automatic domain classification returned no valid hierarchy. Using 'General'.")
                     domain_to_pass_api = "General"
            except Exception as e:
                logger.error(f"Domain classification process failed: {e}. Using 'General'.", exc_info=True)
                domain_to_pass_api = "General"
        else:
             logger.info("No domain path provided and no classifier available. Using 'General'.")
             domain_to_pass_api = "General"
        
        # Add context if provided
        if context:
             domain_to_pass_api = f"{domain_to_pass_api} (Context: {context})"


        # --- API Call for Term Extraction ---
        terms_list: List[Term] = []
        api_result_metadata: Dict[str, Any] = {}
        api_stats: Dict[str, Any] = {}
        api_response_error: Optional[str] = None

        try:
            api_response = await self.api_client.extract_terms(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain_to_pass_api,
                context=context, # Context is passed separately too
                use_cache=True
            )

            if isinstance(api_response, dict) and "error" in api_response:
                 api_response_error = api_response["error"]
                 logger.error(f"API call returned an error: {api_response_error}")
                 api_result_metadata = api_response.get("_metadata", {})
            elif isinstance(api_response, dict):
                 terms_data = api_response.get("terms", [])
                 logger.info(f"Parsing {len(terms_data) if isinstance(terms_data, list) else 'invalid'} term entries received from API.")
                 terms_list = self._parse_terms(terms_data)
                 logger.info(f"Successfully parsed {len(terms_list)} Term objects.")
                 api_domain_hierarchy = api_response.get("domain_hierarchy")
                 if isinstance(api_domain_hierarchy, list) and api_domain_hierarchy:
                      domain_hierarchy_list = api_domain_hierarchy
                 api_stats = api_response.get("statistics", {})
                 api_result_metadata = api_response.get("_metadata", {})
            else:
                 api_response_error = f"Unexpected response type from API manager: {type(api_response)}"
                 logger.error(api_response_error)

        except Exception as e:
             logger.error(f"Error during API call via manager for term extraction: {e}", exc_info=True)
             api_response_error = f"Term extraction call failed: {e}"

        if api_response_error:
             api_result_metadata["error"] = api_response_error

        # --- Create ExtractionResult Object ---
        extraction_result = ExtractionResult(
            terms=terms_list,
            domain_hierarchy=domain_hierarchy_list,
            language_pair=f"{source_lang}-{target_lang or source_lang}",
            source_language=source_lang,
            target_language=target_lang,
            statistics=api_stats,
            metadata=api_result_metadata,
        )

        # Apply final relevance threshold filter
        threshold_to_apply = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
        
        # We apply filtering *after* merging, so return the full result for this chunk
        if threshold_to_apply >= 0:
             logger.debug(f"Chunk processed, returning {len(extraction_result.terms)} terms (filtering will happen after merge).")
        
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
        """
        Extract terms from a file.
        Handles parsing and chunking for large files.
        """
        logger.info(f"Starting file extraction process for: {file_path}")
        file_path = Path(file_path)
        parser = FileParser()
        parsed_content: Optional[Dict[str, Any]] = None
        error_metadata = {}

        try:
            parsed_content = await parser.parse_file(file_path)
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
                 error_metadata["warning"] = warn_msg
            else:
                 logger.info(f"Successfully parsed {file_path.name}. Text length: {len(parsed_content['text'])}")

        except Exception as e:
             logger.error(f"Unexpected error during file parsing stage for {file_path}: {e}", exc_info=True)
             error_metadata["error"] = f"File parsing stage failed: {e}"

        if "error" in error_metadata:
             return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                     metadata=error_metadata)

        text_content = parsed_content.get("text", "") if parsed_content else ""
        if not text_content.strip():
             logger.warning(f"File {file_path.name} contains no non-whitespace text. Returning 0 terms.")
             return ExtractionResult(terms=[], source_language=source_lang, target_language=target_lang,
                                     metadata=error_metadata)

        # --- CHUNKING LOGIC ---
        # Nested function for smart chunking
        def split_text_into_chunks(text, chunk_size):
            """Splits text into chunks, respecting paragraph boundaries where possible."""
            chunks = []
            current_chunk = ""
            # Split by double newline (paragraphs), keeping the separator
            paragraphs = re.split(r'(\n\s*\n)', text)
            
            for i in range(0, len(paragraphs), 2): # Iterate over text + separator
                paragraph = paragraphs[i]
                separator = paragraphs[i+1] if i+1 < len(paragraphs) else ""
                
                if not paragraph: # Skip empty paragraphs
                     if separator: current_chunk += separator
                     continue

                # If adding the next paragraph fits, add it
                if len(current_chunk) + len(paragraph) + len(separator) <= chunk_size:
                    current_chunk += paragraph + separator
                # If the paragraph *itself* is too large, split it
                elif len(paragraph) > chunk_size:
                    if current_chunk.strip(): # Save what we have
                        chunks.append(current_chunk.strip())
                    # Force-split the large paragraph
                    for j in range(0, len(paragraph), chunk_size):
                         chunks.append(paragraph[j:j+chunk_size])
                    current_chunk = separator # Start new chunk with separator
                # Otherwise, the paragraph doesn't fit, save current chunk and start new
                else:
                    if current_chunk.strip():
                         chunks.append(current_chunk.strip())
                    current_chunk = paragraph + separator
            
            # Add the last remaining chunk
            if current_chunk.strip():
                 chunks.append(current_chunk.strip())
                 
            logger.info(f"Split text into {len(chunks)} smart chunks based on paragraphs (max size {chunk_size} chars).")
            return chunks

        if len(text_content) <= TEXT_CHUNK_SIZE:
            logger.info("Text is small enough, processing as single chunk.")
            result = await self.extract_from_text(
                text=text_content,
                source_lang=source_lang,
                target_lang=target_lang,
                domain_path=domain_path,
                relevance_threshold=-1, # Don't filter yet
            )
            # Apply filter now
            threshold_to_apply = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
            if threshold_to_apply >= 0:
                 final_result = result.filter_by_relevance(threshold_to_apply)
                 logger.info(f"Returning {len(final_result.terms)} terms after applying threshold >= {threshold_to_apply:.1f} (Initial: {len(result.terms)})")
                 final_result.statistics = final_result._calculate_statistics(final_result.terms)
                 final_result.metadata = result.metadata
                 return final_result
            else:
                 logger.info(f"Returning {len(result.terms)} terms (no threshold filter).")
                 return result
        else:
            # Text is large, split into chunks
            logger.info(f"Text is large ({len(text_content)} chars), splitting into chunks...")
            chunks = split_text_into_chunks(text_content, TEXT_CHUNK_SIZE)
            logger.info(f"Processing {len(chunks)} chunks in parallel (using asyncio.gather).")

            tasks = []
            for i, chunk in enumerate(chunks):
                chunk_context = f"This is chunk {i+1} of {len(chunks)} from the document."
                tasks.append(
                    self.extract_from_text(
                        text=chunk,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        domain_path=domain_path,
                        context=chunk_context,
                        relevance_threshold=-1 # IMPORTANT: Disable filtering until after merge
                    )
                )
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results: List[ExtractionResult] = []
            for i, res in enumerate(results_list):
                if isinstance(res, ExtractionResult):
                    valid_results.append(res)
                    if "error" in res.metadata:
                        logger.warning(f"Chunk {i+1} processed but API returned an error: {res.metadata['error']}")
                elif isinstance(res, Exception):
                    logger.error(f"Chunk {i+1} processing failed with exception: {res}", exc_info=True)
                else:
                    logger.error(f"Chunk {i+1} returned unknown type: {type(res)}")
            
            logger.info(f"Merging results from {len(valid_results)} successful chunks.")
            merged_result = await self.merge_results(valid_results, deduplicate=True)
            
            # Apply the final threshold filter *after* merging
            threshold_to_apply = relevance_threshold if relevance_threshold is not None else self.default_relevance_threshold
            if threshold_to_apply >= 0:
                 final_filtered_result = merged_result.filter_by_relevance(threshold_to_apply)
                 logger.info(f"Returning {len(final_filtered_result.terms)} terms after merging and applying threshold >= {threshold_to_apply:.1f} (Total unique terms: {len(merged_result.terms)})")
                 final_filtered_result.statistics = final_filtered_result._calculate_statistics(final_filtered_result.terms)
                 final_filtered_result.metadata["total_chunks"] = len(chunks)
                 final_filtered_result.metadata["successful_chunks"] = len(valid_results)
                 return final_filtered_result
            else:
                 logger.info(f"Returning {len(merged_result.terms)} merged terms (no threshold filter applied).")
                 merged_result.metadata["total_chunks"] = len(chunks)
                 merged_result.metadata["successful_chunks"] = len(valid_results)
                 return merged_result


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
                term_text = str(term_dict.get("term", "")).strip()
                if not term_text:
                     logger.warning(f"Skipping item #{i+1}: 'term' field is missing or empty.")
                     continue

                term = Term(
                    term=term_text,
                    translation=term_dict.get("translation"),
                    domain=str(term_dict.get("domain", "General")),
                    subdomain=term_dict.get("subdomain"),
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
         relevance_threshold: Optional[float] = None
     ) -> List[ExtractionResult]:
         """Extract terms from multiple texts in batches."""
         if not texts: return []
         logger.info(f"Starting batch extraction for {len(texts)} texts with batch size {batch_size}.")

         task_id = "batch_extraction_list"
         if show_progress and self.progress_tracker:
             self.progress_tracker.create_task(task_id, "Batch Text Extraction", len(texts))
             self.progress_tracker.start_task(task_id)

         results: List[ExtractionResult] = []
         text_chunks_for_batching = chunk_list(texts, batch_size) # Groups texts
         processed_count = 0

         for batch_idx, batch_of_texts in enumerate(text_chunks_for_batching):
             logger.info(f"Processing batch {batch_idx+1}/{len(text_chunks_for_batching)}...")
             batch_tasks = []
             for text in batch_of_texts:
                 if text and text.strip():
                      # Each text item is processed as a "file" (which handles chunking)
                      # Call extract_from_text directly, assuming items in 'texts' aren't huge
                      batch_tasks.append(
                           self.extract_from_text(
                                text=text,
                                source_lang=source_lang,
                                target_lang=target_lang,
                                domain_path=domain_path,
                                relevance_threshold=relevance_threshold
                           )
                      )
                 else:
                      processed_count += 1
                      results.append(ExtractionResult(terms=[], metadata={"warning": "Empty text provided."}))


             if not batch_tasks: continue

             batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

             for result_or_exc in batch_results:
                  processed_count += 1
                  if isinstance(result_or_exc, Exception):
                       logger.error(f"Error in batch item {processed_count}: {result_or_exc}")
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

         logger.info(f"Batch extraction finished. Process
