"""Main terminology extraction module."""

import asyncio
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from api.api_manager import APIManager
from .domain_classifier import DomainClassifier
from .language_processor import LanguageProcessor
from core.progress_tracker import ProgressTracker
from utils.helpers import chunk_list


@dataclass
class Term:
    """Represents an extracted term."""

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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def filter_by_relevance(self, threshold: float) -> "ExtractionResult":
        """
        Filter terms by relevance threshold.

        Args:
            threshold: Minimum relevance score (0-100)

        Returns:
            New ExtractionResult with filtered terms
        """
        filtered_terms = [
            term for term in self.terms if term.relevance_score >= threshold
        ]

        return ExtractionResult(
            terms=filtered_terms,
            domain_hierarchy=self.domain_hierarchy,
            language_pair=self.language_pair,
            source_language=self.source_language,
            target_language=self.target_language,
            statistics=self._calculate_statistics(filtered_terms),
            metadata=self.metadata,
        )

    def get_high_relevance_terms(self, threshold: float = 80) -> List[Term]:
        """Get terms with high relevance."""
        return [t for t in self.terms if t.relevance_score >= threshold]

    def get_medium_relevance_terms(
        self, low_threshold: float = 60, high_threshold: float = 80
    ) -> List[Term]:
        """Get terms with medium relevance."""
        return [
            t
            for t in self.terms
            if low_threshold <= t.relevance_score < high_threshold
        ]

    def get_low_relevance_terms(self, threshold: float = 60) -> List[Term]:
        """Get terms with low relevance."""
        return [t for t in self.terms if t.relevance_score < threshold]

    def _calculate_statistics(self, terms: List[Term]) -> Dict[str, Any]:
        """Calculate statistics for terms."""
        if not terms:
            return {"total_terms": 0}

        high = len([t for t in terms if t.relevance_score >= 80])
        medium = len([t for t in terms if 60 <= t.relevance_score < 80])
        low = len([t for t in terms if t.relevance_score < 60])

        return {
            "total_terms": len(terms),
            "high_relevance": high,
            "medium_relevance": medium,
            "low_relevance": low,
            "average_relevance": sum(t.relevance_score for t in terms) / len(terms),
            "average_confidence": sum(t.confidence_score for t in terms) / len(terms),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "terms": [term.to_dict() for term in self.terms],
            "domain_hierarchy": self.domain_hierarchy,
            "language_pair": self.language_pair,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "statistics": self.statistics,
            "metadata": self.metadata,
        }


class TermExtractor:
    """
    Main terminology extraction system.

    Features:
    - AI-powered term extraction
    - Bilingual and monolingual processing
    - Domain-specific extraction
    - Batch processing
    - Progress tracking
    """

    def __init__(
        self,
        api_client: APIManager,
        domain_classifier: Optional[DomainClassifier] = None,
        language_processor: Optional[LanguageProcessor] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        default_relevance_threshold: float = 70,
    ):
        """
        Initialize TermExtractor.

        Args:
            api_client: API manager instance
            domain_classifier: Domain classifier (creates default if None)
            language_processor: Language processor (creates default if None)
            progress_tracker: Progress tracker (creates default if None)
            default_relevance_threshold: Default relevance threshold
        """
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
        """
        Extract terms from text.

        Args:
            text: Source text
            source_lang: Source language code
            target_lang: Target language code (None for monolingual)
            domain_path: Manual domain path (e.g., "Medical/Healthcare")
            context: Additional context
            relevance_threshold: Relevance threshold (None = use default)

        Returns:
            ExtractionResult with extracted terms
        """
        logger.info(
            f"Extracting terms from text ({len(text)} chars, {source_lang}->{target_lang or 'mono'})"
        )

        # Determine domain if not provided
        domain = domain_path
        if not domain:
            domain_result = await self.domain_classifier.classify(text)
            domain = " / ".join(domain_result.hierarchy)

        # Extract terms using API
        result = await self.api_client.extract_terms(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            context=context,
        )

        # Parse results
        terms = self._parse_terms(result.get("terms", []))

        # Create extraction result
        extraction_result = ExtractionResult(
            terms=terms,
            domain_hierarchy=result.get("domain_hierarchy", [domain]),
            language_pair=f"{source_lang}-{target_lang or source_lang}",
            source_language=source_lang,
            target_language=target_lang,
            statistics=result.get("statistics", {}),
            metadata=result.get("_metadata", {}),
        )

        # Apply relevance threshold
        threshold = relevance_threshold or self.default_relevance_threshold
        if threshold > 0:
            extraction_result = extraction_result.filter_by_relevance(threshold)

        logger.info(f"Extracted {len(extraction_result.terms)} terms")

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
        Extract terms from file.

        Args:
            file_path: Path to file
            source_lang: Source language code
            target_lang: Target language code
            domain_path: Manual domain path
            relevance_threshold: Relevance threshold

        Returns:
            ExtractionResult
        """
        from io.file_parser import FileParser

        logger.info(f"Extracting terms from file: {file_path}")

        # Parse file
        parser = FileParser()
        parsed_content = await parser.parse_file(Path(file_path))

        # Extract from text
        return await self.extract_from_text(
            text=parsed_content["text"],
            source_lang=source_lang,
            target_lang=target_lang,
            domain_path=domain_path,
            relevance_threshold=relevance_threshold,
        )

    async def extract_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: Optional[str] = None,
        domain_path: Optional[str] = None,
        batch_size: int = 10,
        show_progress: bool = True,
    ) -> List[ExtractionResult]:
        """
        Extract terms from multiple texts in batches.

        Args:
            texts: List of texts
            source_lang: Source language
            target_lang: Target language
            domain_path: Domain path
            batch_size: Batch size
            show_progress: Show progress bar

        Returns:
            List of ExtractionResults
        """
        logger.info(f"Batch extracting from {len(texts)} texts")

        # Create progress task
        if show_progress:
            task_id = "batch_extraction"
            self.progress_tracker.create_task(
                task_id=task_id,
                task_name="Batch Term Extraction",
                total_items=len(texts),
            )
            self.progress_tracker.start_task(task_id)

        # Process in batches
        results = []
        text_chunks = chunk_list(texts, batch_size)

        for batch_idx, batch in enumerate(text_chunks):
            batch_results = await asyncio.gather(
                *[
                    self.extract_from_text(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                        domain_path=domain_path,
                    )
                    for text in batch
                ]
            )
            results.extend(batch_results)

            if show_progress:
                self.progress_tracker.update_progress(
                    task_id=task_id,
                    completed=(batch_idx + 1) * batch_size,
                    message=f"Batch {batch_idx + 1}/{len(text_chunks)}",
                )

        if show_progress:
            self.progress_tracker.complete_task(task_id)

        logger.info(f"Batch extraction completed: {len(results)} results")
        return results

    def _parse_terms(self, terms_data: List[Dict[str, Any]]) -> List[Term]:
        """
        Parse terms from API response.

        Args:
            terms_data: List of term dictionaries

        Returns:
            List of Term objects
        """
        terms = []

        for term_dict in terms_data:
            try:
                term = Term(
                    term=term_dict.get("term", ""),
                    translation=term_dict.get("translation"),
                    domain=term_dict.get("domain", "General"),
                    subdomain=term_dict.get("subdomain"),
                    pos=term_dict.get("pos", "NOUN"),
                    definition=term_dict.get("definition", ""),
                    context=term_dict.get("context", ""),
                    relevance_score=term_dict.get("relevance_score", 0.0),
                    confidence_score=term_dict.get("confidence_score", 0.0),
                    frequency=term_dict.get("frequency", 1),
                    is_compound=term_dict.get("is_compound", False),
                    is_abbreviation=term_dict.get("is_abbreviation", False),
                    variants=term_dict.get("variants", []),
                    related_terms=term_dict.get("related_terms", []),
                )
                terms.append(term)

            except Exception as e:
                logger.warning(f"Failed to parse term: {e}")
                continue

        return terms

    async def merge_results(
        self,
        results: List[ExtractionResult],
        deduplicate: bool = True,
    ) -> ExtractionResult:
        """
        Merge multiple extraction results.

        Args:
            results: List of extraction results
            deduplicate: Remove duplicate terms

        Returns:
            Merged ExtractionResult
        """
        if not results:
            return ExtractionResult(terms=[])

        # Combine all terms
        all_terms = []
        for result in results:
            all_terms.extend(result.terms)

        # Deduplicate if requested
        if deduplicate:
            seen = set()
            unique_terms = []
            for term in all_terms:
                key = (term.term, term.translation or "")
                if key not in seen:
                    seen.add(key)
                    unique_terms.append(term)
            all_terms = unique_terms

        # Use first result's metadata as base
        base_result = results[0]

        return ExtractionResult(
            terms=all_terms,
            domain_hierarchy=base_result.domain_hierarchy,
            language_pair=base_result.language_pair,
            source_language=base_result.source_language,
            target_language=base_result.target_language,
            statistics=base_result._calculate_statistics(all_terms),
            metadata={"merged_from": len(results), "deduplicated": deduplicate},
        )
