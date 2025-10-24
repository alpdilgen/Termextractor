"""Export extraction results to various formats."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

from termextractor.extraction.term_extractor import ExtractionResult, Term


class FormatExporter:
    """
    Export extraction results to various formats.

    Supported formats:
    - Excel (.xlsx)
    - CSV (.csv)
    - TBX (.tbx)
    - JSON (.json)
    """

    def __init__(self):
        """Initialize FormatExporter."""
        logger.info("FormatExporter initialized")

    async def export(
        self,
        result: ExtractionResult,
        output_path: Path,
        format_type: Optional[str] = None,
        include_metadata: bool = True,
    ) -> None:
        """
        Export extraction result.

        Args:
            result: ExtractionResult to export
            output_path: Output file path
            format_type: Format type (auto-detect from extension if None)
            include_metadata: Include metadata in export

        Raises:
            ValueError: If format is not supported
        """
        if format_type is None:
            # Auto-detect from extension
            ext = output_path.suffix.lower()
            format_type = ext[1:]  # Remove dot

        logger.info(f"Exporting to {format_type}: {output_path}")

        if format_type == "xlsx":
            await self._export_excel(result, output_path, include_metadata)
        elif format_type == "csv":
            await self._export_csv(result, output_path)
        elif format_type == "tbx":
            await self._export_tbx(result, output_path, include_metadata)
        elif format_type == "json":
            await self._export_json(result, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

        logger.info(f"Export completed: {output_path}")

    async def _export_excel(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool,
    ) -> None:
        """Export to Excel format."""
        try:
            import pandas as pd

            # Create DataFrame
            data = []
            for term in result.terms:
                row = {
                    "Term": term.term,
                    "Translation": term.translation or "",
                    "Domain": term.domain,
                    "Subdomain": term.subdomain or "",
                    "Part of Speech": term.pos,
                    "Definition": term.definition,
                    "Context": term.context,
                    "Relevance Score": term.relevance_score,
                    "Confidence Score": term.confidence_score,
                    "Frequency": term.frequency,
                }
                data.append(row)

            df = pd.DataFrame(data)

            # Create Excel writer
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # All terms sheet
                df.to_excel(writer, sheet_name="All Terms", index=False)

                # High relevance terms sheet
                high_relevance = df[df["Relevance Score"] >= 80]
                high_relevance.to_excel(
                    writer, sheet_name="High Relevance", index=False
                )

                # Statistics sheet
                if include_metadata:
                    stats_data = {
                        "Metric": [
                            "Total Terms",
                            "High Relevance",
                            "Medium Relevance",
                            "Low Relevance",
                            "Source Language",
                            "Target Language",
                            "Domain",
                        ],
                        "Value": [
                            len(result.terms),
                            len([t for t in result.terms if t.relevance_score >= 80]),
                            len(
                                [
                                    t
                                    for t in result.terms
                                    if 60 <= t.relevance_score < 80
                                ]
                            ),
                            len([t for t in result.terms if t.relevance_score < 60]),
                            result.source_language,
                            result.target_language or "N/A",
                            " → ".join(result.domain_hierarchy),
                        ],
                    }
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        except ImportError:
            logger.error("pandas/openpyxl not installed. Install with: pip install pandas openpyxl")
            raise

    async def _export_csv(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export to CSV format."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Term",
                "Translation",
                "Domain",
                "Subdomain",
                "Part of Speech",
                "Definition",
                "Context",
                "Relevance Score",
                "Confidence Score",
                "Frequency",
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for term in result.terms:
                writer.writerow(
                    {
                        "Term": term.term,
                        "Translation": term.translation or "",
                        "Domain": term.domain,
                        "Subdomain": term.subdomain or "",
                        "Part of Speech": term.pos,
                        "Definition": term.definition,
                        "Context": term.context,
                        "Relevance Score": term.relevance_score,
                        "Confidence Score": term.confidence_score,
                        "Frequency": term.frequency,
                    }
                )

    async def _export_tbx(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool,
    ) -> None:
        """Export to TBX format."""
        try:
            from lxml import etree

            # Create TBX root
            root = etree.Element("martif")
            root.set("type", "TBX")
            root.set("xml:lang", result.source_language)

            # Header
            header = etree.SubElement(root, "martifHeader")
            file_desc = etree.SubElement(header, "fileDesc")

            source_desc = etree.SubElement(file_desc, "sourceDesc")
            p = etree.SubElement(source_desc, "p")
            p.text = f"Generated by TermExtractor on {datetime.now().isoformat()}"

            # Body
            text = etree.SubElement(root, "text")
            body = etree.SubElement(text, "body")

            # Add terms
            for term in result.terms:
                term_entry = etree.SubElement(body, "termEntry")
                term_entry.set("id", f"term_{hash(term.term)}")

                # Domain
                descrip = etree.SubElement(term_entry, "descrip")
                descrip.set("type", "subjectField")
                descrip.text = term.domain

                # Source language term
                lang_set = etree.SubElement(term_entry, "langSet")
                lang_set.set("xml:lang", result.source_language)

                tig = etree.SubElement(lang_set, "tig")
                term_elem = etree.SubElement(tig, "term")
                term_elem.text = term.term

                # POS
                term_note = etree.SubElement(tig, "termNote")
                term_note.set("type", "partOfSpeech")
                term_note.text = term.pos

                # Target language term (if bilingual)
                if term.translation and result.target_language:
                    lang_set_tgt = etree.SubElement(term_entry, "langSet")
                    lang_set_tgt.set("xml:lang", result.target_language)

                    tig_tgt = etree.SubElement(lang_set_tgt, "tig")
                    term_elem_tgt = etree.SubElement(tig_tgt, "term")
                    term_elem_tgt.text = term.translation

            # Write to file
            tree = etree.ElementTree(root)
            tree.write(
                str(output_path),
                pretty_print=True,
                xml_declaration=True,
                encoding="UTF-8",
            )

        except ImportError:
            logger.error("lxml not installed. Install with: pip install lxml")
            raise

    async def _export_json(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export to JSON format."""
        data = result.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
