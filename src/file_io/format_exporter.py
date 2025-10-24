"""Export extraction results to various formats."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
import pandas as pd # Ensure pandas is imported if used
from lxml import etree # Ensure lxml is imported if used

# FIXED: Corrected import relative to src/
from extraction.term_extractor import ExtractionResult, Term


class FormatExporter:
    """Export extraction results to various formats."""
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
        """Export extraction result."""
        if format_type is None:
            ext = output_path.suffix.lower()
            if not ext:
                 raise ValueError("Output path has no extension, cannot determine format.")
            format_type = ext[1:]

        logger.info(f"Exporting to {format_type}: {output_path}")

        try:
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
            logger.info(f"Export completed successfully: {output_path}")
        except Exception as e:
             logger.error(f"Export to {format_type} failed for {output_path}: {e}")
             # Re-raise or handle as appropriate for the application
             raise


    async def _export_excel(self, result: ExtractionResult, output_path: Path, include_metadata: bool) -> None:
        """Export to Excel format."""
        # ... (implementation seems okay, ensure pandas/openpyxl are in requirements) ...
        try:
            # Create DataFrame
            data = [term.to_dict() for term in result.terms] # Use to_dict method
            df = pd.DataFrame(data)

            # Select and order columns for export if needed
            columns_to_export = [
                 "term", "translation", "domain", "subdomain", "pos",
                 "definition", "context", "relevance_score", "confidence_score",
                 "frequency", "is_compound", "is_abbreviation", "variants", "related_terms"
                 # Add metadata fields if desired, converting lists/dicts to strings
            ]
            # Filter df to only include desired columns in a specific order
            df_export = df[[col for col in columns_to_export if col in df.columns]]

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                df_export.to_excel(writer, sheet_name="All Terms", index=False)

                high_relevance_terms = [t.to_dict() for t in result.get_high_relevance_terms()]
                if high_relevance_terms:
                     df_high = pd.DataFrame(high_relevance_terms)
                     df_high_export = df_high[[col for col in columns_to_export if col in df_high.columns]]
                     df_high_export.to_excel(writer, sheet_name="High Relevance", index=False)

                if include_metadata:
                    stats_data = result.statistics.copy() # Use calculated statistics
                    stats_data["Source Language"] = result.source_language
                    stats_data["Target Language"] = result.target_language or "N/A"
                    stats_data["Domain Hierarchy"] = " → ".join(result.domain_hierarchy)
                    # Add other metadata if needed
                    stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)

        except ImportError:
            logger.error("pandas or openpyxl not installed. Install with: pip install pandas openpyxl")
            raise
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise


    async def _export_csv(self, result: ExtractionResult, output_path: Path) -> None:
        """Export to CSV format."""
        # ... (implementation seems okay) ...
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                # Define fieldnames based on Term dataclass + desired order
                fieldnames = [
                    "term", "translation", "domain", "subdomain", "pos",
                    "definition", "context", "relevance_score", "confidence_score",
                    "frequency", "is_compound", "is_abbreviation", "variants", "related_terms"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore') # Ignore extra fields from to_dict
                writer.writeheader()
                for term in result.terms:
                     term_dict = term.to_dict()
                     # Convert list fields to strings for CSV compatibility
                     term_dict['variants'] = '; '.join(term_dict.get('variants', []))
                     term_dict['related_terms'] = '; '.join(term_dict.get('related_terms', []))
                     # Handle potential complex metadata if included
                     term_dict.pop('metadata', None) # Remove metadata dict for simple CSV
                     writer.writerow(term_dict)
        except Exception as e:
             logger.error(f"Error exporting to CSV: {e}")
             raise

    async def _export_tbx(self, result: ExtractionResult, output_path: Path, include_metadata: bool) -> None:
        """Export to TBX format."""
        # ... (implementation seems okay, ensure lxml is in requirements) ...
        try:
            # TBX Basic structure (adjust namespaces/structure if needed for specific TBX version/dialect)
            NSMAP = {None: "urn:iso:std:iso:30042:ed-1"} # Example TBX Basic namespace
            root = etree.Element("tbx", nsmap=NSMAP) # Use default namespace
            root.set("{http://www.w3.org/XML/1998/namespace}lang", result.source_language)

            header = etree.SubElement(root, "tbxHeader")
            fileDesc = etree.SubElement(header, "fileDesc")
            sourceDesc = etree.SubElement(fileDesc, "sourceDesc")
            etree.SubElement(sourceDesc, "p").text = f"Generated by TermExtractor on {datetime.now().isoformat()}"
            # Add encodingDesc etc. if needed

            text = etree.SubElement(root, "text")
            body = etree.SubElement(text, "body")

            for term in result.terms:
                termEntry = etree.SubElement(body, "termEntry")
                # Add unique ID if possible, otherwise use hash (less ideal for TBX)
                # termEntry.set("id", f"term-{uuid.uuid4()}")

                # Add descriptive info (like domain) using <descrip>
                etree.SubElement(termEntry, "descrip", type="subjectField").text = term.domain
                if term.subdomain:
                     etree.SubElement(termEntry, "descrip", type="subSubjectField").text = term.subdomain
                if term.definition:
                     etree.SubElement(termEntry, "descrip", type="definition").text = term.definition

                # Source Language Term
                langSet_source = etree.SubElement(termEntry, "langSet")
                langSet_source.set("{http://www.w3.org/XML/1998/namespace}lang", result.source_language)
                tig_source = etree.SubElement(langSet_source, "tig") # term information group
                etree.SubElement(tig_source, "term").text = term.term
                etree.SubElement(tig_source, "termNote", type="partOfSpeech").text = term.pos
                etree.SubElement(tig_source, "descrip", type="context").text = term.context
                # Add other metadata as termNote or descrip
                etree.SubElement(tig_source, "admin", type="relevanceScore").text = str(term.relevance_score)
                etree.SubElement(tig_source, "admin", type="confidenceScore").text = str(term.confidence_score)


                # Target Language Term
                if term.translation and result.target_language:
                    langSet_target = etree.SubElement(termEntry, "langSet")
                    langSet_target.set("{http://www.w3.org/XML/1998/namespace}lang", result.target_language)
                    tig_target = etree.SubElement(langSet_target, "tig")
                    etree.SubElement(tig_target, "term").text = term.translation
                    # Add POS/context for target if available

            tree = etree.ElementTree(root)
            tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")

        except ImportError:
            logger.error("lxml not installed. Install with: pip install lxml")
            raise
        except Exception as e:
            logger.error(f"Error exporting to TBX: {e}")
            raise

    async def _export_json(self, result: ExtractionResult, output_path: Path) -> None:
        """Export to JSON format."""
        # ... (implementation seems okay) ...
        try:
            data = result.to_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
             logger.error(f"Error exporting to JSON: {e}")
             raise
