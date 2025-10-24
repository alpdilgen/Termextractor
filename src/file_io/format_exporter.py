"""Export extraction results to various formats."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
import pandas as pd
from lxml import etree
from io import BytesIO

# Corrected import from the new data_models file
from extraction.data_models import ExtractionResult, Term


class FormatExporter:
    """
    Export ExtractionResult objects to various standard formats.
    Supported: Excel (.xlsx), CSV (.csv), TBX (.tbx), JSON (.json)
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
        Export extraction result to the specified file path and format.
        Determines format from output_path extension if not specified.
        """
        if not isinstance(result, ExtractionResult):
             logger.error(f"Invalid type passed to export: {type(result)}. Expected ExtractionResult.")
             raise TypeError("Invalid result type provided for export.")

        if not isinstance(output_path, Path):
            output_path = Path(output_path) # Ensure it's a Path object

        if format_type is None:
            ext = output_path.suffix.lower()
            if not ext:
                 raise ValueError(f"Output path '{output_path}' has no file extension, cannot determine format.")
            format_type = ext[1:] # Remove dot, e.g., 'xlsx'

        format_type = format_type.lower() # Normalize format type

        logger.info(f"Starting export of {len(result.terms)} terms to {format_type.upper()} format: {output_path}")

        try:
            # Ensure output directory exists before writing
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == "xlsx":
                await self._export_excel(result, output_path, include_metadata)
            elif format_type == "csv":
                await self._export_csv(result, output_path)
            elif format_type == "tbx":
                await self._export_tbx(result, output_path) # include_metadata handled within _export_tbx
            elif format_type == "json":
                await self._export_json(result, output_path)
            else:
                logger.error(f"Unsupported export format requested: {format_type}")
                raise ValueError(f"Unsupported export format: {format_type}")

            logger.info(f"Export completed successfully: {output_path}")

        except Exception as e:
            logger.error(f"Export to {format_type.upper()} failed for {output_path}: {e}", exc_info=True)
            # Re-raise to signal failure to the caller
            raise RuntimeError(f"Export failed: {e}") from e

    # --- Specific Format Exporters ---

    async def _export_excel(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool,
    ) -> None:
        """Export data to a multi-sheet Excel file (.xlsx)."""
        logger.debug(f"Starting Excel export to {output_path}")
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # --- All Terms Sheet ---
                if result.terms:
                    all_terms_data = [term.to_dict() for term in result.terms]
                    df_all = pd.DataFrame(all_terms_data)
                    # Define and order columns
                    std_columns = [
                        "term", "translation", "relevance_score", "confidence_score", "pos",
                        "domain", "subdomain", "definition", "context", "frequency",
                        "is_compound", "is_abbreviation", "variants", "related_terms"
                    ]
                    df_all_export = df_all[[col for col in std_columns if col in df_all.columns]].copy()
                    # Convert list columns for Excel
                    for col in ['variants', 'related_terms']:
                        if col in df_all_export.columns:
                            df_all_export[col] = df_all_export[col].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)
                    df_all_export.to_excel(writer, sheet_name="All Terms", index=False)
                    logger.debug(f"Wrote {len(df_all_export)} terms to 'All Terms' sheet.")
                else:
                    logger.warning("No terms to write to 'All Terms' sheet.")
                    # Optionally create an empty sheet with headers
                    pd.DataFrame(columns=std_columns).to_excel(writer, sheet_name="All Terms", index=False)


                # --- High Relevance Sheet ---
                high_relevance_terms = result.get_high_relevance_terms() # Uses default >= 80
                if high_relevance_terms:
                    df_high_data = [term.to_dict() for term in high_relevance_terms]
                    df_high = pd.DataFrame(df_high_data)
                    df_high_export = df_high[[col for col in std_columns if col in df_high.columns]].copy()
                    for col in ['variants', 'related_terms']:
                         if col in df_high_export.columns:
                              df_high_export[col] = df_high_export[col].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)
                    df_high_export.to_excel(writer, sheet_name="High Relevance (>=80)", index=False)
                    logger.debug(f"Wrote {len(df_high_export)} terms to 'High Relevance (>=80)' sheet.")

                # --- Statistics Sheet ---
                if include_metadata:
                    stats_data = result.statistics.copy() if result.statistics else {}
                    stats_data["Source Language"] = result.source_language
                    stats_data["Target Language"] = result.target_language or "N/A"
                    stats_data["Domain Hierarchy"] = " → ".join(result.domain_hierarchy) if result.domain_hierarchy else "N/A"
                    # Add API metadata if it exists in result.metadata
                    if result.metadata:
                        stats_data.update({f"API_{k}": v for k, v in result.metadata.items() if k != 'error'})
                    stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                    logger.debug("Wrote Statistics sheet.")

        except ImportError:
            logger.error("Excel export requires `pandas` and `openpyxl`. Install with `pip install pandas openpyxl`.")
            raise ImportError("pandas and openpyxl are required for Excel export.")
        except Exception as e:
            logger.error(f"Failed during Excel export process: {e}", exc_info=True)
            raise


    async def _export_csv(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export terms to a CSV file."""
        logger.debug(f"Starting CSV export to {output_path}")
        # Define the exact columns and order for the CSV
        fieldnames = [
            "term", "translation", "relevance_score", "confidence_score", "pos",
            "domain", "subdomain", "definition", "context", "frequency",
            "is_compound", "is_abbreviation", "variants", "related_terms"
        ]
        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()

                if not result.terms:
                     logger.warning("No terms found to write to CSV file.")
                     return # File with only header is created

                for term in result.terms:
                    term_dict = term.to_dict()
                    # Prepare row for CSV: handle None, convert lists
                    row_data = {}
                    for field in fieldnames:
                         value = term_dict.get(field)
                         if isinstance(value, list):
                              row_data[field] = '; '.join(map(str, value)) # Convert lists to string
                         elif value is None:
                              row_data[field] = '' # Use empty string for None
                         else:
                              row_data[field] = value
                    writer.writerow(row_data)

        except Exception as e:
            logger.error(f"Failed during CSV export: {e}", exc_info=True)
            raise


    async def _export_tbx(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool, # Currently unused, but kept for signature consistency
    ) -> None:
        """Export terms to TBX-Basic format (.tbx)."""
        logger.debug(f"Starting TBX export to {output_path}")
        try:
            XML_LANG = "{[http://www.w3.org/XML/1998/namespace](http://www.w3.org/XML/1998/namespace)}lang" # XML namespace for lang attribute
            # Use TBX-Basic namespace by default
            NSMAP = {None: "urn:iso:std:iso:30042:ed-1"}
            root = etree.Element("tbx", nsmap=NSMAP, attrib={XML_LANG: result.source_language or 'zxx'}) # Use 'zxx' if lang unknown

            # --- Header ---
            header = etree.SubElement(root, "tbxHeader")
            fileDesc = etree.SubElement(header, "fileDesc")
            titleStmt = etree.SubElement(fileDesc, "titleStmt")
            etree.SubElement(titleStmt, "title").text = f"TermExtractor Export - {datetime.now().date()}"
            sourceDesc = etree.SubElement(fileDesc, "sourceDesc")
            etree.SubElement(sourceDesc, "p").text = f"Generated by TermExtractor on {datetime.now().isoformat()}"
            encodingDesc = etree.SubElement(header, "encodingDesc")
            # Point to TBX-Basic dialect XCS file (standard practice)
            etree.SubElement(encodingDesc, "p", type="XCSURI").text = "TBX-Basic.xcs"

            # --- Body ---
            text_el = etree.SubElement(root, "text")
            body = etree.SubElement(text_el, "body")

            if not result.terms:
                 logger.warning("No terms found to write to TBX file.")
                 # Write empty body to create a valid minimal TBX
            else:
                for term_obj in result.terms:
                    if not term_obj.term: continue # Skip terms with empty text

                    termEntry = etree.SubElement(body, "termEntry")

                    # Concept Level Information (e.g., domain, definition)
                    descripGrp_concept = etree.SubElement(termEntry, "descripGrp")
                    etree.SubElement(descripGrp_concept, "descrip", type="subjectField").text = term_obj.domain or "General"
                    if term_obj.subdomain:
                        etree.SubElement(descripGrp_concept, "descrip", type="subSubjectField").text = term_obj.subdomain
                    if term_obj.definition:
                        etree.SubElement(descripGrp_concept, "descrip", type="definition").text = term_obj.definition

                    # Source Language Term Info
                    langSet_source = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.source_language})
                    tig_source = etree.SubElement(langSet_source, "tig") # Term Information Group
                    etree.SubElement(tig_source, "term").text = term_obj.term
                    etree.SubElement(tig_source, "termNote", type="partOfSpeech").text = term_obj.pos or "unknown"
                    if term_obj.context:
                        etree.SubElement(tig_source, "descrip", type="context").text = term_obj.context

                    # Add custom TermExtractor scores using adminGrp (or descripGrp if preferred)
                    adminGrp_source = etree.SubElement(tig_source, "adminGrp")
                    etree.SubElement(adminGrp_source, "admin", type="termExtractor-relevanceScore").text = f"{term_obj.relevance_score:.2f}"
                    etree.SubElement(adminGrp_source, "admin", type="termExtractor-confidenceScore").text = f"{term_obj.confidence_score:.2f}"
                    etree.SubElement(adminGrp_source, "admin", type="termExtractor-frequency").text = str(term_obj.frequency)
                    # Add boolean flags if needed
                    # etree.SubElement(adminGrp_source, "admin", type="termExtractor-isCompound").text = str(term_obj.is_compound).lower()

                    # Target Language Term Info (if available)
                    if term_obj.translation and result.target_language:
                        langSet_target = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.target_language})
                        tig_target = etree.SubElement(langSet_target, "tig")
                        etree.SubElement(tig_target, "term").text = term_obj.translation
                        # Optionally add target POS if available

            # Write the complete XML tree to the file
            tree = etree.ElementTree(root)
            with open(output_path, 'wb') as f: # Open in binary mode for write_c14n
                 # Use write_c14n for canonical XML or standard write
                 tree.write(f, pretty_print=True, xml_declaration=True, encoding="UTF-8")

        except ImportError:
            logger.error("`lxml` library not found. Install with `pip install lxml` for TBX export.")
            raise ImportError("lxml is required for TBX export.")
        except Exception as e:
            logger.error(f"Failed during TBX export: {e}", exc_info=True)
            raise

    async def _export_json(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export the full ExtractionResult object to JSON format."""
        logger.debug(f"Starting JSON export to {output_path}")
        try:
            # Use the to_dict method of ExtractionResult for serialization
            data_to_export = result.to_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_to_export, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed during JSON export: {e}", exc_info=True)
            raise
