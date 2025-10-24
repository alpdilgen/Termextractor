"""Export extraction results to various formats."""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger
import pandas as pd
from lxml import etree
from io import BytesIO # Needed for in-memory Excel/TBX generation

# Corrected import from the new data_models file
from extraction.data_models import ExtractionResult, Term


class FormatExporter:
    """
    Export extraction results to various formats.
    Supported formats: Excel (.xlsx), CSV (.csv), TBX (.tbx), JSON (.json)
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
        """
        if not isinstance(result, ExtractionResult):
             logger.error("Invalid result type passed to export. Expected ExtractionResult.")
             raise TypeError("Invalid result type provided for export.")

        if format_type is None:
            ext = output_path.suffix.lower()
            if not ext:
                raise ValueError("Output path has no extension, cannot determine format.")
            format_type = ext[1:] # Remove dot

        logger.info(f"Starting export to {format_type.upper()} format: {output_path}")

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

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
            logger.error(f"Export to {format_type.upper()} failed for {output_path}: {e}", exc_info=True)
            # Re-raise the exception to signal failure
            raise

    async def _export_excel(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool,
    ) -> None:
        """Export to Excel format (.xlsx)."""
        if not result.terms:
             logger.warning("No terms found in result, creating Excel file with only stats (if applicable).")
             # Still create the file but maybe only with stats sheet if include_metadata is True
             if not include_metadata: return # Do nothing if no terms and no metadata

        try:
            # Prepare data for DataFrame using Term.to_dict()
            all_terms_data = [term.to_dict() for term in result.terms]
            df_all = pd.DataFrame(all_terms_data)

            # Define standard columns order for consistency
            std_columns = [
                 "term", "translation", "domain", "subdomain", "pos", "definition",
                 "context", "relevance_score", "confidence_score", "frequency",
                 "is_compound", "is_abbreviation", "variants", "related_terms"
            ]
            # Filter DataFrame to include only standard columns that exist
            df_all_export = df_all[[col for col in std_columns if col in df_all.columns]].copy()
            # Convert list columns to strings for better Excel compatibility
            for col in ['variants', 'related_terms']:
                 if col in df_all_export.columns:
                      df_all_export[col] = df_all_export[col].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)


            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Sheet 1: All Terms
                df_all_export.to_excel(writer, sheet_name="All Terms", index=False)
                logger.debug(f"Wrote {len(df_all_export)} terms to 'All Terms' sheet.")

                # Sheet 2: High Relevance Terms (if any)
                high_relevance_terms = result.get_high_relevance_terms()
                if high_relevance_terms:
                    df_high_data = [term.to_dict() for term in high_relevance_terms]
                    df_high = pd.DataFrame(df_high_data)
                    df_high_export = df_high[[col for col in std_columns if col in df_high.columns]].copy()
                    for col in ['variants', 'related_terms']:
                         if col in df_high_export.columns:
                              df_high_export[col] = df_high_export[col].apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) else x)
                    df_high_export.to_excel(writer, sheet_name="High Relevance", index=False)
                    logger.debug(f"Wrote {len(df_high_export)} terms to 'High Relevance' sheet.")

                # Sheet 3: Statistics and Metadata
                if include_metadata:
                    stats_data = result.statistics.copy() if result.statistics else {}
                    stats_data["Source Language"] = result.source_language
                    stats_data["Target Language"] = result.target_language or "N/A"
                    stats_data["Domain Hierarchy"] = " → ".join(result.domain_hierarchy) if result.domain_hierarchy else "N/A"
                    # Add API metadata if it exists
                    if result.metadata:
                        stats_data.update({f"API_{k}": v for k, v in result.metadata.items() if k != 'error'}) # Exclude potential error message

                    stats_df = pd.DataFrame(list(stats_data.items()), columns=['Metric', 'Value'])
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)
                    logger.debug("Wrote Statistics sheet.")

        except ImportError:
            logger.error("Required libraries `pandas` or `openpyxl` not found for Excel export.")
            raise ImportError("pandas and openpyxl are required for Excel export.")
        except Exception as e:
            logger.error(f"Failed during Excel export: {e}", exc_info=True)
            raise

    async def _export_csv(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export to CSV format."""
        if not result.terms:
             logger.warning("No terms found in result, creating empty CSV file with headers.")
             # Create empty file with header
             with open(output_path, "w", newline="", encoding="utf-8") as f:
                  fieldnames = ["term", "translation", "domain", "subdomain", "pos", "definition", "context",
                                "relevance_score", "confidence_score", "frequency"]
                  writer = csv.DictWriter(f, fieldnames=fieldnames)
                  writer.writeheader()
             return

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                # Define fieldnames based on desired CSV output
                fieldnames = [
                    "term", "translation", "domain", "subdomain", "pos", "definition",
                    "context", "relevance_score", "confidence_score", "frequency",
                    # Add others if needed, like variants, related_terms as strings
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore') # Ignore extra fields
                writer.writeheader()

                for term in result.terms:
                    term_dict = term.to_dict()
                    # Prepare for CSV: simplify complex fields if necessary
                    term_dict['translation'] = term_dict.get('translation', '') or '' # Ensure empty string, not None
                    term_dict['subdomain'] = term_dict.get('subdomain', '') or ''
                    # Convert lists to semicolon-separated strings if including them
                    # term_dict['variants'] = '; '.join(term_dict.get('variants', []))
                    # term_dict['related_terms'] = '; '.join(term_dict.get('related_terms', []))
                    writer.writerow(term_dict)

        except Exception as e:
            logger.error(f"Failed during CSV export: {e}", exc_info=True)
            raise


    async def _export_tbx(
        self,
        result: ExtractionResult,
        output_path: Path,
        include_metadata: bool, # Parameter kept for consistency, can add more metadata if True
    ) -> None:
        """Export to TBX format (TBX-Basic)."""
        if not result.terms:
             logger.warning("No terms found in result, creating minimal TBX file.")
             # Create minimal valid TBX structure
             root = etree.Element("martif", type="TBX", attrib={"{http://www.w3.org/XML/1998/namespace}lang": result.source_language or 'en'})
             header = etree.SubElement(root, "martifHeader")
             fileDesc = etree.SubElement(header, "fileDesc")
             sourceDesc = etree.SubElement(fileDesc, "sourceDesc")
             etree.SubElement(sourceDesc, "p").text = f"Generated by TermExtractor (No Terms Found) on {datetime.now().isoformat()}"
             text_el = etree.SubElement(root, "text")
             etree.SubElement(text_el, "body") # Empty body
             tree = etree.ElementTree(root)
             try:
                  tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")
             except Exception as write_e:
                  logger.error(f"Failed to write minimal TBX file: {write_e}", exc_info=True)
                  raise
             return

        try:
            # TBX Basic structure
            XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
            root = etree.Element("martif", type="TBX", attrib={XML_LANG: result.source_language or 'en'})

            header = etree.SubElement(root, "martifHeader")
            fileDesc = etree.SubElement(header, "fileDesc")
            titleStmt = etree.SubElement(fileDesc, "titleStmt")
            etree.SubElement(titleStmt, "title").text = f"Terminology export from TermExtractor - {datetime.now().date()}"
            sourceDesc = etree.SubElement(fileDesc, "sourceDesc")
            etree.SubElement(sourceDesc, "p").text = f"Generated by TermExtractor on {datetime.now().isoformat()}"
            # Add encodingDesc
            encodingDesc = etree.SubElement(header, "encodingDesc")
            etree.SubElement(encodingDesc, "p", type="XCSURI").text = "TBX-Basic.xcs" # Indicate TBX-Basic dialect

            text_el = etree.SubElement(root, "text")
            body = etree.SubElement(text_el, "body")

            for term_obj in result.terms:
                termEntry = etree.SubElement(body, "termEntry")

                # --- Concept Level Info (descripGrp if needed) ---
                descripGrp_concept = etree.SubElement(termEntry, "descripGrp")
                etree.SubElement(descripGrp_concept, "descrip", type="subjectField").text = term_obj.domain or "General"
                if term_obj.subdomain:
                    etree.SubElement(descripGrp_concept, "descrip", type="subSubjectField").text = term_obj.subdomain
                if term_obj.definition:
                    etree.SubElement(descripGrp_concept, "descrip", type="definition").text = term_obj.definition

                # --- Source Language Term ---
                langSet_source = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.source_language})
                tig_source = etree.SubElement(langSet_source, "tig") # term information group
                etree.SubElement(tig_source, "term").text = term_obj.term
                etree.SubElement(tig_source, "termNote", type="partOfSpeech").text = term_obj.pos or "unknown"
                if term_obj.context:
                     etree.SubElement(tig_source, "descrip", type="context").text = term_obj.context
                # Add scores as adminGrp or descripGrp within tig
                adminGrp_source = etree.SubElement(tig_source, "adminGrp")
                etree.SubElement(adminGrp_source, "admin", type="termExtractor-relevanceScore").text = str(term_obj.relevance_score)
                etree.SubElement(adminGrp_source, "admin", type="termExtractor-confidenceScore").text = str(term_obj.confidence_score)
                etree.SubElement(adminGrp_source, "admin", type="termExtractor-frequency").text = str(term_obj.frequency)


                # --- Target Language Term (if available) ---
                if term_obj.translation and result.target_language:
                    langSet_target = etree.SubElement(termEntry, "langSet", attrib={XML_LANG: result.target_language})
                    tig_target = etree.SubElement(langSet_target, "tig")
                    etree.SubElement(tig_target, "term").text = term_obj.translation
                    # Add target POS etc. if available


            # Write the tree to the file
            tree = etree.ElementTree(root)
            tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding="UTF-8")

        except ImportError:
            logger.error("`lxml` library not found. Please install it for TBX export (`pip install lxml`).")
            raise ImportError("lxml is required for TBX export.")
        except Exception as e:
            logger.error(f"Failed during TBX export: {e}", exc_info=True)
            raise


    async def _export_json(
        self,
        result: ExtractionResult,
        output_path: Path,
    ) -> None:
        """Export to JSON format."""
        try:
            data = result.to_dict() # Use the existing method on ExtractionResult
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed during JSON export: {e}", exc_info=True)
            raise
