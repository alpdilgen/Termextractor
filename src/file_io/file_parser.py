"""File parser for various document formats."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Union # Added Union
from loguru import logger
import docx2txt # Added
from lxml import etree # Added
import PyPDF2 # Added
from bs4 import BeautifulSoup # Added

# Corrected import relative to src/
from utils.helpers import get_file_extension


class FileParser:
    """
    Parse various document formats to extract text content.
    Handles: TXT, DOCX, PDF, HTML, XML, XLIFF variants.
    """
    def __init__(self):
        """Initialize FileParser."""
        logger.info("FileParser initialized")

    async def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse file based on extension and return text content."""
        if not isinstance(file_path, Path): # Ensure input is Path object
            file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = get_file_extension(file_path) # Uses helper
        logger.info(f"Attempting to parse file: {file_path.name} (type: {ext})")

        parsed_data: Dict[str, Any] = {
            "text": "",
            "format": ext.replace(".", ""),
            "metadata": {"file_name": file_path.name, "file_size": file_path.stat().st_size}
        }

        try:
            if ext == ".txt":
                parsed_data = await self._parse_txt(file_path)
            elif ext == ".docx":
                parsed_data = await self._parse_docx(file_path)
            elif ext == ".pdf":
                parsed_data = await self._parse_pdf(file_path)
            elif ext in [".html", ".htm"]:
                parsed_data = await self._parse_html(file_path)
            elif ext == ".xml":
                # Note: Generic XML parsing. Might need specific handlers for known XML types.
                parsed_data = await self._parse_xml(file_path)
            elif ext in [".xliff", ".sdlxliff", ".mqxliff", ".xlf"]:
                 parsed_data = await self._parse_xliff(file_path)
            # --- Add other format handlers here ---
            # elif ext == ".doc": parsed_data = await self._parse_doc(file_path) # Needs implementation
            # elif ext == ".rtf": parsed_data = await self._parse_rtf(file_path) # Needs implementation
            # elif ext == ".tmx": parsed_data = await self._parse_tmx(file_path) # Needs implementation
            # elif ext == ".ttx": parsed_data = await self._parse_ttx(file_path) # Needs implementation
            else:
                raise ValueError(f"Unsupported file format: {ext}")

            # Ensure text is string and add char_count if missing
            if "text" not in parsed_data or not isinstance(parsed_data["text"], str):
                 parsed_data["text"] = "" # Default to empty string if parsing failed internally
                 logger.warning(f"Parsing method for {ext} did not return valid text.")
            parsed_data["metadata"]["char_count"] = len(parsed_data["text"])
            logger.info(f"Successfully parsed {file_path.name}. Extracted {parsed_data['metadata']['char_count']} characters.")

        except Exception as e:
             logger.error(f"Failed to parse {file_path.name} due to error: {e}", exc_info=True)
             parsed_data["text"] = "" # Ensure text is empty on error
             parsed_data["metadata"]["error"] = f"Parsing failed: {e}"
             # Do not re-raise here, return the dict with error info

        return parsed_data


    async def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """Parse plain text file, trying common encodings."""
        text = None
        detected_encoding = None
        for enc in ['utf-8', 'latin-1', 'cp1252']: # Common encodings
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                detected_encoding = enc
                logger.debug(f"Read TXT file {file_path.name} with encoding: {enc}")
                break # Success
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode TXT {file_path.name} with {enc}, trying next.")
            except Exception as e:
                 logger.error(f"Error reading TXT file {file_path.name}: {e}", exc_info=True)
                 raise ValueError(f"Could not read TXT file {file_path.name}") from e # Re-raise other read errors

        if text is None:
             raise ValueError(f"Could not decode TXT file {file_path.name} with attempted encodings.")

        return {
            "text": text.strip(), # Remove leading/trailing whitespace
            "format": "txt",
            "metadata": {"file_name": file_path.name, "file_size": file_path.stat().st_size, "encoding": detected_encoding},
        }


    async def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Parse Microsoft Word document using docx2txt."""
        try:
            text = docx2txt.process(file_path)
            text = text.strip() if text else "" # Ensure string, strip whitespace
            logger.info(f"Successfully extracted {len(text)} chars using docx2txt from {file_path.name}.")
            return {
                "text": text,
                "format": "docx",
                "metadata": {"file_name": file_path.name, "file_size": file_path.stat().st_size},
            }
        except Exception as e:
            logger.error(f"Error parsing DOCX {file_path.name} with docx2txt: {e}", exc_info=True)
            raise ValueError(f"Failed to parse DOCX file {file_path.name}: {e}") from e


    async def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF document using PyPDF2."""
        text_list = []
        num_pages = 0
        pdf_reader = None
        try:
            with open(file_path, "rb") as f:
                # strict=False helps with some malformed PDFs
                pdf_reader = PyPDF2.PdfReader(f, strict=False)
                num_pages = len(pdf_reader.pages)
                logger.debug(f"Opened PDF {file_path.name} with {num_pages} pages.")
                for i, page in enumerate(pdf_reader.pages):
                     try:
                          page_text = page.extract_text()
                          if page_text:
                               text_list.append(page_text.strip())
                     except Exception as page_e:
                          logger.warning(f"Could not extract text from page {i+1} of PDF {file_path.name}: {page_e}")
            combined_text = "\n\n".join(text_list).strip() # Join pages with double newline
            logger.info(f"Extracted {len(combined_text)} chars from PDF {file_path.name}.")
            return {
                "text": combined_text, "format": "pdf",
                "metadata": {"file_name": file_path.name, "pages": num_pages, "file_size": file_path.stat().st_size},
            }
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise ImportError("PyPDF2 is required for PDF parsing.")
        except PyPDF2.errors.PdfReadError as pdf_err: # Catch specific PyPDF2 errors
            logger.error(f"PyPDF2 could not read PDF {file_path.name}: {pdf_err}")
            raise ValueError(f"Failed to read PDF file (possibly corrupted or encrypted): {file_path.name}") from pdf_err
        except Exception as e:
            logger.error(f"Unexpected error parsing PDF {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse PDF file {file_path.name}") from e


    async def _parse_html(self, file_path: Path) -> Dict[str, Any]:
        """Parse HTML document using BeautifulSoup."""
        html_content = None
        detected_encoding = None
        # Try reading with common encodings
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    html_content = f.read()
                detected_encoding = enc
                logger.debug(f"Read HTML file {file_path.name} with encoding: {enc}")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode HTML {file_path.name} with {enc}, trying next.")
            except Exception as e:
                 logger.error(f"Error reading HTML file {file_path.name}: {e}", exc_info=True)
                 raise ValueError(f"Could not read HTML file {file_path.name}") from e

        if html_content is None:
             raise ValueError(f"Could not decode HTML file {file_path.name} with attempted encodings.")

        try:
            # Use lxml for speed if available, fallback to html.parser
            soup = BeautifulSoup(html_content, "lxml")
            # Remove script, style, and often noisy header/footer/nav elements
            for element in soup(["script", "style", "header", "footer", "nav", "aside", "form"]):
                element.decompose()
            # Get text, using separator for block elements, strip extra whitespace
            text = soup.get_text(separator='\n', strip=True)
            logger.info(f"Extracted {len(text)} chars from HTML {file_path.name}.")
            return {
                "text": text, "format": "html",
                "metadata": {"file_name": file_path.name, "encoding": detected_encoding, "file_size": file_path.stat().st_size},
            }
        except ImportError:
            logger.error("Required libraries `beautifulsoup4` or `lxml` not found.")
            raise ImportError("BeautifulSoup4 and lxml are required for HTML parsing.")
        except Exception as e:
            logger.error(f"Error parsing HTML {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse HTML file {file_path.name}") from e


    async def _parse_xml(self, file_path: Path) -> Dict[str, Any]:
        """Parse generic XML document using lxml, extracting all text content."""
        try:
            # Use recover=True to handle minor XML errors
            parser = etree.XMLParser(recover=True, ns_clean=True, remove_comments=True, remove_pis=True)
            tree = etree.parse(str(file_path), parser)
            root = tree.getroot()
            # Extract text content from all elements, joined by spaces
            # Use XPath '//text()' to get all text nodes, strip whitespace, join non-empty
            text_nodes = [text.strip() for text in root.xpath('//text()') if text.strip()]
            text = "\n".join(text_nodes).strip() # Join with newlines
            logger.info(f"Extracted {len(text)} chars from generic XML {file_path.name}.")
            return {
                "text": text, "format": "xml",
                "metadata": {"file_name": file_path.name, "root_tag": root.tag, "file_size": file_path.stat().st_size},
            }
        except ImportError:
            logger.error("`lxml` library not found. Install with `pip install lxml`.")
            raise ImportError("lxml is required for XML parsing.")
        except etree.XMLSyntaxError as e:
            logger.error(f"Invalid XML syntax in {file_path.name}: {e}")
            raise ValueError(f"Invalid XML structure in {file_path.name}") from e
        except Exception as e:
            logger.error(f"Error parsing generic XML {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse generic XML file {file_path.name}") from e


    async def _parse_xliff(self, file_path: Path) -> Dict[str, Any]:
        """Parse XLIFF variants (.xliff, .sdlxliff, .mqxliff), extracting source segments."""
        try:
            namespaces = { # Define common XLIFF namespaces
                'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
                'sdl': '[http://sdl.com/FileTypes/SdlXliff/1.0](http://sdl.com/FileTypes/SdlXliff/1.0)',
                'mq': 'MQXliff' # Replace with actual URI if known
                # Add others if needed: e.g., 'mda': 'urn:oasis:names:tc:xliff:metadata:1.0'
            }
            # Use recover mode to handle potential minor issues
            parser = etree.XMLParser(recover=True, ns_clean=True, remove_comments=True)
            tree = etree.parse(str(file_path), parser)
            root = tree.getroot()

            # --- Attempt different XPath queries to find source text ---
            # Prioritize standard XLIFF, then SDL-specific, then namespace-agnostic
            xpath_queries = [
                './/xliff:trans-unit/xliff:source', # XLIFF 1.2 standard
                './/sdl:seg-source', # SDLXLIFF (often within trans-unit or group)
                './/trans-unit/source', # No namespace specified
                # Add specific paths for MQXLIFF if known, e.g. './/mq:trans-unit/mq:source'
            ]

            source_elements = []
            unique_element_ids = set() # To avoid duplicates if multiple queries match same element

            for query in xpath_queries:
                try:
                    elements = root.xpath(query, namespaces=namespaces)
                    for elem in elements:
                        # Use element's internal ID or hash to track uniqueness
                        elem_id = elem.get('id') or hash(etree.tostring(elem))
                        if elem_id not in unique_element_ids:
                             source_elements.append(elem)
                             unique_element_ids.add(elem_id)
                except etree.XPathEvalError as e:
                    logger.warning(f"XPath query failed for {file_path.name}: {query} - Error: {e}")

            if not source_elements:
                logger.warning(f"No source text elements found in {file_path.name} using common XPaths. Check file structure/namespaces.")
                extracted_text = ""
            else:
                # Extract text content using itertext() to handle internal tags, join segments with newline
                segment_texts = [''.join(elem.itertext()).strip() for elem in source_elements]
                extracted_text = "\n".join(filter(None, segment_texts)) # Filter out empty strings after stripping

            logger.info(f"Extracted {len(extracted_text)} chars ({len(source_elements)} source segments) from XLIFF: {file_path.name}")

            return {
                "text": extracted_text,
                "format": file_path.suffix.replace(".", ""), # Specific format like 'sdlxliff'
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "segment_count": len(source_elements)
                },
            }
        except ImportError:
             logger.error("`lxml` library not found. Install with `pip install lxml`.")
             raise ImportError("lxml is required for XLIFF parsing.")
        except etree.XMLSyntaxError as e:
            logger.error(f"Invalid XML syntax in {file_path.name}: {e}")
            raise ValueError(f"Invalid XML structure in {file_path.name}") from e
        except Exception as e:
            logger.error(f"Error parsing XLIFF {file_path.name}: {e}", exc_info=True)
            raise ValueError(f"Failed to parse XLIFF file {file_path.name}") from e

    # --- Add Placeholder Methods for Other Formats ---
    # async def _parse_doc(self, file_path: Path) -> Dict[str, Any]:
    #     """Parse legacy Word document (.doc). Requires external tools like antiword or catdoc."""
    #     logger.warning("Parsing .doc files is complex and may require external tools not available here.")
    #     raise NotImplementedError("Legacy .doc parsing requires external tools or specific libraries.")

    # async def _parse_rtf(self, file_path: Path) -> Dict[str, Any]:
    #     """Parse Rich Text Format (.rtf)."""
    #     try:
    #         from striprtf.striprtf import rtf_to_text
    #         with open(file_path, "r") as f: # Determine correct encoding if needed
    #              rtf_content = f.read()
    #         text = rtf_to_text(rtf_content)
    #         return {"text": text.strip(), "format": "rtf", "metadata": {...}}
    #     except ImportError:
    #         logger.error("`striprtf` needed. Install with `pip install striprtf`")
    #         raise ImportError("striprtf required for RTF.")
    #     except Exception as e: ...

    # async def _parse_tmx(self, file_path: Path) -> Dict[str, Any]:
    #      """Parse TMX file (extracts source text from <seg> elements)."""
    #      # Adapt _parse_xliff, changing XPath to './/tu/tuv[@xml:lang="SOURCE_LANG"]/seg'
    #      # Need to determine source language or extract all <seg>
    #      raise NotImplementedError("TMX parsing not implemented.")

    # async def _parse_ttx(self, file_path: Path) -> Dict[str, Any]:
    #      """Parse TTX file (Trados TagEditor)."""
    #      # Requires specific parsing logic for TTX structure
    #      raise NotImplementedError("TTX parsing not implemented.")
