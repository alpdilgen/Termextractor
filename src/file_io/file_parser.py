"""File parser for various document formats."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
import docx2txt # Added
from lxml import etree # Added

# FIXED: Corrected import relative to src/
from utils.helpers import get_file_extension


class FileParser:
    """
    Parse various document formats to extract text.
    """
    def __init__(self):
        """Initialize FileParser."""
        logger.info("FileParser initialized")

    async def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse file and extract text content."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = get_file_extension(file_path)
        logger.info(f"Parsing file: {file_path} ({ext})")

        try:
            if ext == ".txt":
                return await self._parse_txt(file_path)
            elif ext == ".docx":
                # FIXED: Now uses the updated _parse_docx with docx2txt
                return await self._parse_docx(file_path)
            elif ext == ".pdf":
                return await self._parse_pdf(file_path)
            elif ext in [".html", ".htm"]:
                return await self._parse_html(file_path)
            elif ext == ".xml":
                 # Consider if a more specific XML parser is needed for certain types
                return await self._parse_xml(file_path)
            # FIXED: Added conditions for XLIFF variants
            elif ext in [".xliff", ".sdlxliff", ".mqxliff", ".xlf"]:
                 return await self._parse_xliff(file_path)
            # Add other format handlers here
            # elif ext == ".doc": return await self._parse_doc(file_path)
            # elif ext == ".rtf": return await self._parse_rtf(file_path)
            # elif ext == ".tmx": return await self._parse_tmx(file_path) # Likely adapt _parse_xliff
            # elif ext == ".ttx": return await self._parse_ttx(file_path) # Likely adapt _parse_xliff
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        except Exception as e:
             logger.error(f"Failed to parse {file_path.name} due to error: {e}")
             # Return a dictionary indicating failure, maybe with empty text
             return {
                  "text": "",
                  "format": ext.replace(".", ""),
                  "metadata": {"error": f"Parsing failed: {e}", "file_name": file_path.name},
             }

    async def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """Parse plain text file."""
        # ... (implementation seems okay) ...
        try:
            # Try UTF-8 first, then fallback
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            text = None
            for enc in encodings_to_try:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        text = f.read()
                    logger.debug(f"Read TXT file with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode TXT with {enc}, trying next.")
                except Exception as e:
                     raise ValueError(f"Could not read TXT file {file_path.name}") from e # Re-raise other errors
            if text is None:
                 raise ValueError(f"Could not decode TXT file {file_path.name} with attempted encodings.")

        except Exception as e:
            logger.error(f"Error parsing TXT: {e}")
            raise ValueError(f"Failed to parse TXT file {file_path.name}") from e

        return {
            "text": text, "format": "txt",
            "metadata": {"file_name": file_path.name, "file_size": file_path.stat().st_size, "char_count": len(text)},
        }


    async def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Parse Microsoft Word document using docx2txt."""
        # FIXED: Implementation using docx2txt
        try:
            text = docx2txt.process(file_path)
            if text:
                 text = text.strip() # Remove leading/trailing whitespace
            else:
                 text = "" # Ensure text is empty string if nothing extracted
            logger.info(f"Successfully extracted {len(text)} chars using docx2txt from {file_path.name}.")

            return {
                "text": text,
                "format": "docx",
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "char_count": len(text),
                },
            }
        except Exception as e:
            logger.error(f"Error parsing DOCX with docx2txt: {e}")
            raise ValueError(f"Failed to parse DOCX file {file_path.name}: {e}") from e

    async def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF document."""
        # ... (implementation seems okay, ensure PyPDF2 is robust enough) ...
        try:
            import PyPDF2
            text_list = []
            num_pages = 0
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f, strict=False) # Add strict=False for potentially problematic PDFs
                num_pages = len(reader.pages)
                for i, page in enumerate(reader.pages):
                     try:
                          page_text = page.extract_text()
                          if page_text: # Append only if text was extracted
                               text_list.append(page_text)
                     except Exception as page_e: # Catch errors during page extraction
                          logger.warning(f"Could not extract text from page {i+1} of {file_path.name}: {page_e}")
            combined_text = "\n".join(text_list).strip()
            logger.info(f"Extracted {len(combined_text)} chars from PDF with {num_pages} pages.")
            return {
                "text": combined_text, "format": "pdf",
                "metadata": {"file_name": file_path.name, "pages": num_pages, "char_count": len(combined_text)},
            }
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path.name}: {e}")
            raise ValueError(f"Failed to parse PDF file {file_path.name}") from e


    async def _parse_html(self, file_path: Path) -> Dict[str, Any]:
        """Parse HTML document."""
        # ... (implementation seems okay) ...
        try:
            from bs4 import BeautifulSoup
            # Try common encodings
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
            html_content = None
            for enc in encodings_to_try:
                 try:
                      with open(file_path, "r", encoding=enc) as f:
                           html_content = f.read()
                      logger.debug(f"Read HTML file with encoding: {enc}")
                      break
                 except UnicodeDecodeError:
                      logger.warning(f"Failed to decode HTML with {enc}, trying next.")
                 except Exception as e:
                      raise ValueError(f"Could not read HTML file {file_path.name}") from e
            if html_content is None:
                 raise ValueError(f"Could not decode HTML file {file_path.name} with attempted encodings.")

            soup = BeautifulSoup(html_content, "lxml") # Use lxml parser
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text = soup.get_text(separator='\n', strip=True) # Better whitespace handling
            return {
                "text": text, "format": "html",
                "metadata": {"file_name": file_path.name, "char_count": len(text)},
            }
        except ImportError:
            logger.error("beautifulsoup4 or lxml not installed. Install with: pip install beautifulsoup4 lxml")
            raise
        except Exception as e:
            logger.error(f"Error parsing HTML {file_path.name}: {e}")
            raise ValueError(f"Failed to parse HTML file {file_path.name}") from e

    async def _parse_xml(self, file_path: Path) -> Dict[str, Any]:
        """Parse generic XML document, extracting all text."""
        # ... (implementation seems okay, but might grab unwanted text) ...
        try:
            # Use itertext() for potentially better text extraction across elements
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            # Join text nodes, separated by newline, filter empty strings
            text_nodes = [text.strip() for text in root.xpath('//text()') if text.strip()]
            text = "\n".join(text_nodes)

            return {
                "text": text, "format": "xml",
                "metadata": {"file_name": file_path.name, "root_tag": root.tag, "char_count": len(text)},
            }
        except ImportError:
            logger.error("lxml not installed. Install with: pip install lxml")
            raise
        except etree.XMLSyntaxError as e:
            logger.error(f"Invalid XML in {file_path.name}: {e}")
            raise ValueError(f"Invalid XML structure in {file_path.name}") from e
        except Exception as e:
            logger.error(f"Error parsing generic XML {file_path.name}: {e}")
            raise ValueError(f"Failed to parse generic XML file {file_path.name}") from e


    # FIXED: Added _parse_xliff method
    async def _parse_xliff(self, file_path: Path) -> Dict[str, Any]:
        """Parse XLIFF variants (.xliff, .sdlxliff, .mqxliff). Extracts source text."""
        try:
            namespaces = { # Common namespaces
                'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
                'sdl': 'http://sdl.com/FileTypes/SdlXliff/1.0',
                'mq': 'MQXliff' # Assumed prefix, replace with actual URI if known
            }
            # Attempt to parse, recovering if possible from minor errors
            parser = etree.XMLParser(recover=True, ns_clean=True)
            tree = etree.parse(str(file_path), parser)
            root = tree.getroot()

            # More robust XPath to find source segments in various XLIFF flavors
            # Prioritize specific paths, then fall back to simpler ones
            xpath_queries = [
                './/xliff:trans-unit/xliff:source', # Standard XLIFF 1.2 source
                './/sdl:seg-source', # SDLXLIFF segment source (often contains internal tags)
                './/trans-unit/source', # XLIFF without explicit namespace
                './/mq:source', # Hypothetical MQXLIFF source (adjust prefix/URI)
                # Add more specific XPaths if needed for other variants
            ]

            source_elements = []
            for query in xpath_queries:
                 try:
                      elements = root.xpath(query, namespaces=namespaces)
                      if elements:
                           source_elements.extend(elements)
                           # Optional: break here if you only want the first successful query type
                 except etree.XPathEvalError as e:
                      logger.warning(f"XPath query failed for {file_path.name}: {query} - Error: {e}")


            # Deduplicate elements if multiple queries found the same ones
            source_elements = list(dict.fromkeys(source_elements))

            if not source_elements:
                logger.warning(f"No source text elements found in {file_path.name} using common XPaths.")
                extracted_text = ""
            else:
                # Concatenate text, handling potential internal tags gracefully
                segment_texts = []
                for elem in source_elements:
                    # itertext() joins text across sub-elements
                    segment_text = ''.join(elem.itertext()).strip()
                    if segment_text: # Only add non-empty segments
                        segment_texts.append(segment_text)
                extracted_text = "\n".join(segment_texts)

            logger.info(f"Extracted {len(extracted_text)} chars ({len(source_elements)} segments) from XLIFF: {file_path.name}")

            return {
                "text": extracted_text,
                "format": file_path.suffix.replace(".", ""), # Use specific extension
                "metadata": {
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "char_count": len(extracted_text),
                    "segment_count": len(source_elements)
                },
            }
        except etree.XMLSyntaxError as e:
            logger.error(f"Invalid XML syntax in {file_path.name}: {e}")
            raise ValueError(f"Invalid XML structure in {file_path.name}") from e
        except Exception as e:
            logger.error(f"Error parsing XLIFF {file_path.name}: {e}")
            raise ValueError(f"Failed to parse XLIFF file {file_path.name}") from e

    # Add placeholders for other methods if needed ( _parse_rtf, _parse_doc, _parse_tmx, etc.)
