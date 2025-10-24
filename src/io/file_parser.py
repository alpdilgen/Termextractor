"""File parser for various document formats."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger

from termextractor.utils.helpers import get_file_extension


class FileParser:
    """
    Parse various document formats to extract text.

    Supported formats:
    - Plain text (.txt)
    - Microsoft Word (.docx)
    - PDF (.pdf)
    - HTML (.html, .htm)
    - XML (.xml)
    """

    def __init__(self):
        """Initialize FileParser."""
        logger.info("FileParser initialized")

    async def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse file and extract text content.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with text and metadata

        Raises:
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = get_file_extension(file_path)

        logger.info(f"Parsing file: {file_path} ({ext})")

        if ext == ".txt":
            return await self._parse_txt(file_path)
        elif ext == ".docx":
            return await self._parse_docx(file_path)
        elif ext == ".pdf":
            return await self._parse_pdf(file_path)
        elif ext in [".html", ".htm"]:
            return await self._parse_html(file_path)
        elif ext == ".xml":
            return await self._parse_xml(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    async def _parse_txt(self, file_path: Path) -> Dict[str, Any]:
        """Parse plain text file."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return {
            "text": text,
            "format": "txt",
            "metadata": {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "char_count": len(text),
            },
        }

    async def _parse_docx(self, file_path: Path) -> Dict[str, Any]:
        """Parse Microsoft Word document."""
        try:
            from docx import Document

            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)

            return {
                "text": text,
                "format": "docx",
                "metadata": {
                    "file_name": file_path.name,
                    "paragraphs": len(paragraphs),
                    "char_count": len(text),
                },
            }

        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            raise
        except Exception as e:
            logger.error(f"Error parsing DOCX: {e}")
            raise

    async def _parse_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF document."""
        try:
            import PyPDF2

            text = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())

            combined_text = "\n".join(text)

            return {
                "text": combined_text,
                "format": "pdf",
                "metadata": {
                    "file_name": file_path.name,
                    "pages": len(reader.pages),
                    "char_count": len(combined_text),
                },
            }

        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            raise

    async def _parse_html(self, file_path: Path) -> Dict[str, Any]:
        """Parse HTML document."""
        try:
            from bs4 import BeautifulSoup

            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)

            return {
                "text": text,
                "format": "html",
                "metadata": {
                    "file_name": file_path.name,
                    "char_count": len(text),
                },
            }

        except ImportError:
            logger.error("beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
            raise
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            raise

    async def _parse_xml(self, file_path: Path) -> Dict[str, Any]:
        """Parse XML document."""
        try:
            from lxml import etree

            tree = etree.parse(str(file_path))
            root = tree.getroot()

            # Extract all text content
            text = " ".join(root.itertext())

            return {
                "text": text,
                "format": "xml",
                "metadata": {
                    "file_name": file_path.name,
                    "root_tag": root.tag,
                    "char_count": len(text),
                },
            }

        except ImportError:
            logger.error("lxml not installed. Install with: pip install lxml")
            raise
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            raise
