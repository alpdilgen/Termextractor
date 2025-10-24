"""Constants and enumerations used throughout TermExtractor."""

from enum import Enum
from typing import Dict, List


# Anthropic Models
class AnthropicModel(str, Enum):
    """Available Anthropic Claude models."""

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet-20250219"


ANTHROPIC_MODELS: List[str] = [model.value for model in AnthropicModel]

DEFAULT_MODEL = AnthropicModel.CLAUDE_3_5_SONNET.value


# Supported Languages
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "bg": "Bulgarian",
    "ro": "Romanian",
    "tr": "Turkish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "he": "Hebrew",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "cs": "Czech",
    "sv": "Swedish",
    "da": "Danish",
}


# File Formats
class FileFormat(str, Enum):
    """Supported file formats."""

    # Translation formats
    MQXLIFF = ".mqxliff"
    SDLXLIFF = ".sdlxliff"
    XLIFF = ".xliff"
    XLF = ".xlf"
    TMX = ".tmx"
    TTX = ".ttx"

    # Document formats
    DOCX = ".docx"
    DOC = ".doc"
    PDF = ".pdf"
    RTF = ".rtf"
    TXT = ".txt"
    HTML = ".html"
    HTM = ".htm"
    XML = ".xml"

    # Termbase formats
    TBX = ".tbx"
    XLSX = ".xlsx"
    XLS = ".xls"
    CSV = ".csv"
    MTF = ".mtf"
    SDLTB = ".sdltb"


FILE_FORMATS: Dict[str, List[str]] = {
    "translation": [".mqxliff", ".sdlxliff", ".xliff", ".xlf", ".tmx", ".ttx"],
    "document": [".docx", ".doc", ".pdf", ".rtf", ".txt", ".html", ".htm", ".xml"],
    "termbase": [".tbx", ".xlsx", ".xls", ".csv", ".mtf", ".sdltb"],
}


# Export Formats
class ExportFormat(str, Enum):
    """Supported export formats."""

    TBX = "tbx"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"
    XML = "xml"


# Processing Modes
class ProcessingMode(str, Enum):
    """Processing modes for term extraction."""

    FAST = "fast"  # Approximate, faster processing
    BALANCED = "balanced"  # Balance between speed and quality
    DETAILED = "detailed"  # Thorough, slower processing


# Quality Tiers
class QualityTier(str, Enum):
    """Quality tiers for extracted terms."""

    HIGH = "high"  # 80+ relevance score
    MEDIUM = "medium"  # 60-79 relevance score
    LOW = "low"  # <60 relevance score


# Domain Categories (examples, can be extended)
DOMAIN_CATEGORIES: Dict[str, List[str]] = {
    "Medical": [
        "Healthcare",
        "Veterinary Medicine",
        "Dentistry",
        "Pharmacy",
        "Biotechnology",
    ],
    "Technology": [
        "Software Development",
        "Hardware Engineering",
        "AI/Machine Learning",
        "Cybersecurity",
        "Telecommunications",
    ],
    "Legal": [
        "Contract Law",
        "Intellectual Property",
        "Corporate Law",
        "Criminal Law",
        "International Law",
    ],
    "Finance": [
        "Banking",
        "Investment",
        "Insurance",
        "Accounting",
        "Cryptocurrency",
    ],
    "Engineering": [
        "Mechanical Engineering",
        "Civil Engineering",
        "Electrical Engineering",
        "Chemical Engineering",
        "Aerospace Engineering",
    ],
    "Science": [
        "Physics",
        "Chemistry",
        "Biology",
        "Environmental Science",
        "Astronomy",
    ],
}


# Metadata Categories
class MetadataCategory(str, Enum):
    """Metadata categories for term enrichment."""

    STATISTICAL = "statistical"
    LINGUISTIC = "linguistic"
    SEMANTIC = "semantic"
    DOMAIN_CONTEXT = "domain_context"
    TRANSLATION = "translation"
    QUALITY = "quality"
    TEMPORAL = "temporal"
    REFERENCE = "reference"
    STRUCTURAL = "structural"
    RELATIONSHIP = "relationship"
    USAGE = "usage"


# Part of Speech Tags
POS_TAGS: List[str] = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PROPN",
    "PRON",
    "DET",
    "ADP",
    "NUM",
    "CONJ",
    "PART",
    "PUNCT",
    "SYM",
    "X",
]


# API Configuration
API_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 4
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff: 2s, 4s, 8s, 16s


# Token Limits (approximate)
TOKEN_LIMITS: Dict[str, int] = {
    AnthropicModel.CLAUDE_3_5_SONNET.value: 200000,
    AnthropicModel.CLAUDE_3_5_HAIKU.value: 200000,
    AnthropicModel.CLAUDE_3_OPUS.value: 200000,
    AnthropicModel.CLAUDE_3_HAIKU.value: 200000,
    AnthropicModel.CLAUDE_3_7_SONNET.value: 200000,
}


# Cost per 1K tokens (approximate, update as needed)
COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = {
    AnthropicModel.CLAUDE_3_5_SONNET.value: {"input": 0.003, "output": 0.015},
    AnthropicModel.CLAUDE_3_5_HAIKU.value: {"input": 0.0008, "output": 0.004},
    AnthropicModel.CLAUDE_3_OPUS.value: {"input": 0.015, "output": 0.075},
    AnthropicModel.CLAUDE_3_HAIKU.value: {"input": 0.00025, "output": 0.00125},
    AnthropicModel.CLAUDE_3_7_SONNET.value: {"input": 0.003, "output": 0.015},
}


# Default configuration values
DEFAULT_BATCH_SIZE = 100
DEFAULT_RELEVANCE_THRESHOLD = 70
DEFAULT_MIN_TERM_FREQUENCY = 2
DEFAULT_MAX_PARALLEL_FILES = 5
DEFAULT_CACHE_TTL_HOURS = 24


# Regular expressions for term extraction
TERM_PATTERNS = {
    "compound": r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b",  # CamelCase compounds
    "abbreviation": r"\b[A-Z]{2,}\b",  # All caps abbreviations
    "hyphenated": r"\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)*\b",  # Hyphenated terms
}


# Security
ENCRYPTION_ALGORITHM = "AES-256-GCM"
DEFAULT_DATA_RETENTION_DAYS = 7
AUDIT_LOG_RETENTION_DAYS = 90
