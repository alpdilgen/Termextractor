"""Helper functions and utilities for TermExtractor."""

import hashlib
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
from loguru import logger

from .constants import (
    SUPPORTED_LANGUAGES,
    ANTHROPIC_MODELS,
    FILE_FORMATS,
)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration using loguru.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format
    """
    # Remove default handler
    logger.remove()

    # Default format
    if log_format is None:
        log_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{name}:{function}:{line} - {message}"
        )

    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format=log_format,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        logger.add(
            sink=log_file,
            format=log_format,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="zip",
        )

    logger.info(f"Logging initialized at {level} level")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default.

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def get_language_name(language_code: str) -> str:
    """
    Get full language name from language code.

    Args:
        language_code: ISO 639-1 language code (e.g., 'en', 'de')

    Returns:
        Full language name

    Raises:
        ValueError: If language code is not supported
    """
    if language_code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language code: {language_code}. "
            f"Supported: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    return SUPPORTED_LANGUAGES[language_code]


def estimate_tokens(text: str, method: str = "simple") -> int:
    """
    Estimate number of tokens in text.

    Args:
        text: Input text
        method: Estimation method ('simple', 'words', 'chars')

    Returns:
        Estimated token count
    """
    if method == "simple":
        # Rough estimate: ~4 chars per token
        return len(text) // 4
    elif method == "words":
        # Estimate based on words: ~1.3 tokens per word
        words = len(text.split())
        return int(words * 1.3)
    elif method == "chars":
        # Character-based estimate
        return len(text) // 4
    else:
        # Default to simple
        return len(text) // 4


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cost_per_1k: Dict[str, Dict[str, float]],
) -> float:
    """
    Calculate API cost based on token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier
        cost_per_1k: Cost per 1000 tokens dictionary

    Returns:
        Total cost in USD
    """
    if model not in cost_per_1k:
        logger.warning(f"Cost data not available for model {model}")
        return 0.0

    costs = cost_per_1k[model]
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    return input_cost + output_cost


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename
        max_length: Maximum filename length

    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(". ")
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        max_name_length = max_length - len(ext) - 1
        filename = f"{name[:max_name_length]}.{ext}" if ext else name[:max_length]
    return filename


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text for caching purposes.

    Args:
        text: Input text
        algorithm: Hash algorithm (md5, sha1, sha256)

    Returns:
        Hex digest of hash
    """
    hash_func = getattr(hashlib, algorithm)
    return hash_func(text.encode("utf-8")).hexdigest()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get file extension from path.

    Args:
        file_path: Path to file

    Returns:
        File extension (lowercase, with dot)
    """
    return Path(file_path).suffix.lower()


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file format is supported.

    Args:
        file_path: Path to file

    Returns:
        True if supported, False otherwise
    """
    ext = get_file_extension(file_path)
    all_formats = (
        FILE_FORMATS["translation"]
        + FILE_FORMATS["document"]
        + FILE_FORMATS["termbase"]
    )
    return ext in all_formats


def format_timestamp(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp.

    Args:
        dt: Datetime object (None = current time)
        fmt: Format string

    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)


def parse_domain_path(domain_path: str) -> List[str]:
    """
    Parse hierarchical domain path into components.

    Args:
        domain_path: Domain path string (e.g., "Medical/Healthcare/Veterinary")

    Returns:
        List of domain components
    """
    # Split by common separators
    separators = ["/", "→", ">", ">>", "->"]
    for sep in separators:
        if sep in domain_path:
            return [part.strip() for part in domain_path.split(sep)]
    # If no separator found, return as single component
    return [domain_path.strip()]


def merge_dicts_deep(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    return result


def validate_language_pair(source_lang: str, target_lang: Optional[str] = None) -> bool:
    """
    Validate language pair.

    Args:
        source_lang: Source language code
        target_lang: Target language code (None for monolingual)

    Returns:
        True if valid, False otherwise
    """
    if source_lang not in SUPPORTED_LANGUAGES:
        return False
    if target_lang is not None and target_lang not in SUPPORTED_LANGUAGES:
        return False
    return True


def normalize_text(text: str, preserve_case: bool = False) -> str:
    """
    Normalize text for processing.

    Args:
        text: Input text
        preserve_case: Whether to preserve case

    Returns:
        Normalized text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Optionally lowercase
    if not preserve_case:
        text = text.lower()

    return text


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """
    Create directory structure for outputs.

    Args:
        base_path: Base directory path
    """
    base_path = Path(base_path)
    subdirs = ["exports", "cache", "logs", "temp"]

    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Directory structure created at {base_path}")
