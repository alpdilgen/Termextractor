# src/utils/helpers.py

"""Helper functions and utilities for TermExtractor."""

import hashlib
import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import sys # Needed for logger setup
import logging # Using standard logging as a fallback/alternative if loguru fails
from loguru import logger

# Ensure constants are imported correctly relative to src/
# If constants.py also fails to load, this could be the root cause
try:
    from .constants import (
        SUPPORTED_LANGUAGES,
        ANTHROPIC_MODELS,
        FILE_FORMATS,
        COST_PER_1K_TOKENS # Needed for calculate_cost
    )
    constants_loaded = True
except ImportError as e:
    print(f"[ERROR] Failed to import constants in helpers.py: {e}. Some functions may fail.", file=sys.stderr)
    constants_loaded = False
    # Define fallbacks if needed, though this indicates a major structure issue
    SUPPORTED_LANGUAGES = {"en": "English"}
    ANTHROPIC_MODELS = ["default-model"]
    FILE_FORMATS = {"document": [".txt"]}
    COST_PER_1K_TOKENS = {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    use_loguru: bool = True # Flag to control using loguru or standard logging
) -> None:
    """
    Set up logging using Loguru (preferred) or standard logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file (e.g., "app.log" or Path object).
                  If None, logs only to console. If path provided, creates dir if needed.
        log_format: Optional custom log format string.
        use_loguru: If True, use Loguru. If False, use standard logging.
    """
    effective_level = level.upper()
    log_to_file = log_file is not None

    # --- Loguru Setup ---
    if use_loguru:
        try:
            logger.remove() # Remove existing handlers to avoid duplication

            # Default format if none provided
            if log_format is None:
                log_format = (
                    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                    # "{name}:{function}:{line} - {message}" # Can be verbose
                    "{level_name} - {message}" # Simpler format
                )

            # Console handler (stderr)
            logger.add(
                sink=sys.stderr, # Use stderr for console logs
                format=log_format,
                level=effective_level,
                colorize=True, # Keep colors for console
            )

            # File handler (if log_file path is provided)
            if log_to_file:
                log_file_path = Path(log_file) # Ensure it's a Path object
                # Create log directory if it doesn't exist
                try:
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as dir_e:
                    print(f"[ERROR] Could not create log directory {log_file_path.parent}: {dir_e}", file=sys.stderr)
                    # Continue without file logging if directory fails

                logger.add(
                    sink=log_file_path, # Path object works directly
                    format=log_format, # Use the same format for file
                    level=effective_level,
                    rotation="10 MB", # Rotate log file when it reaches 10 MB
                    retention="7 days", # Keep logs for 7 days
                    compression="zip", # Compress rotated files
                    encoding="utf-8", # Explicitly set encoding
                    enqueue=True, # Make logging asynchronous (safer for web apps)
                    catch=True, # Catch errors within Loguru itself
                )
                logger.info(f"Loguru file logging enabled: {log_file_path}")

            logger.info(f"Loguru logging initialized at level {effective_level}.")
            print(f"DEBUG: Loguru setup complete. Level={effective_level}, File={log_file_path if log_to_file else 'None'}") # Debug print

        except Exception as loguru_e:
            print(f"[ERROR] Failed to initialize Loguru: {loguru_e}. Falling back to standard logging.", file=sys.stderr)
            use_loguru = False # Force fallback

    # --- Standard Logging Fallback ---
    if not use_loguru:
        try:
            log_handlers = []
            # Console handler
            console_handler = logging.StreamHandler(sys.stderr)
            log_handlers.append(console_handler)

            # File handler
            if log_to_file:
                log_file_path = Path(log_file)
                try:
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
                    log_handlers.append(file_handler)
                    print(f"Standard logging file handler enabled: {log_file_path}")
                except Exception as dir_e:
                    print(f"[ERROR] Could not create log directory for standard logging {log_file_path.parent}: {dir_e}", file=sys.stderr)

            # Default format
            if log_format is None:
                 log_format = "%(asctime)s | %(levelname)-8s | %(message)s"

            logging.basicConfig(
                level=effective_level,
                format=log_format,
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=log_handlers,
                force=True # Override any previous basicConfig
            )
            logging.info(f"Standard logging initialized at level {effective_level}.")
            print(f"DEBUG: Standard logging setup complete. Level={effective_level}, File={log_file_path if log_to_file else 'None'}")

        except Exception as std_log_e:
            print(f"[CRITICAL ERROR] Failed to initialize BOTH Loguru and standard logging: {std_log_e}", file=sys.stderr)
            # Basic print logging might be the only option left
            logging.basicConfig(level=logging.WARNING) # Minimal fallback
            logging.error("Logging system failed to initialize properly.")


# --- Rest of the helper functions ---

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Define a default config path relative to this helper file's location
        # Assumes config.yaml is in a 'config' folder one level above 'src'
        # Adjust if your structure is different (e.g., inside src)
        default_config_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        # Alternative: look inside src/config
        # default_config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        config_path = default_config_path
        logger.info(f"No config path provided, trying default: {config_path}")
    else:
        config_path = Path(config_path).resolve() # Use absolute path

    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}. Returning empty config.")
        return {} # Return empty dict instead of raising error immediately

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty YAML file
             logger.warning(f"Configuration file {config_path} is empty. Returning empty config.")
             return {}
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        raise ValueError(f"Invalid YAML configuration file: {e}") from e
    except Exception as e:
         logger.error(f"Failed to load configuration from {config_path}: {e}", exc_info=True)
         raise IOError(f"Could not load configuration file: {e}") from e


def get_language_name(language_code: str) -> str:
    """Get full language name from language code."""
    if not constants_loaded: return language_code # Fallback if constants failed
    name = SUPPORTED_LANGUAGES.get(language_code)
    if name is None:
        logger.warning(f"Unsupported language code: {language_code}")
        # Decide: raise error or return code? Returning code might be more robust.
        return language_code
        # raise ValueError(f"Unsupported language code: {language_code}")
    return name


def estimate_tokens(text: str, method: str = "simple") -> int:
    """Estimate number of tokens in text (simple approximation)."""
    if not text: return 0
    # Simple estimate: ~4 chars per token (rough average for English-like text)
    # This is very approximate and varies by language and tokenizer.
    estimated = len(text) // 4
    # Ensure at least 1 token for non-empty string
    return max(1, estimated) if text else 0


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    cost_per_1k: Dict[str, Dict[str, float]], # Pass the constant here
) -> float:
    """Calculate API cost based on token usage and model rates."""
    if not constants_loaded or not cost_per_1k:
         logger.warning("Cost constants not loaded, cannot calculate cost.")
         return 0.0

    model_costs = cost_per_1k.get(model)
    if model_costs is None:
        logger.warning(f"Cost data not available for model '{model}'. Returning 0.0 cost.")
        # Try finding a base model name if variant isn't listed (e.g. claude-3-5-sonnet from dated version)
        base_model = model.split('-')[0] + '-' + model.split('-')[1] # Very basic attempt
        model_costs = cost_per_1k.get(base_model)
        if model_costs is None:
             return 0.0

    try:
        input_cost = (input_tokens / 1000.0) * model_costs.get("input", 0.0)
        output_cost = (output_tokens / 1000.0) * model_costs.get("output", 0.0)
        return input_cost + output_cost
    except Exception as e:
        logger.error(f"Error calculating cost for model {model}: {e}")
        return 0.0


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename by removing invalid characters and limiting length."""
    if not isinstance(filename, str): filename = str(filename)
    # Remove characters invalid in Windows/Linux filenames
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", filename)
    # Remove leading/trailing spaces, dots, and underscores
    sanitized = sanitized.strip("._ ")
    # Replace multiple consecutive underscores with a single one
    sanitized = re.sub(r"_+", "_", sanitized)
    # Truncate if too long, preserving extension
    if len(sanitized) > max_length:
        stem, sep, ext = sanitized.rpartition(".")
        if sep and len(ext) < 10: # Check if it looks like a valid extension
            max_stem_length = max_length - len(ext) - 1
            sanitized = f"{stem[:max_stem_length]}.{ext}"
        else: # No extension or very long one, just truncate
            sanitized = sanitized[:max_length]
    # Ensure filename isn't empty after sanitizing
    return sanitized if sanitized else "_invalid_filename_"


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Generate hash of text for caching keys."""
    try:
        hash_func = getattr(hashlib, algorithm)
        return hash_func(text.encode("utf-8")).hexdigest()
    except AttributeError:
        logger.warning(f"Invalid hash algorithm '{algorithm}', using sha256.")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error(f"Hashing failed: {e}")
        # Fallback? Maybe hash a timestamp? For cache keys, needs consistency.
        # Fallback to simple hash of first N chars might work but risk collision
        return hashlib.sha256(text[:1000].encode("utf-8")).hexdigest()


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    if chunk_size <= 0: chunk_size = len(items) or 1 # Avoid infinite loop or zero division
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension (lowercase, with dot)."""
    try:
        return Path(file_path).suffix.lower()
    except Exception:
        return "" # Return empty if path is invalid


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """Check if file format is in the list of supported formats."""
    if not constants_loaded: return False # Cannot check if constants failed
    ext = get_file_extension(file_path)
    all_formats = set()
    for fmt_list in FILE_FORMATS.values():
         all_formats.update(fmt_list)
    return ext in all_formats


def format_timestamp(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime object or current time."""
    if dt is None: dt = datetime.now()
    try: return dt.strftime(fmt)
    except: return str(dt) # Fallback


def parse_domain_path(domain_path: str) -> List[str]:
    """Parse hierarchical domain path string into components."""
    if not isinstance(domain_path, str) or not domain_path.strip():
        return []
    # Split by common separators, prioritize longer ones first if needed
    separators = [" → ", "/", "->", ">>", ">"] # Check for space around arrow
    cleaned_path = domain_path.strip()
    for sep in separators:
        if sep in cleaned_path:
            # Split and strip whitespace from each part, filter out empty parts
            parts = [part.strip() for part in cleaned_path.split(sep) if part.strip()]
            return parts[:5] # Limit depth
    # If no separator found, return as single component list
    return [cleaned_path][:5] # Limit depth


def merge_dicts_deep(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries (dict2 values overwrite dict1)."""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts_deep(result[key], value)
        else:
            result[key] = value
    return result


def validate_language_pair(source_lang: str, target_lang: Optional[str] = None) -> bool:
    """Validate if language codes are in the supported list."""
    if not constants_loaded: return False # Cannot validate
    if source_lang not in SUPPORTED_LANGUAGES:
        return False
    if target_lang is not None and target_lang not in SUPPORTED_LANGUAGES:
        return False
    return True


def normalize_text(text: str, preserve_case: bool = False) -> str:
    """Normalize text: collapse whitespace, optionally lowercase."""
    if not isinstance(text, str): return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not preserve_case:
        normalized = normalized.lower()
    return normalized


def create_directory_structure(base_path: Union[str, Path]) -> None:
    """Create standard subdirectories if they don't exist."""
    try:
        base_p = Path(base_path).resolve() # Use absolute path
        subdirs = ["exports", "cache", "logs", "temp", "config", "data"] # Added config/data

        for subdir_name in subdirs:
            subdir_path = base_p / subdir_name
            subdir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {subdir_path}")

        logger.info(f"Verified base directory structure at {base_p}")
    except Exception as e:
         logger.error(f"Failed to create/verify directory structure at {base_path}: {e}", exc_info=True)
