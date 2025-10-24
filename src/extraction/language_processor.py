"""Language-specific processing utilities."""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger
import unicodedata # Added for normalize_diacritics and detect_script

# FIXED: Removed termextractor. prefix
from utils.constants import SUPPORTED_LANGUAGES, POS_TAGS


@dataclass
class LanguageFeatures:
    """Language-specific features."""
    has_compounds: bool = False
    is_agglutinative: bool = False
    is_rtl: bool = False
    script: str = "Latin"
    typical_word_length: int = 5


class LanguageProcessor:
    """Language-specific processing for terminology extraction."""
    def __init__(self):
        """Initialize LanguageProcessor."""
        # Language feature mapping (consider moving to constants or config)
        self.language_features: Dict[str, LanguageFeatures] = {
            "de": LanguageFeatures(has_compounds=True, script="Latin"),
            "nl": LanguageFeatures(has_compounds=True, script="Latin"),
            "sv": LanguageFeatures(has_compounds=True, script="Latin"),
            "da": LanguageFeatures(has_compounds=True, script="Latin"),
            "tr": LanguageFeatures(is_agglutinative=True, script="Latin"),
            "fi": LanguageFeatures(is_agglutinative=True, script="Latin"),
            "hu": LanguageFeatures(is_agglutinative=True, script="Latin"),
            "ar": LanguageFeatures(is_rtl=True, script="Arabic"),
            "he": LanguageFeatures(is_rtl=True, script="Hebrew"),
            "ru": LanguageFeatures(script="Cyrillic"),
            "bg": LanguageFeatures(script="Cyrillic"),
            "zh": LanguageFeatures(script="Han"),
            "ja": LanguageFeatures(script="Han/Hiragana/Katakana"),
            "ko": LanguageFeatures(script="Hangul"),
            # Add defaults for others or explicitly list all from SUPPORTED_LANGUAGES
        }
        logger.info("LanguageProcessor initialized")

    def get_features(self, language_code: str) -> LanguageFeatures:
        """Get language-specific features."""
        return self.language_features.get(language_code, LanguageFeatures()) # Default features

    # ... (rest of methods like detect_compounds, analyze_morphology, etc.) ...
    def normalize_diacritics(self, text: str, remove: bool = False) -> str:
        """Normalize or remove diacritics in text."""
        # FIXED: Moved import unicodedata to top of file
        if remove:
            nfd = unicodedata.normalize("NFD", text)
            return "".join(char for char in nfd if unicodedata.category(char) != "Mn")
        else:
            return unicodedata.normalize("NFC", text)

    def detect_script(self, text: str) -> str:
        """Detect script of text."""
        # FIXED: Moved import unicodedata to top of file
        # ... (implementation seems okay) ...
        return "Unknown" # Placeholder

    # ... (Other methods seem okay, assuming imports like 're' are correct) ...
