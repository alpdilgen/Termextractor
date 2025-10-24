"""Language-specific processing utilities."""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger

from termextractor.utils.constants import SUPPORTED_LANGUAGES, POS_TAGS


@dataclass
class LanguageFeatures:
    """Language-specific features."""

    has_compounds: bool = False
    is_agglutinative: bool = False
    is_rtl: bool = False
    script: str = "Latin"
    typical_word_length: int = 5


class LanguageProcessor:
    """
    Language-specific processing for terminology extraction.

    Features:
    - Compound word analysis (Germanic languages)
    - Morphological processing
    - Script detection
    - Language-specific tokenization
    - Diacritic handling
    """

    def __init__(self):
        """Initialize LanguageProcessor."""
        # Language feature mapping
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
        }

        logger.info("LanguageProcessor initialized")

    def get_features(self, language_code: str) -> LanguageFeatures:
        """
        Get language-specific features.

        Args:
            language_code: Language code

        Returns:
            LanguageFeatures for the language
        """
        return self.language_features.get(
            language_code,
            LanguageFeatures(),  # Default features
        )

    def detect_compounds(
        self,
        text: str,
        language_code: str,
    ) -> List[str]:
        """
        Detect compound words in text.

        Args:
            text: Text to analyze
            language_code: Language code

        Returns:
            List of detected compounds
        """
        features = self.get_features(language_code)

        if not features.has_compounds:
            return []

        compounds = []

        # Simple heuristics for Germanic languages
        if language_code in ["de", "nl", "sv", "da"]:
            # Look for long words (likely compounds)
            words = text.split()
            for word in words:
                # German compounds are typically longer
                if len(word) > 15 and word.isalpha():
                    compounds.append(word)

                # CamelCase compounds
                if re.match(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", word):
                    compounds.append(word)

        return compounds

    def analyze_morphology(
        self,
        word: str,
        language_code: str,
    ) -> Dict[str, any]:
        """
        Analyze morphology of a word.

        Args:
            word: Word to analyze
            language_code: Language code

        Returns:
            Morphological analysis
        """
        features = self.get_features(language_code)

        analysis = {
            "word": word,
            "language": language_code,
            "length": len(word),
            "is_compound": False,
            "is_agglutinative": features.is_agglutinative,
            "script": features.script,
        }

        # Simple compound detection
        if features.has_compounds and len(word) > 15:
            analysis["is_compound"] = True
            analysis["estimated_parts"] = self._estimate_compound_parts(word)

        return analysis

    def _estimate_compound_parts(self, word: str) -> int:
        """
        Estimate number of parts in compound word.

        Args:
            word: Compound word

        Returns:
            Estimated number of parts
        """
        # Very simple heuristic: divide by average word length
        avg_length = 6
        return max(2, len(word) // avg_length)

    def normalize_diacritics(
        self,
        text: str,
        remove: bool = False,
    ) -> str:
        """
        Normalize diacritics in text.

        Args:
            text: Text to normalize
            remove: If True, remove diacritics; if False, standardize them

        Returns:
            Normalized text
        """
        if remove:
            import unicodedata

            # Remove diacritics
            nfd = unicodedata.normalize("NFD", text)
            return "".join(char for char in nfd if unicodedata.category(char) != "Mn")
        else:
            # Standardize to NFC form
            import unicodedata

            return unicodedata.normalize("NFC", text)

    def detect_script(self, text: str) -> str:
        """
        Detect script of text.

        Args:
            text: Text to analyze

        Returns:
            Script name
        """
        import unicodedata

        # Sample first 100 chars
        sample = text[:100]

        scripts = set()
        for char in sample:
            if char.isalpha():
                try:
                    script = unicodedata.name(char).split()[0]
                    scripts.add(script)
                except ValueError:
                    pass

        # Determine primary script
        if "LATIN" in scripts:
            return "Latin"
        elif "CYRILLIC" in scripts:
            return "Cyrillic"
        elif "ARABIC" in scripts:
            return "Arabic"
        elif "HEBREW" in scripts:
            return "Hebrew"
        elif "CJK" in " ".join(scripts) or "HIRAGANA" in scripts:
            return "Han/Japanese"
        elif "HANGUL" in scripts:
            return "Hangul"
        else:
            return "Unknown"

    def tokenize(
        self,
        text: str,
        language_code: str,
    ) -> List[str]:
        """
        Language-specific tokenization.

        Args:
            text: Text to tokenize
            language_code: Language code

        Returns:
            List of tokens
        """
        features = self.get_features(language_code)

        # For CJK languages, might need character-based tokenization
        if features.script in ["Han", "Han/Hiragana/Katakana", "Hangul"]:
            # Simple character-based for now
            # In production, use proper tokenizers like jieba, MeCab, etc.
            return list(text)

        # For other languages, whitespace-based
        return text.split()

    def extract_abbreviations(self, text: str) -> List[str]:
        """
        Extract abbreviations from text.

        Args:
            text: Text to analyze

        Returns:
            List of abbreviations
        """
        # All-caps words (2+ letters)
        abbreviations = re.findall(r"\b[A-Z]{2,}\b", text)

        # Acronyms with dots (e.g., U.S.A.)
        abbreviations.extend(re.findall(r"\b(?:[A-Z]\.){2,}", text))

        return list(set(abbreviations))

    def is_stopword(self, word: str, language_code: str) -> bool:
        """
        Check if word is a stopword.

        Args:
            word: Word to check
            language_code: Language code

        Returns:
            True if stopword
        """
        # Basic stopwords for major languages
        # In production, use comprehensive stopword lists
        stopwords = {
            "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"},
            "de": {"der", "die", "das", "und", "oder", "in", "an", "auf", "für"},
            "fr": {"le", "la", "les", "un", "une", "et", "ou", "dans", "sur"},
            "es": {"el", "la", "los", "las", "un", "una", "y", "o", "en"},
        }

        word_lower = word.lower()
        lang_stopwords = stopwords.get(language_code, set())

        return word_lower in lang_stopwords

    def validate_language_pair(
        self,
        source_lang: str,
        target_lang: Optional[str],
    ) -> bool:
        """
        Validate language pair.

        Args:
            source_lang: Source language
            target_lang: Target language (None for monolingual)

        Returns:
            True if valid
        """
        if source_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported source language: {source_lang}")
            return False

        if target_lang and target_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported target language: {target_lang}")
            return False

        return True
