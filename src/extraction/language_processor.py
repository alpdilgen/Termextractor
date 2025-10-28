# src/extraction/language_processor.py

"""Language-specific processing utilities."""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger
import unicodedata # normalize_diacritics ve detect_script için

# utils.constants modülünden importlar
try:
    from utils.constants import SUPPORTED_LANGUAGES, POS_TAGS
    constants_loaded = True
except ImportError:
    logger.warning("utils.constants modülü yüklenemedi. Bazı özellikler kısıtlı olabilir.")
    constants_loaded = False
    SUPPORTED_LANGUAGES = {} # Fallback
    POS_TAGS = [] # Fallback


@dataclass
class LanguageFeatures:
    """Language-specific features."""
    has_compounds: bool = False      # Bitişik kelime yapma eğilimi (örn: Almanca)
    is_agglutinative: bool = False   # Eklemeli dil (örn: Türkçe, Fince)
    is_rtl: bool = False             # Sağdan sola yazım (örn: Arapça)
    script: str = "Latin"            # Kullanılan alfabe (Latin, Cyrillic, Greek, etc.)
    typical_word_length: int = 5     # (Bu özellik şu an aktif kullanılmıyor)


class LanguageProcessor:
    """
    Terminoloji çıkarma için dile özgü işlemleri yönetir.
    Özellikler: Bitişik kelime analizi, morfoloji, alfabe tespiti vb.
    """

    def __init__(self):
        """Initialize LanguageProcessor."""
        # --- GÜNCELLENMİŞ VE GENİŞLETİLMİŞ DİL LİSTESİ ---
        # Mevcut dillere ek olarak tüm resmi AB dillerini (24) içerir.
        self.language_features: Dict[str, LanguageFeatures] = {
            # Mevcut EU Dilleri
            "bg": LanguageFeatures(script="Cyrillic"),                        # Bulgarca (EU)
            "da": LanguageFeatures(has_compounds=True, script="Latin"),      # Danca (EU)
            "de": LanguageFeatures(has_compounds=True, script="Latin"),      # Almanca (EU)
            "el": LanguageFeatures(has_compounds=True, script="Greek"),      # Yunanca (EU)
            "es": LanguageFeatures(has_compounds=False, script="Latin"),     # İspanyolca (EU)
            "fi": LanguageFeatures(has_compounds=True, is_agglutinative=True, script="Latin"), # Fince (EU)
            "fr": LanguageFeatures(has_compounds=False, script="Latin"),     # Fransızca (EU)
            "hu": LanguageFeatures(is_agglutinative=True, script="Latin"),   # Macarca (EU)
            "it": LanguageFeatures(has_compounds=False, script="Latin"),     # İtalyanca (EU)
            "nl": LanguageFeatures(has_compounds=True, script="Latin"),      # Hollandaca (EU)
            "pl": LanguageFeatures(has_compounds=False, script="Latin"),     # Lehçe (EU)
            "pt": LanguageFeatures(has_compounds=False, script="Latin"),     # Portekizce (EU)
            "ro": LanguageFeatures(has_compounds=False, script="Latin"),     # Rumence (EU)
            "sv": LanguageFeatures(has_compounds=True, script="Latin"),      # İsveççe (EU)

            # Eklenen Diğer EU Dilleri
            "cs": LanguageFeatures(has_compounds=False, script="Latin"),     # Çekçe (EU)
            "en": LanguageFeatures(has_compounds=True, script="Latin"),      # İngilizce (EU)
            "et": LanguageFeatures(is_agglutinative=True, script="Latin"),   # Estonca (EU)
            "ga": LanguageFeatures(has_compounds=False, script="Latin"),     # İrlandaca (EU)
            "hr": LanguageFeatures(has_compounds=False, script="Latin"),     # Hırvatça (EU)
            "lt": LanguageFeatures(has_compounds=False, script="Latin"),     # Litvanca (EU)
            "lv": LanguageFeatures(has_compounds=False, script="Latin"),     # Letonca (EU)
            "mt": LanguageFeatures(is_agglutinative=True, script="Latin"),   # Maltaca (EU)
            "sk": LanguageFeatures(has_compounds=False, script="Latin"),     # Slovakça (EU)
            "sl": LanguageFeatures(has_compounds=False, script="Latin"),     # Slovence (EU)

            # Mevcut Diğer Diller (EU Dışı)
            "tr": LanguageFeatures(is_agglutinative=True, script="Latin"),   # Türkçe
            "ar": LanguageFeatures(is_rtl=True, script="Arabic"),            # Arapça
            "he": LanguageFeatures(is_rtl=True, script="Hebrew"),            # İbranice
            "ru": LanguageFeatures(script="Cyrillic"),                        # Rusça
            "zh": LanguageFeatures(script="Han"),                             # Çince
            "ja": LanguageFeatures(script="Han/Hiragana/Katakana"),           # Japonca
            "ko": LanguageFeatures(script="Hangul"),                          # Korece
        }
        logger.info(f"LanguageProcessor {len(self.language_features)} dil özelliği ile başlatıldı.")

    def get_features(self, language_code: str) -> LanguageFeatures:
        """
        Dil koduna göre dile özgü özellikleri döndürür.
        Eğer dil listede yoksa, varsayılan (default) özellikleri döndürür.
        """
        return self.language_features.get(language_code, LanguageFeatures())

    def detect_compounds(
        self,
        text: str,
        language_code: str,
    ) -> List[str]:
        """
        Metindeki bitişik kelimeleri tespit eder (basit yöntem).
        """
        features = self.get_features(language_code)
        if not features.has_compounds:
            return []

        compounds = []
        words = text.split()

        # Almanca gibi diller için basit bulgusal (heuristic) yöntem
        if language_code in ["de", "nl", "sv", "da", "fi"]:
            for word in words:
                # 15 karakterden uzun ve sadece harflerden oluşan kelimeler
                if len(word) > 15 and word.isalpha():
                    compounds.append(word)
                # CamelCase (BüyükHarfleBaşlayanBitişik) kelimeler
                if re.match(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", word):
                    compounds.append(word)

        return list(set(compounds)) # Tekilleştirerek döndür

    def analyze_morphology(
        self,
        word: str,
        language_code: str,
    ) -> Dict[str, Any]:
        """
        Bir kelimenin morfolojisini (yapısını) analiz eder.
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

        # Basit bitişik kelime tespiti
        if features.has_compounds and len(word) > 15:
            analysis["is_compound"] = True
            analysis["estimated_parts"] = self._estimate_compound_parts(word)

        return analysis

    def _estimate_compound_parts(self, word: str) -> int:
        """Bitişik kelimenin parçalarını tahmin eder (çok basit yöntem)."""
        avg_length = 6 # Ortalama kelime uzunluğu varsayımı
        return max(2, len(word) // avg_length)

    def normalize_diacritics(
        self,
        text: str,
        remove: bool = False,
    ) -> str:
        """
        Metindeki aksanları/vurgu işaretlerini (diacritics) normalize eder veya kaldırır.
        """
        if remove:
            # Aksanları kaldır (örn: "é" -> "e")
            nfd = unicodedata.normalize("NFD", text)
            return "".join(char for char in nfd if unicodedata.category(char) != "Mn")
        else:
            # Standart forma getir (örn: farklı "e" ve "´" kombinasyonlarını tek "é" yapmak)
            return unicodedata.normalize("NFC", text)

    def detect_script(self, text: str) -> str:
        """
        Metnin hangi alfabeyle yazıldığını tespit eder (ilk 100 karaktere göre).
        """
        sample = text[:100]
        scripts: Set[str] = set()

        for char in sample:
            if char.isalpha():
                try:
                    # Karakterin Unicode adını al (örn: "LATIN SMALL LETTER A")
                    name = unicodedata.name(char).upper()
                    # Adın ilk kelimesini (alfabe adı) al
                    script_name = name.split()[0]
                    # Bazı CJK (Çin/Japon/Kore) karakterleri "CJK UNIFIED IDEOGRAPH" olarak geçer
                    if "CJK" in name:
                        scripts.add("HAN")
                    elif "HIRAGANA" in name:
                         scripts.add("HIRAGANA")
                    elif "KATAKANA" in name:
                         scripts.add("KATAKANA")
                    elif "HANGUL" in name:
                         scripts.add("HANGUL")
                    elif script_name:
                         scripts.add(script_name)
                except (ValueError, IndexError):
                    pass # Bazı karakterlerin adı olmayabilir

        # Tespit edilen alfabelere göre baskın olanı belirle
        if "ARABIC" in scripts: return "Arabic"
        if "HEBREW" in scripts: return "Hebrew"
        if "CYRILLIC" in scripts: return "Cyrillic"
        if "GREEK" in scripts: return "Greek"
        if "HAN" in scripts or "HIRAGANA" in scripts or "KATAKANA" in scripts: return "Han/Japanese"
        if "HANGUL" in scripts: return "Hangul"
        if "LATIN" in scripts: return "Latin"
        
        return "Unknown"

    def tokenize(
        self,
        text: str,
        language_code: str,
    ) -> List[str]:
        """
        Metni dile özgü olarak token'lara (kelimelere) ayırır.
        """
        features = self.get_features(language_code)

        # CJK (Çince/Japonca/Korece) dilleri için karakter bazlı
        if features.script in ["Han", "Han/Hiragana/Katakana", "Hangul"]:
            # Not: Bu çok basit bir yöntemdir. Üretim ortamında jieba (Çince),
            # MeCab (Japonca) gibi özel tokenizer'lar gerekir.
            return list(re.sub(r"\s+", "", text)) # Boşlukları kaldır ve karakterlere ayır

        # Diğer diller için boşluk bazlı basit ayırma
        return text.split()

    def extract_abbreviations(self, text: str) -> List[str]:
        """
        Metinden kısaltmaları (basit yöntemle) çıkarır.
        """
        # 2 veya daha fazla büyük harften oluşan kelimeler (örn: "API", "GDPR")
        abbreviations = re.findall(r"\b[A-Z]{2,}\b", text)
        # Noktalı kısaltmalar (örn: "U.S.A.")
        abbreviations.extend(re.findall(r"\b(?:[A-Z]\.){2,}", text))
        return list(set(abbreviations))

    def is_stopword(self, word: str, language_code: str) -> bool:
        """
        Kelimenin bir stopword (genel, anlamsız kelime) olup olmadığını kontrol eder.
        """
        # Not: Bu liste çok basittir ve constants.py'den daha kapsamlı bir
        # listeyle değiştirilmelidir.
        stopwords = {
            "en": {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"},
            "de": {"der", "die", "das", "und", "oder", "in", "an", "auf", "für"},
            "fr": {"le", "la", "les", "un", "une", "et", "ou", "dans", "sur"},
            "es": {"el", "la", "los", "las", "un", "una", "y", "o", "en"},
            "tr": {"ve", "ile", "bir", "bu", "o", "ama", "için", "mi"},
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
        Dil çiftinin desteklenip desteklenmediğini kontrol eder.
        """
        if not constants_loaded:
             logger.warning("Dil sabitleri yüklenemedi, doğrulama atlanıyor.")
             return True # Doğrulamayı atla
             
        if source_lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Desteklenmeyen kaynak dil: {source_lang}")
            return False

        if target_lang and (target_lang not in SUPPORTED_LANGUAGES):
            logger.warning(f"Desteklenmeyen hedef dil: {target_lang}")
            return False

        return True
