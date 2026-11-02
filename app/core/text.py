import re
import unicodedata

_TRANSLATION = str.maketrans({
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
    "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا", "ـ": "",
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_WHITESPACE = re.compile(r"\s+")


def normalize_persian(text: str) -> str:
    """Normalize Arabic/Persian glyph variants, diacritics and whitespace."""
    value = unicodedata.normalize("NFKC", text).translate(_TRANSLATION)
    value = _DIACRITICS.sub("", value)
    value = re.sub(r"\s*\u200c\s*", "\u200c", value)
    return _WHITESPACE.sub(" ", value).strip()
