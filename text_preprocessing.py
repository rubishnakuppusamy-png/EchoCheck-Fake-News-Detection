"""
EchoCheck - Text Preprocessing Pipeline
Used consistently during both Model Training and Real-Time Inference.
"""

import re
import string

# Regex patterns for cleaning news article text
RE_URL = re.compile(r'https?://\S+|www\.\S+')
RE_HTML = re.compile(r'<.*?>')
RE_TWITTER = re.compile(r'@\w+')
RE_SPECIAL_CHARS = re.compile(r'[^a-zA-Z\s]')
RE_MULTISPACE = re.compile(r'\s+')

# Standard wire/agency datelines (e.g., "WASHINGTON (Reuters) -", "NEW YORK (AP) -")
# Removing these prevents the model from overfitting solely to news-agency tags.
RE_DATELINE = re.compile(
    r'^\s*([A-Za-z\s,]+)?\((?:reuters|ap|afp|bloomberg|pti|al jazeera)\)\s*[-–—:]\s*',
    re.IGNORECASE
)
RE_AGENCY_KEYWORDS = re.compile(r'\b(reuters|associated press)\b', re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Cleans raw news article text:
    1. Casts to string and lowercases.
    2. Removes wire datelines and agency signatures.
    3. Strips URLs and HTML tags.
    4. Removes social handles and non-alphabet characters.
    5. Normalizes whitespace.
    """
    if not isinstance(text, str):
        text = str(text or '')

    # Lowercase
    cleaned = text.lower()

    # Remove datelines at start of articles
    cleaned = RE_DATELINE.sub(' ', cleaned)

    # Remove explicit wire tags that leak source identity
    cleaned = RE_AGENCY_KEYWORDS.sub(' ', cleaned)

    # Remove URLs and HTML
    cleaned = RE_URL.sub(' ', cleaned)
    cleaned = RE_HTML.sub(' ', cleaned)
    cleaned = RE_TWITTER.sub(' ', cleaned)

    # Remove non-alphabetical characters
    cleaned = RE_SPECIAL_CHARS.sub(' ', cleaned)

    # Normalize whitespace
    cleaned = RE_MULTISPACE.sub(' ', cleaned).strip()

    return cleaned


def extract_linguistic_signals(raw_text: str) -> dict:
    """
    Extracts high-level stylistic and linguistic metrics for explainability:
    - Uppercase ratio (shouting/sensationalism indicator)
    - Exclamation and question mark frequency
    - Average word length
    - Total word count
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            "word_count": 0,
            "char_count": 0,
            "uppercase_ratio": 0.0,
            "exclamation_count": 0,
            "question_count": 0,
            "avg_word_length": 0.0,
        }

    words = raw_text.split()
    total_chars = len(raw_text)
    uppercase_chars = sum(1 for c in raw_text if c.isupper())
    exclamation_count = raw_text.count('!')
    question_count = raw_text.count('?')

    word_count = len(words)
    avg_word_len = sum(len(w) for w in words) / max(1, word_count)
    uppercase_ratio = (uppercase_chars / max(1, total_chars)) * 100

    return {
        "word_count": word_count,
        "char_count": total_chars,
        "uppercase_ratio": round(uppercase_ratio, 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "avg_word_length": round(avg_word_len, 2),
    }
