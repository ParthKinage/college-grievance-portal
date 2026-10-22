"""
Text preprocessing.

Machine learning models work better when text is consistent, so we clean
every complaint the same way before training and before prediction.
Stop words ("the", "is", "and") are removed later by the TF-IDF vectorizer.
"""

import re

# Pre-compiled regular expressions (compiled once, reused for every call)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")   # http://..., https://..., www....
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")          # name@example.com
NUMBER_PATTERN = re.compile(r"\d+")                   # any run of digits
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")          # anything that is not a-z or space
SPACES_PATTERN = re.compile(r"\s+")                   # one or more whitespace characters


def clean_text(text: str) -> str:
    """Lowercase text and remove URLs, emails, numbers, punctuation and extra spaces."""
    # Non-string input (None, NaN, numbers) becomes an empty string instead of crashing
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # URLs and emails must be removed before punctuation, otherwise
    # "www.site.com" would turn into the meaningless word "wwwsitecom"
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)
    text = NON_LETTER_PATTERN.sub(" ", text)
    text = SPACES_PATTERN.sub(" ", text)
    return text.strip()
