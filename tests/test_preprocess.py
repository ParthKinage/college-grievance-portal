"""Tests for text cleaning."""

from src.preprocess import clean_text


def test_lowercases_text():
    """Uppercase letters become lowercase."""
    assert clean_text("My RESULT Is LATE") == "my result is late"


def test_removes_urls_and_emails():
    """URLs and email addresses are removed completely."""
    result = clean_text("See https://portal.example.com/result or mail help@example.com now")
    assert "http" not in result
    assert "example" not in result
    assert result == "see or mail now"


def test_removes_punctuation_and_numbers():
    """Punctuation, special characters and digits are stripped."""
    assert clean_text("Fees paid twice!!! (Rs. 45000) #angry :(") == "fees paid twice rs angry"


def test_collapses_whitespace():
    """Multiple spaces, tabs and newlines become one space."""
    assert clean_text("  too    many \n\t spaces  ") == "too many spaces"


def test_handles_empty_and_non_string_input():
    """Empty strings and non-strings return an empty string, not an error."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(123) == ""
