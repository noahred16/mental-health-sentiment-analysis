import pytest
import pandas as pd
import os
from unittest import mock
import utils


def test_preprocess_text_basic():
    cases = [
        # lowercase and punctuation removal
        ("Hello World!", "hello world"),
        # "spaces" -> "space" after lemmatization
        ("  Multiple   spaces   ", "multiple space"),
        # keep alphanumeric, remove URLs and emails
        ("1234 &*()@# hi https://Test.com", "1234 hi"),
        (None, ""),
        ("", ""),
    ]
    for input_text, expected_output in cases:
        assert utils.preprocessor.preprocess_text(input_text) == expected_output


def test_load_data():
    data = utils.load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

    # assert statement and status of first row
    assert "statement" in data.columns
    assert "status" in data.columns
    assert "processed_text" in data.columns

    first_row = data.iloc[0]
    assert first_row["statement"] == "oh my gosh"
    assert first_row["status"] == "Anxiety"
    assert first_row["processed_text"] == "oh gosh"  # "my" is a stop word
