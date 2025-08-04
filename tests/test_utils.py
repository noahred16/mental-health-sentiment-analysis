import pytest
import pandas as pd
import os
from unittest import mock
import utils


def test_preprocess_text_basic():
    cases = [
        ("Hello World!", "hello world"),
        ("  Multiple   spaces   ", "multiple spaces"),
        ("1234 &*()@# Test", "1234 test"),
        (None, ""),
        ("", ""),
        ("Special characters !@#$%^&*()", "special characters"),
    ]
    for input_text, expected_output in cases:
        assert utils.preprocess_text(input_text) == expected_output


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
    assert first_row["processed_text"] == "oh my gosh"
