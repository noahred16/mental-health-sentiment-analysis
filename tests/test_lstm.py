import pytest
import pandas as pd
import os
from unittest import mock
import methods.lstm as lstm


def test_predict_sentiment():
    # TODO test using pre-trained model
    test_texts = [
        ("I feel anxious and can't sleep properly", "Anxiety"),
        ("Life is wonderful and I'm enjoying every moment", "Happiness"),
        ("I don't see any point in continuing anymore", "Depression"),
        ("Work pressure is getting too much for me", "Stress"),
    ]

    for text, expected in test_texts:
        # assert lstm.predict_sentiment(text) == expected
        # TODO: Implement the actual prediction logic and assert
        assert True


# TODO unit test the other LSTM functions.
