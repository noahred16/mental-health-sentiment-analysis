import pytest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the methods to test
import methods.tfidf as tfidf
import methods.lstm as lstm
import methods.RoBERTa as roberta


class TestTFIDFMethod:
    """Test cases for TF-IDF method"""

    def test_tfidf_checkpoint_loading_exists(self):
        """Test that TF-IDF loads existing checkpoint correctly"""
        checkpoint_path = "saved_models/tfidf_model.pkl"

        if os.path.exists(checkpoint_path):
            # Test loading existing checkpoint
            result = tfidf.train_tfidf_model(None, None, checkpoint_path)

            assert result is not None
            assert len(result) == 2  # Should return (model, label_mapping)
            model, label_mapping = result
            assert model is not None
            assert label_mapping is not None
            assert isinstance(label_mapping, dict)
        else:
            pytest.skip(
                "TF-IDF checkpoint not found - create one by training the model first"
            )

    def test_tfidf_load_model_function(self):
        """Test TF-IDF load_model function directly"""
        checkpoint_path = "saved_models/tfidf_model.pkl"

        if os.path.exists(checkpoint_path):
            model, label_mapping = tfidf.load_model(checkpoint_path)

            assert model is not None
            assert label_mapping is not None
            assert hasattr(model, "best_estimator_")
            assert hasattr(model.best_estimator_, "predict")
            assert hasattr(model.best_estimator_, "predict_proba")
        else:
            pytest.skip("TF-IDF checkpoint not found")

    def test_tfidf_demo_with_real_checkpoint(self):
        """Test TF-IDF demo function with real checkpoint"""
        checkpoint_path = "saved_models/tfidf_model.pkl"

        if os.path.exists(checkpoint_path):
            # Test demo with default texts
            results = tfidf.demo()

            if results is not None:  # Only test if model loaded successfully
                assert len(results) > 0

                # Check structure of first result
                result = results[0]
                expected_keys = {
                    "text",
                    "prediction",
                    "expected",
                    "correct",
                    "probabilities",
                }
                assert expected_keys.issubset(result.keys())

                # Check data types
                assert isinstance(result["text"], str)
                assert isinstance(result["prediction"], str)
                assert isinstance(result["expected"], str)
                assert isinstance(result["correct"], bool)
                assert isinstance(result["probabilities"], list)

                # Check probabilities are sorted (highest first)
                probs = [prob for _, prob in result["probabilities"]]
                assert probs == sorted(probs, reverse=True)
        else:
            pytest.skip("TF-IDF checkpoint not found")

    def test_tfidf_demo_custom_input(self):
        """Test TF-IDF demo with custom input"""
        checkpoint_path = "saved_models/tfidf_model.pkl"

        if os.path.exists(checkpoint_path):
            test_texts = ["I feel very anxious today"]
            expected_labels = ["Anxiety"]

            results = tfidf.demo(test_texts, expected_labels)

            if results is not None:
                assert len(results) == 1
                assert results[0]["text"] == "I feel very anxious today"
                assert results[0]["expected"] == "Anxiety"
                assert isinstance(results[0]["prediction"], str)
        else:
            pytest.skip("TF-IDF checkpoint not found")


class TestLSTMMethod:
    """Test cases for LSTM method"""

    def test_lstm_checkpoint_loading_exists(self):
        """Test that LSTM loads existing checkpoint correctly"""
        checkpoint_path = "saved_models/lstm_model.pth"

        if os.path.exists(checkpoint_path):
            model, vocab = lstm.load_model(checkpoint_path)

            assert model is not None
            assert vocab is not None
            assert hasattr(vocab, "label_encoder")
            assert hasattr(vocab, "word2idx")
            assert isinstance(vocab.label_encoder, dict)
        else:
            pytest.skip("LSTM checkpoint not found - train the model first")

    def test_lstm_predict_sentiment(self):
        """Test LSTM predict_sentiment function"""
        checkpoint_path = "saved_models/lstm_model.pth"

        if os.path.exists(checkpoint_path):
            model, vocab = lstm.load_model(checkpoint_path)

            # Test prediction
            test_text = "I feel anxious"
            prediction, probabilities = lstm.predict_sentiment(test_text, model, vocab)

            assert isinstance(prediction, str)
            assert prediction in vocab.label_encoder.keys()
            assert isinstance(
                probabilities, (list, tuple, np.ndarray)
            )  # Accept numpy arrays too
            assert len(probabilities) == len(vocab.label_encoder)

            # Probabilities should sum to approximately 1
            prob_sum = sum(probabilities)
            assert abs(prob_sum - 1.0) < 0.01
        else:
            pytest.skip("LSTM checkpoint not found")

    def test_lstm_demo_with_real_checkpoint(self):
        """Test LSTM demo function with real checkpoint"""
        checkpoint_path = "saved_models/lstm_model.pth"

        if os.path.exists(checkpoint_path):
            results = lstm.demo()

            if results is not None:
                assert len(results) > 0

                result = results[0]
                expected_keys = {
                    "text",
                    "prediction",
                    "expected",
                    "correct",
                    "probabilities",
                }
                assert expected_keys.issubset(result.keys())

                # Check data types
                assert isinstance(result["text"], str)
                assert isinstance(result["prediction"], str)
                assert isinstance(result["expected"], str)
                assert isinstance(result["correct"], bool)
                assert isinstance(result["probabilities"], list)

                # Check probabilities are sorted
                probs = [prob for _, prob in result["probabilities"]]
                assert probs == sorted(probs, reverse=True)
        else:
            pytest.skip("LSTM checkpoint not found")


class TestRoBERTaMethod:
    """Test cases for RoBERTa method"""

    def test_roberta_checkpoint_loading_exists(self):
        """Test that RoBERTa loads existing checkpoint correctly"""
        checkpoint_path = "saved_models/roberta_model.pth"

        if os.path.exists(checkpoint_path):
            model, tokenizer, label_encoder = roberta.load_model(checkpoint_path)

            assert model is not None
            assert tokenizer is not None
            assert label_encoder is not None
            assert hasattr(label_encoder, "classes_")
            assert hasattr(tokenizer, "__call__")  # Should be callable
        else:
            pytest.skip("RoBERTa checkpoint not found - train the model first")

    def test_roberta_predict_sentiment(self):
        """Test RoBERTa predict_sentiment function"""
        checkpoint_path = "saved_models/roberta_model.pth"

        if os.path.exists(checkpoint_path):
            model, tokenizer, label_encoder = roberta.load_model(checkpoint_path)

            # Test prediction
            test_text = "I feel anxious"
            prediction, probabilities = roberta.predict_sentiment(
                test_text, model, tokenizer, label_encoder
            )

            assert isinstance(prediction, str)
            assert prediction in label_encoder.classes_
            assert isinstance(
                probabilities, (list, tuple, np.ndarray)
            )  # Accept numpy arrays too
            assert len(probabilities) == len(label_encoder.classes_)

            # Probabilities should sum to approximately 1
            prob_sum = sum(probabilities)
            assert abs(prob_sum - 1.0) < 0.01
        else:
            pytest.skip("RoBERTa checkpoint not found")

    def test_roberta_demo_with_real_checkpoint(self):
        """Test RoBERTa demo function with real checkpoint"""
        checkpoint_path = "saved_models/roberta_model.pth"

        if os.path.exists(checkpoint_path):
            results = roberta.demo()

            if results is not None:
                assert len(results) > 0

                result = results[0]
                expected_keys = {
                    "text",
                    "prediction",
                    "expected",
                    "correct",
                    "probabilities",
                }
                assert expected_keys.issubset(result.keys())

                # Check data types
                assert isinstance(result["text"], str)
                assert isinstance(result["prediction"], str)
                assert isinstance(result["expected"], str)
                assert isinstance(result["correct"], bool)
                assert isinstance(result["probabilities"], list)

                # Check probabilities are sorted
                probs = [prob for _, prob in result["probabilities"]]
                assert probs == sorted(probs, reverse=True)
        else:
            pytest.skip("RoBERTa checkpoint not found")


class TestDemoIntegration:
    """Integration tests for demo functions"""

    def test_all_demo_functions_consistent_format(self):
        """Test that all available demo functions return consistent format"""
        expected_keys = {"text", "prediction", "expected", "correct", "probabilities"}

        # Test each method that has a checkpoint
        methods_to_test = []

        if os.path.exists("saved_models/tfidf_model.pkl"):
            methods_to_test.append(("TF-IDF", tfidf.demo))

        if os.path.exists("saved_models/lstm_model.pth"):
            methods_to_test.append(("LSTM", lstm.demo))

        if os.path.exists("saved_models/roberta_model.pth"):
            methods_to_test.append(("RoBERTa", roberta.demo))

        if not methods_to_test:
            pytest.skip("No model checkpoints found")

        # Test with same input
        test_texts = ["I feel very anxious and worried"]
        expected_labels = ["Anxiety"]

        for method_name, demo_func in methods_to_test:
            results = demo_func(test_texts, expected_labels)

            if results is not None:  # Skip if model failed to load
                assert len(results) == 1, f"{method_name} should return 1 result"

                result = results[0]
                assert isinstance(result, dict), f"{method_name} should return dict"
                assert expected_keys.issubset(
                    result.keys()
                ), f"{method_name} missing required keys"

                # Verify data types
                assert isinstance(result["text"], str)
                assert isinstance(result["prediction"], str)
                assert isinstance(result["expected"], str)
                assert isinstance(result["correct"], bool)
                assert isinstance(result["probabilities"], list)

                # Verify probabilities are properly sorted
                probs = [prob for _, prob in result["probabilities"]]
                assert probs == sorted(
                    probs, reverse=True
                ), f"{method_name} probabilities not sorted"

    def test_demo_functions_with_multiple_inputs(self):
        """Test demo functions with multiple test cases"""
        test_texts = [
            "I feel anxious and worried",
            "Today is a beautiful day",
            "I don't want to exist anymore",
        ]
        expected_labels = ["Anxiety", "Normal", "Depression"]

        # Test each available method
        if os.path.exists("saved_models/tfidf_model.pkl"):
            results = tfidf.demo(test_texts, expected_labels)
            if results is not None:
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result["text"] == test_texts[i]
                    assert result["expected"] == expected_labels[i]

        if os.path.exists("saved_models/lstm_model.pth"):
            results = lstm.demo(test_texts, expected_labels)
            if results is not None:
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result["text"] == test_texts[i]
                    assert result["expected"] == expected_labels[i]

        if os.path.exists("saved_models/roberta_model.pth"):
            results = roberta.demo(test_texts, expected_labels)
            if results is not None:
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result["text"] == test_texts[i]
                    assert result["expected"] == expected_labels[i]


class TestCheckpointConsistency:
    """Tests for checkpoint consistency and error handling"""

    def test_checkpoint_paths_different(self):
        """Test that different methods use different checkpoint paths"""
        tfidf_path = "saved_models/tfidf_model.pkl"
        lstm_path = "saved_models/lstm_model.pth"
        roberta_path = "saved_models/roberta_model.pth"

        # Verify paths are different to avoid conflicts
        paths = [tfidf_path, lstm_path, roberta_path]
        assert len(set(paths)) == 3, "Checkpoint paths should be unique"

    def test_missing_checkpoint_handling(self):
        """Test how methods handle missing checkpoints"""
        fake_tfidf_path = "/tmp/fake_tfidf.pkl"
        fake_lstm_path = "/tmp/fake_lstm.pth"
        fake_roberta_path = "/tmp/fake_roberta.pth"

        # TF-IDF should handle missing checkpoint gracefully in demo
        result = tfidf.demo()
        if not os.path.exists("saved_models/tfidf_model.pkl"):
            assert result is None

        # LSTM should handle missing checkpoint gracefully in demo
        result = lstm.demo()
        if not os.path.exists("saved_models/lstm_model.pth"):
            assert result is None

        # RoBERTa should handle missing checkpoint gracefully in demo
        result = roberta.demo()
        if not os.path.exists("saved_models/roberta_model.pth"):
            assert result is None

        # Direct load_model calls should raise appropriate exceptions
        with pytest.raises(Exception):  # FileNotFoundError or similar
            lstm.load_model(fake_lstm_path)

        with pytest.raises(FileNotFoundError):
            roberta.load_model(fake_roberta_path)

    def test_saved_models_directory_exists(self):
        """Test that saved_models directory exists"""
        assert os.path.exists("saved_models"), "saved_models directory should exist"
        assert os.path.isdir("saved_models"), "saved_models should be a directory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
