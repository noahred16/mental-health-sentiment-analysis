import methods.lstm as lstm
import methods.tfidf as tfidf
import methods.RoBERTa as roberta
import utils
import pandas as pd

# For the demo, we'll probably want to save our trained models and just load them in here.

# We should try to make it so that we all use the same evaluation metrics and format.
# So our methods should make sure to accept the data as parameters for training and testing.
# Because we'll want to make sure that we use the same 80/20 split.
# Alternatively we could use a random seed for reproducibility to ensure consistent results.

# We also hopefully can benefit from sharing the same data preprocessing functions.

# We can print out the results of each method here.


def compare_model_predictions():
    """Compare predictions from all three models side by side"""
    print("Running model comparisons...")
    
    # Get results from all models
    tfidf_results = tfidf.demo()
    lstm_results = lstm.demo()
    roberta_results = roberta.demo()
    
    # Check which models are available
    available_models = []
    if tfidf_results is not None:
        available_models.append(("TF-IDF", tfidf_results))
    if lstm_results is not None:
        available_models.append(("LSTM", lstm_results))
    if roberta_results is not None:
        available_models.append(("RoBERTa", roberta_results))
    
    if not available_models:
        print("No trained models available")
        return
    
    print(f"Comparing {len(available_models)} available models: {', '.join([name for name, _ in available_models])}")
    
    # Create comparison table
    comparison_data = []
    for i in range(len(available_models[0][1])):  # Number of test cases
        text = available_models[0][1][i]["text"]
        expected = available_models[0][1][i]["expected"]
        
        row_data = {
            "Text": text[:40] + "..." if len(text) > 40 else text,
            "Expected": expected,
        }
        
        # Add predictions for each available model
        for model_name, results in available_models:
            if i < len(results):
                result = results[i]
                pred = result["prediction"]
                correct = "✓" if result["correct"] else "✗"
                conf = result["probabilities"][0][1]  # Top probability
                
                row_data[f"{model_name} Pred"] = pred
                row_data[f"{model_name} ✓"] = correct
                row_data[f"{model_name} Conf"] = f"{conf:.3f}"
        
        comparison_data.append(row_data)
    
    # Display as table
    df = pd.DataFrame(comparison_data)
    print("\n" + "="*140)
    print("MODEL COMPARISON RESULTS")
    print("="*140)
    print(df.to_string(index=False))
    
    # Show detailed probabilities for each prediction
    print("\n" + "="*140)
    print("DETAILED PROBABILITIES")
    print("="*140)
    
    for i in range(len(available_models[0][1])):
        text = available_models[0][1][i]["text"]
        expected = available_models[0][1][i]["expected"]
        
        print(f"\nText {i+1}: '{text}'")
        print(f"Expected: {expected}")
        
        for model_name, results in available_models:
            if i < len(results):
                result = results[i]
                print(f"\n{model_name} Predictions (sorted by confidence):")
                for label, prob in result["probabilities"]:
                    marker = " ← PREDICTED" if label == result["prediction"] else ""
                    print(f"  {label}: {prob:.4f}{marker}")
        
        print("-" * 100)

# Describe the data

print("=========================================================================")
print("============================ Dataset Description ========================")
print("=========================================================================")
data = utils.load_data()
utils.describe_data(data)

# Run model comparison
compare_model_predictions()

# Method 3
