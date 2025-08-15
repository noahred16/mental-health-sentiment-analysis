import methods.lstm as lstm
import methods.tfidf as tfidf
import utils

# For the demo, we'll probably want to save our trained models and just load them in here.

# We should try to make it so that we all use the same evaluation metrics and format.
# So our methods should make sure to accept the data as parameters for training and testing.
# Because we'll want to make sure that we use the same 80/20 split.
# Alternatively we could use a random seed for reproducibility to ensure consistent results.

# We also hopefully can benefit from sharing the same data preprocessing functions.

# We can print out the results of each method here.

# Describe the data

print("=========================================================================")
print("============================ Dataset Description ========================")
print("=========================================================================")
data = utils.load_data()
utils.describe_data(data)

# Method 1: LSTM
print("=========================================================================")
print("=============================== LSTM Method =============================")
print("=========================================================================")
lstm.demo()

# Method 2: TF-IDF
print("=========================================================================")
print("============================== TF-IDF Method ============================")
print("=========================================================================")
tfidf.demo()

# Method 3
