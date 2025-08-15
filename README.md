# Mental Health Sentiment Analysis - CS6620 NLP Course Project
Description

**Team members:**  
Noah Smith  
Deepashree Srinivasa Rao Rannore  
Rajkesh Prakash Shetty  


## Files
```
.
├── README.md
├── data
│   └── mental_health_sentiment_analysis.csv
├── demo.py
├── methods
│   ├── lstm.py
│   ├── tfidf.py
│   └── RoBERTa.py
├── metrics
├── requirements.txt
├── saved_models
│   ├── lstm_model.pkl
│   ├── tfidf_model.pkl
│   └── RoBERTa_model.pth
├── tests
│   └── test_utils.py
└── utils.py
```

## Setup
- `pip install -r requirements.txt`
- `python demo.py`

## Commands
- `python demo.py`: Run the demo script to see the results of the pre-trained models.
- `pytest`: Run the tests.
- `python -m methods.lstm`: Run the LSTM model training and evaluation.
- `python -m methods.tfidf`: Run the TF-IDF model training and evaluation.


## Methods
We implemented three methods for sentiment analysis:

### LSTM-based Neural Network
- File: `methods/lstm.py`
- Description: This method uses an LSTM-based neural network to classify the sentiment of mental health
- Training: `python -m methods.lstm`
- Model Saving: The trained model is saved as `saved_models/lstm_model.pth`

### TF-IDF with Classical ML Classifiers
- File: `methods/tfidf.py`
- Description: This method uses TF-IDF vectorization combined with classical machine learning classifiers like Logistic Regression, SVM, etc.
- Training: `python -m methods.tfidf`
- Model Saving: The trained models are saved in the `saved_models/` directory.



