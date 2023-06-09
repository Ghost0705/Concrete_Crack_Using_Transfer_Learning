# Ecommerce Sentiment Analysis using Bidirectional LSTM

Creating a sentiment analysis for Ecommerce using deep learning approach with Bidirectional LSTM neural network

### Description
Objective: Create a deep learning model using Bidirectional LSTM neural
network to create a sentiment analysis for Ecommerce dataset

* Model training - Deep learning
* Method: Sequential, Embedding, Tokenizer, LabelEncoder, Bidirectional LSTM
* Module: Sklearn & Tensorflow

In this analysis, dataset used from https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification

### About The Dataset:
The dataset used in this analysis is ecommerce dataset that contain 4 categories and 50425 text.

To do the sentiment analysis, the dataset used tokenizer and label encoder. There are 1 missing data and many duplicated found in the dataset. the data cleaning process were applied.

### Deep learning model with Bidirectional LSTM layer
A sequential model was created with 1 Embedding layer, 1 Bidirectional LSTM layer, 2 Dense layer:
<p align="center">
  <img src="https://github.com/Ghost0705/Ecommerce_Sentiment_Analysis_Bidirectional_LSTM/blob/main/image/architecture.png">
</p>

Data were trained with 10 epoch:
<p align="center">
  <img src="https://github.com/Ghost0705/Ecommerce_Sentiment_Analysis_Bidirectional_LSTM/blob/main/image/model_training.png">
</p>


After the deployment of model accuracy able to achieve 99% and f1_score of 0.87%. The model is good enough to be used to predict the text according to its categories 

### How to run the pythons file:
1. Load the review_handler.py
2. Run the ecommerce_text_classification.py 

Enjoy!

