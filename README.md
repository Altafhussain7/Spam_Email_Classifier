# Spam Email Classifier

## Overview

This project uses Machine Learning to classify messages as Spam or Ham (Not Spam).

The model analyzes the text of an email or SMS message and predicts whether it is spam.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Joblib

## Project Workflow

1. Load the dataset
2. Clean and prepare the data
3. Convert labels into numerical values
4. Split the dataset into training and testing data
5. Convert text into numerical features using TF-IDF
6. Train the Multinomial Naive Bayes model
7. Evaluate the model performance
8. Save the trained model

## Machine Learning Model

Algorithm used:

Multinomial Naive Bayes

Reason for choosing this algorithm:

Naive Bayes performs well for text classification problems and is fast to train.

## Files Description

| File | Description |
|---|---|
| spam_classifier.ipynb | Main project notebook |
| spam.csv | Dataset used for training |
| model.pkl | Saved trained machine learning model |
| vectorizer.pkl | Saved TF-IDF vectorizer |
| requirements.txt | Required Python libraries |

## How to Run

1. Install required libraries:
