# SMS Spam Detector

A machine learning application that classifies SMS messages as **Spam** or **Legitimate** using Natural Language Processing (NLP) and supervised learning.

## Overview

This project uses a Logistic Regression classifier trained on the SMS Spam Collection dataset. It applies TF-IDF vectorization to convert text messages into numerical features before classification.

Users can enter their own messages through an interactive Streamlit interface and receive a prediction with an estimated class probability.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Machine Learning Workflow

1. Download and load the SMS Spam Collection dataset.
2. Split the data into training and testing sets.
3. Convert text messages into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Evaluate the model using accuracy, precision, recall, and F1-score.
6. Classify new messages through the Streamlit interface.

## Project Structure

```text
spam-detector/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd spam-detector
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

The application will automatically download the dataset if it is not already available.

## Dataset

The project uses the SMS Spam Collection dataset provided by the UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Learning Objectives

This project demonstrates practical implementation of:

- Supervised machine learning
- Natural Language Processing
- Text preprocessing and feature extraction
- Classification algorithms
- Model evaluation
- Interactive ML application development

## Disclaimer

This is an educational machine learning project. Predictions are not guaranteed to be correct and should not be used as the sole basis for identifying malicious messages.