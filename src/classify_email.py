#!/usr/bin/env python3
"""
classify_email.py

Loads the trained spam-detection pipeline and classifies
input email text (pre‐tokenized numeric features assumed).

Usage (command‐line):
  $ echo "raw email text here" | python classify_email.py
"""
import sys
import joblib
import numpy as np

MODEL_PATH = "/Users/omar/Desktop/College/Semester 4/Intelligent Programming/Projects/email-spam-detector/models/rf_best_model.pkl"  # adjust if your best model was different

def extract_features_from_text(text: str) -> np.ndarray:
    """
    Placeholder for feature extraction.
    Since this project uses the UCI Spambase numeric features,
    you would replace this with the same transformations
    you applied to raw text when building the dataset.
    """
    raise NotImplementedError("Convert raw text into the 57 numeric features.")

def classify_text(text: str) -> str:
    """Load the pipeline, transform text, and predict."""
    # 1. Load the trained model pipeline
    pipeline = joblib.load(MODEL_PATH)

    # 2. Convert raw text to numeric feature vector
    features = extract_features_from_text(text)
    features = features.reshape(1, -1)  # single sample

    # 3. Predict (0 = not spam, 1 = spam)
    label = pipeline.predict(features)[0]

    return "Spam" if label == 1 else "Not Spam"

def main():
    # Read entire stdin as the email body
    raw_email = sys.stdin.read().strip()
    if not raw_email:
        print("Please provide email text via stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        result = classify_text(raw_email)
        print(result)
    except NotImplementedError:
        print("Feature‐extraction function is not yet implemented.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
