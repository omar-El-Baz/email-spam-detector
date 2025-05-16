# Email Spam Detector

## Project Overview

This project implements a machine learning model to automatically classify emails as "Spam" or "Not Spam." The goal is to provide an automated solution for filtering unwanted emails, thereby improving user productivity and inbox management. The model is trained on the Spambase dataset from the UCI Machine Learning Repository and deployed as an interactive web application using Streamlit.

**Live Application:** https://email-spam-detector-intelligent-programming.streamlit.app/

---

## Table of Contents
1.  [Scenario & Objectives](#scenario--objectives)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Feature Set](#feature-set)
    *   [Model Training & Selection](#model-training--selection)
    *   [Hyperparameter Tuning](#hyperparameter-tuning)
4.  [Performance Evaluation](#performance-evaluation)
    *   [Best Performing Model & Hyperparameters](#best-performing-model--hyperparameters)
    *   [Performance on Test Set](#performance-on-test-set)
    *   [Discussion of Metrics](#discussion-of-metrics)
5.  [Technology Stack](#technology-stack)
6.  [Setup and Usage](#setup-and-usage)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Running the Streamlit App Locally](#running-the-streamlit-app-locally)
---

## 1. Scenario & Objectives

**Scenario:** An incoming email needs to be automatically classified as "Spam" or "Not Spam" to assist users in managing their email inboxes effectively.

**Objectives:**
*   Automate the classification of emails to filter out spam.
*   Develop a machine learning model capable of accurately distinguishing between spam and legitimate (ham) emails.
*   Provide an accessible interface for users to test the model with new email text.

---

## 2. Dataset

*   **Name:** Spambase Dataset
*   **Source:** UCI Machine Learning Repository ([Link](https://archive.ics.uci.edu/ml/datasets/spambase))
*   **Description:** The dataset consists of 4601 email messages, which have been pre-processed into 57 continuous and binary attributes. These attributes represent frequencies of specific words and characters, as well as statistics about sequences of capital letters. The final attribute is the class label, indicating whether the email was considered spam (1) or not spam (0).
*   **Original Instances:** 4601
*   **Features:** 57 (excluding the class label)
*   **Data Files Used:**
    *   `spambase.data`: The raw feature data.
    *   `spambase.names`: Contains the names for the 58 columns.

---

## 3. Methodology

### Data Preprocessing
1.  The `spambase.data` file (containing feature values) was loaded into a pandas DataFrame.
2.  Column names were assigned using the information from `spambase.names`.
3.  Duplicate rows were identified and removed from the dataset.
    *   Initially, 391 duplicate rows were found.
    *   After removal, the dataset contained 4210 unique instances. *(Verify this number from your `df.shape` after `drop_duplicates()` in `data_preprocessing.ipynb`)*
4.  The cleaned dataset was saved as `spambase_cleaned.csv` for subsequent model training steps.

### Feature Set
The project utilized the 57 pre-engineered features provided in the Spambase dataset. These include:
*   48 features representing the percentage of words in an email that match a given word (e.g., `word_freq_make`, `word_freq_free`).
*   6 features representing the percentage of characters in an email that match a given character (e.g., `char_freq_!`, `char_freq_$`).
*   3 features related to capital letters: average length of uninterrupted sequences, length of the longest sequence, and total number of capital letters.

### Model Training & Selection
1.  **Train-Test Split:** The cleaned dataset (`spambase_cleaned.csv`) was split into training (80%) and testing (20%) sets. The split was stratified by the target variable (`is_spam`) to ensure similar class proportions in both sets, using `random_state=42` for reproducibility.
2.  **Pipelines:** Scikit-learn `Pipeline` objects were used to streamline the workflow. Each pipeline consisted of:
    *   `StandardScaler()`: To scale the numeric features.
    *   A classifier model.
3.  **Models Evaluated:**
    *   Gaussian Naive Bayes (GNB)
    *   Logistic Regression (LR)
    *   Random Forest Classifier (RF)

### Hyperparameter Tuning
*   **Method:** `GridSearchCV` from scikit-learn was used for comprehensive hyperparameter optimization.
*   **Cross-Validation:** 5-fold cross-validation was applied during the grid search.
*   **Scoring Metric:** The `F1-score` (for the positive class, spam) was chosen as the primary metric for optimization, as it provides a good balance between precision and recall, especially relevant for potentially imbalanced datasets.
*   **Parameter Grids:**
    *   **GNB:** `classifier__var_smoothing`: `[1e-9, 1e-8, 1e-7]`
    *   **LR:** `classifier__C`: `[0.01, 0.1, 1, 10]`, `classifier__penalty`: `['l2']`, `classifier__solver`: `['lbfgs']`
    *   **RF:** `classifier__n_estimators`: `[50, 100, 200]`, `classifier__max_depth`: `[None, 10, 20]`, `classifier__min_samples_leaf`: `[1, 2, 4]`

---

## 4. Performance Evaluation

### Best Performing Model & Hyperparameters
The **Random Forest Classifier** emerged as the best-performing model based on the cross-validated F1-score on the training data.

*   **Optimal Hyperparameters (Random Forest):**
    *   `classifier__max_depth`: 20
    *   `classifier__min_samples_leaf`: 1
    *   `classifier__n_estimators`: 100
*   **Best CV F1-Score (during training):** 0.932

The trained Random Forest pipeline (including the scaler) was exported to `models/rf_best_model.pkl` using `joblib`.

### Performance on Test Set
The tuned Random Forest model was evaluated on the hold-out test set:

**Classification Report (Random Forest - Test Set):**
precision    recall  f1-score   support

0     0.9491    0.9585    0.9538       506
1     0.9366    0.9226    0.9295       336

accuracy                         0.9442       842


**Confusion Matrix (Random Forest - Test Set):**
[[485 21]
[ 26 310]]

Where:
*   True Negatives (Correctly Not Spam): 485
*   False Positives (Not Spam classified as Spam / Type I Error): 21
*   False Negatives (Spam classified as Not Spam / Type II Error): 26
*   True Positives (Correctly Spam): 310

### Discussion of Metrics
*   **Accuracy (94.42%):** The model correctly classifies approximately 94.4% of the emails in the test set.
*   **Precision (Spam - 0.9366):** When the model predicts an email is spam, it is correct 93.66% of the time. A high precision for the spam class is important to minimize the number of legitimate emails (ham) being incorrectly flagged as spam (False Positives).
*   **Recall (Spam - 0.9226):** The model successfully identifies 92.26% of all actual spam emails in the test set. A high recall for the spam class ensures that most spam messages are caught (minimizing False Negatives).
*   **F1-Score (Spam - 0.9295):** This harmonic mean of precision and recall for the spam class indicates a strong and balanced performance in identifying spam.
*   The low number of False Positives (21) is desirable, ensuring fewer legitimate emails are misclassified. The 26 False Negatives indicate some spam might still pass through, but the overall performance is robust.

---

## 5. Technology Stack
*   **Python 3.x**
*   **Pandas:** For data manipulation and loading.
*   **Scikit-learn:** For machine learning tasks (train-test split, scaling, models, pipelines, GridSearchCV, metrics).
*   **Joblib:** For saving and loading the trained model.
*   **Streamlit:** For creating the interactive web application.
*   **Jupyter Notebooks:** For data exploration, model development, and experimentation.


---

## 6. Setup and Usage

### Prerequisites
*   Python 3.7+
*   pip (Python package installer)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/omar-El-Baz/email-spam-detector.git
    cd email-spam-detector
    ```
2.  (Recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Streamlit App Locally
1.  Ensure you are in the project root directory where `app.py` is located.
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

---
