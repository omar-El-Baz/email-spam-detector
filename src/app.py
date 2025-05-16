import streamlit as st
import joblib
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="Email Spam Detector", layout="wide")


# --- Feature Definitions (CRITICAL: Must match your training data order) ---
# These are the 57 features in the order your model expects them.
# This order is derived from the Spambase dataset description.
FEATURE_NAMES_IN_ORDER = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 
    'char_freq_$', 'char_freq_#', 
    'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'
]

# Extract the specific words and characters to look for
WORDS_TO_COUNT = [name.split("word_freq_")[1] for name in FEATURE_NAMES_IN_ORDER if name.startswith("word_freq_")]
CHARS_TO_COUNT = [name.split("char_freq_")[1] for name in FEATURE_NAMES_IN_ORDER if name.startswith("char_freq_")]

# --- Feature Extraction Function ---
def extract_features_from_text(email_text: str) -> np.ndarray | None:
    """
    Extracts the 57 Spambase features from raw email text.
    Returns a 2D numpy array (1, 57) or None if text is empty.
    """
    if not email_text or not email_text.strip():
        return None

    features = []
    
    original_text_for_caps = email_text # Preserve case for capital run length
    email_text_lower = email_text.lower()

    # Tokenize words: using a simple regex for alphanumeric sequences
    all_words_in_email = re.findall(r'[a-zA-Z0-9]+', email_text_lower)
    total_words = len(all_words_in_email)
    # If total_words is 0, all word frequencies will be 0. Avoid division by zero.
    total_words_divisor = total_words if total_words > 0 else 1

    # 1. Word Frequencies (48 features)
    for word_to_find in WORDS_TO_COUNT:
        count = all_words_in_email.count(word_to_find)
        features.append((count / total_words_divisor) * 100)

    # 2. Character Frequencies (6 features)
    total_chars = len(email_text_lower)
    # If total_chars is 0, all char frequencies will be 0. Avoid division by zero.
    total_chars_divisor = total_chars if total_chars > 0 else 1
    
    for char_to_find in CHARS_TO_COUNT:
        count = email_text_lower.count(char_to_find)
        features.append((count / total_chars_divisor) * 100)
        
    # 3. Capital Run Lengths (3 features)
    # Find all sequences of 1 or more uppercase letters
    cap_runs = re.findall(r'[A-Z]+', original_text_for_caps)
    run_lengths = [len(run) for run in cap_runs]

    if run_lengths:
        capital_run_length_average = np.mean(run_lengths)
        capital_run_length_longest = np.max(run_lengths)
    else:
        capital_run_length_average = 0
        capital_run_length_longest = 0
    
    # Total number of capital letters in the e-mail
    capital_run_length_total = sum(1 for char in original_text_for_caps if char.isupper())

    features.append(capital_run_length_average)
    features.append(capital_run_length_longest)
    features.append(capital_run_length_total)
    
    if len(features) != 57:
        # This should not happen if FEATURE_NAMES_IN_ORDER is correct
        st.error(f"Critical Error: Extracted {len(features)} features, but expected 57. Check feature definitions.")
        return None

    return np.array(features).reshape(1, -1)

# --- Load Model ---
@st.cache_resource # Cache the model loading
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure the model is in the 'models' sub-directory.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model("/Users/omar/Desktop/College/Semester 4/Intelligent Programming/Projects/email-spam-detector/models/rf_best_model.pkl")  # Adjust the path if necessary

# --- Streamlit UI ---
st.title("📧 Email Spam Detector")
st.markdown("""
Enter the text of an email below. The model will predict whether it's **Spam** or **Not Spam**.
This model was trained on the Spambase dataset and uses a RandomForestClassifier.
""")

email_input = st.text_area("Paste Email Text Here:", height=250, placeholder="Dear friend, you have won a prize...")

if st.button("🔎 Classify Email", type="primary"):
    if not model:
        st.warning("Model is not loaded. Cannot classify.")
    elif not email_input or not email_input.strip():
        st.warning("Please enter some email text to classify.")
    else:
        with st.spinner("Analyzing email..."):
            # 1. Extract features
            features_vector = extract_features_from_text(email_input)
            
            if features_vector is not None:
                try:
                    # 2. Make prediction (model is a pipeline, handles scaling)
                    prediction = model.predict(features_vector)
                    probability = model.predict_proba(features_vector)

                    # 3. Display result
                    is_spam = (prediction[0] == 1)
                    
                    st.subheader("Classification Result:")
                    if is_spam:
                        spam_prob = probability[0][1] * 100
                        st.error(f"🚨 This email is classified as: SPAM (Confidence: {spam_prob:.2f}%)")
                    else:
                        not_spam_prob = probability[0][0] * 100
                        st.success(f"✅ This email is classified as: NOT SPAM (Confidence: {not_spam_prob:.2f}%)")

                    # Optional: Display a snippet of extracted features for verification
                    # with st.expander("Show Extracted Features (sample)"):
                    #     st.write("First 10 extracted feature values (out of 57):")
                    #     st.json(dict(zip(FEATURE_NAMES_IN_ORDER[:10], features_vector[0][:10].round(4).tolist())))

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                # This case should be caught by the earlier check, but as a safeguard:
                st.warning("Could not extract features from the provided text (perhaps it's empty).")

st.markdown("---")
st.markdown("Developed as part of a project to automate email classification.")