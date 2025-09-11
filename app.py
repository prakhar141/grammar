import streamlit as st
import pickle
from difflib import get_close_matches
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import requests
import os

# ------------------------
# Helper to download file if not exists
# ------------------------
def download_file(url, save_path):
    if not os.path.exists(save_path):
        r = requests.get(url, stream=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return save_path

# ------------------------
# Q-learning Spelling Corrector
# ------------------------
q_learning_url = "https://huggingface.co/prakhar146/grammar/resolve/main/q_table.pkl"
q_learning_file = download_file(q_learning_url, "q_table.pkl")

with open(q_learning_file, "rb") as f:
    model_data = pickle.load(f)

# Try multiple options to extract Q and all_words
if isinstance(model_data, dict):
    if "q_table" in model_data:
        Q = model_data["q_table"]
        all_words = model_data.get("all_words", [])
    elif "Q_table" in model_data:
        Q = model_data["Q_table"]
        all_words = model_data.get("all_words", [])
    elif "Q" in model_data:
        Q = model_data["Q"]
        all_words = model_data.get("all_words", [])
    else:
        # maybe dict itself *is* the Q table
        Q = model_data
        all_words = sorted({a for actions in Q.values() for a in actions.keys()})
else:
    # If pickle is directly the Q-table (not dict wrapper)
    Q = model_data
    all_words = sorted({a for actions in Q.values() for a in actions.keys()})

def word_to_state(word: str):
    """Convert word into a state representation (sorted 2-grams)."""
    if not word:
        return ()
    ngrams = [word[i:i+2] for i in range(len(word)-1)]
    return tuple(sorted(ngrams))

def predict_word(state_word, Q, all_words):
    """Predict the corrected word using Q-table or fallback to close matches."""
    if not state_word:
        return state_word
    state = word_to_state(state_word)
    if state in Q and Q[state]:
        return max(Q[state], key=Q[state].get)
    candidates = get_close_matches(state_word, all_words, n=1)
    return candidates[0] if candidates else state_word
# ------------------------
# Post-processing helpers
# ------------------------
custom_corrections = {
    "teh": "the",
    "Teh": "The",
    "recieve": "receive",
    "adress": "address",
    "enviroment": "environment",
    "acommodate": "accommodate"
}

def apply_custom_corrections(word):
    return custom_corrections.get(word, word)

def remove_duplicate_words(text):
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1].lower() != w.lower():
            cleaned.append(w)
    return " ".join(cleaned)

# ------------------------
# T5 Grammar Model
# ------------------------
t5_repo = "prakhar146/grammar"
tokenizer = T5Tokenizer.from_pretrained(t5_repo, use_fast=True)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_repo)

# Force CPU to avoid "meta tensor" errors
device = torch.device("cpu")
t5_model.to(device)
t5_model.eval()

def correct_sentence(sentence, max_length=128):
    input_text = "grammar: " + sentence
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = t5_model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# Full Spelling Correction Pipeline
# ------------------------
def spelling_correction_pipeline(sentence):
    words = sentence.split()
    corrected_words = [apply_custom_corrections(predict_word(w, Q, all_words)) for w in words]
    corrected_sentence = " ".join(corrected_words)
    return remove_duplicate_words(corrected_sentence)

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📝 Spelling & Grammar Corrector", page_icon="✨", layout="centered")

st.title("📝 Spelling & Grammar Corrector")
st.markdown("### 🔍 Fix your text instantly with AI-powered spelling & grammar correction!")

# Mode selection
option = st.radio(
    "Choose correction mode:",
    ("🅰️  Spelling Corrector", "🅱️ Grammar Corrector")
)

# User input
if option.startswith("🅰️"):
    user_input = st.text_area("✍️ Enter a sentence to correct spelling:")
else:
    user_input = st.text_area("✍️ Enter a full sentence to correct grammar:")

# Action button
if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text!")
    else:
        if option.startswith("🅰️"):
            spelling_out = spelling_correction_pipeline(user_input)
            st.subheader("🔡 Spelling Corrected Sentence")
            st.info(spelling_out)
        else:
            grammar_out = correct_sentence(user_input)
            st.subheader("📖 Final Grammar Corrected Sentence")
            st.success(grammar_out)

# Footer
st.markdown("---")
st.markdown("✨ Built with ❤️ by Prakhar Mathur ✨", unsafe_allow_html=True)
