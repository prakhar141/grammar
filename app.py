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
        Q = model_data
        all_words = sorted({a for actions in Q.values() for a in actions.keys()})
else:
    Q = model_data
    all_words = sorted({a for actions in Q.values() for a in actions.keys()})

def word_to_state(word: str):
    if not word:
        return ()
    ngrams = [word[i:i+2] for i in range(len(word)-1)]
    return tuple(sorted(ngrams))

def predict_word(state_word, Q, all_words, min_similarity: float = 0.8) -> str:
    if not state_word:
        return state_word
    state = word_to_state(state_word)
    if state in Q and Q[state]:
        return max(Q[state], key=Q[state].get)
    candidates = get_close_matches(state_word, all_words, n=1, cutoff=min_similarity)
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
# Full Correction Pipeline
# ------------------------
def full_correction_pipeline(sentence):
    words = sentence.split()
    spelling_corrected = " ".join([apply_custom_corrections(predict_word(w, Q, all_words)) for w in words])
    spelling_cleaned = remove_duplicate_words(spelling_corrected)
    grammar_corrected = correct_sentence(spelling_cleaned)
    return grammar_corrected

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="✨ Grammar & Spelling Corrector", page_icon="📝", layout="centered")
st.title("📝 Grammar & Spelling Corrector")
st.markdown("### ✨ Correct spelling and grammar instantly!")

# Session state
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""

# User input
user_input = st.text_area("✍️ Paste your text here:", height=150)

# Correct Text button
if st.button("✅ Correct My Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text!")
    else:
        with st.spinner("🪄 Correcting text... Please wait ✨"):
            st.session_state.corrected_text = full_correction_pipeline(user_input)
        st.subheader("📖 Corrected Text")
        st.success(st.session_state.corrected_text)

# Footer
st.markdown("---")
st.markdown("<center>✨ Built with ❤️ by Prakhar Mathur ✨</center>", unsafe_allow_html=True)
