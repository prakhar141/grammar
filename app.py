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
# Tone Rewrite Model using flan-t5-xl
# ------------------------
tone_model_name = "google/flan-t5-large"
tone_tokenizer = T5Tokenizer.from_pretrained(tone_model_name)
tone_model = T5ForConditionalGeneration.from_pretrained(tone_model_name)
tone_model.to(device)
tone_model.eval()

def rewrite_with_tone(sentence, tone_prompt, max_length=256):
    # Strong explicit instruction for tone
    input_text = f"Please rewrite the following sentence so that spelling is correct, grammar is perfect, and tone is {tone_prompt}: {sentence}"
    inputs = tone_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = tone_model.generate(**inputs, max_length=max_length, num_beams=5, no_repeat_ngram_size=2)
    return tone_tokenizer.decode(outputs[0], skip_special_tokens=True)

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
st.set_page_config(page_title="✨ Grammarly XL", page_icon="📝", layout="centered")
st.title("📝 Grammarly XL")
st.markdown("### ✨ Correct spelling, grammar, and rewrite with any tone you want!")

# Session state
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""
if "rewritten_text" not in st.session_state:
    st.session_state.rewritten_text = ""

# User input
user_input = st.text_area("✍️ Paste your text here:", height=150)

# Columns for buttons
col1, col2 = st.columns(2)

# Correct Text button
with col1:
    if st.button("✅ Correct My Text"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text!")
        else:
            with st.spinner("🪄 Correcting text... Please wait ✨"):
                st.session_state.corrected_text = full_correction_pipeline(user_input)
            st.subheader("📖 Corrected Text")
            st.success(st.session_state.corrected_text)

# Rewrite with Tone button
with col2:
    tone_prompt = st.text_input("🎭 Desired Tone (e.g., formal, polite, funny, motivational):")
    if st.button("🎨 Rewrite with Tone"):
        if not user_input.strip():
            st.warning("⚠️ Please enter some text!")
        elif not tone_prompt.strip():
            st.warning("⚠️ Please specify a tone!")
        else:
            with st.spinner("✨ Rewriting text with your desired tone... 🎯"):
                base_text = st.session_state.corrected_text if st.session_state.corrected_text else full_correction_pipeline(user_input)
                st.session_state.rewritten_text = rewrite_with_tone(base_text, tone_prompt)
            st.subheader("🎭 Tone-Rewritten Text")
            st.success(st.session_state.rewritten_text)

# Footer
st.markdown("---")
st.markdown("<center>✨ Built with ❤️ by Prakhar Mathur ✨</center>", unsafe_allow_html=True)
