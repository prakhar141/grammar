import streamlit as st
import json
from difflib import get_close_matches
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os

# ------------------------
# SPELLING + GRAMMAR MODEL (UNCHANGED)
# ------------------------
# Load Q-learning spelling correction
q_learning_file = "q_table.json"
with open(q_learning_file, "r") as f:
    model_data = json.load(f)

if isinstance(model_data, dict):
    if "q_table" in model_data:
        Q = model_data["q_table"]
        all_words = model_data.get("all_words", [])
    else:
        Q = model_data
        all_words = sorted({a for actions in Q.values() for a in actions.keys()})
else:
    Q = model_data
    all_words = sorted({a for actions in Q.values() for a in actions.keys()})

custom_corrections = {
    "teh": "the", "Teh": "The", "recieve": "receive",
    "adress": "address", "enviroment": "environment",
    "acommodate": "accommodate"
}

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

def apply_custom_corrections(word):
    return custom_corrections.get(word, word)

def remove_duplicate_words(text):
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1].lower() != w.lower():
            cleaned.append(w)
    return " ".join(cleaned)

# Load Grammar T5 Model
grammar_tokenizer = T5Tokenizer.from_pretrained(".", use_fast=True)  # same dir
grammar_model = T5ForConditionalGeneration.from_pretrained(".")
grammar_device = torch.device("cpu")
grammar_model.to(grammar_device)
grammar_model.eval()

def correct_sentence(sentence, max_length=128):
    input_text = "grammar: " + sentence
    inputs = grammar_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(grammar_device)
    with torch.no_grad():
        outputs = grammar_model.generate(**inputs, max_length=max_length)
    return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)

def full_grammar_pipeline(sentence):
    words = sentence.split()
    spelling_corrected = " ".join([apply_custom_corrections(predict_word(w, Q, all_words)) for w in words])
    spelling_cleaned = remove_duplicate_words(spelling_corrected)
    grammar_corrected = correct_sentence(spelling_cleaned)
    return grammar_corrected

# ------------------------
# PROFESSIONAL TONE MODEL (SAME DIRECTORY)
# ------------------------
# Load same-directory files for tone model
tone_tokenizer = T5Tokenizer.from_pretrained(".")
tone_model = T5ForConditionalGeneration.from_pretrained(".")
tone_device = torch.device("cpu")
tone_model.to(tone_device)
tone_model.eval()

def to_professional(sentence, max_length=128):
    input_text = f"Professional: {sentence}"
    inputs = tone_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(tone_device)
    with torch.no_grad():
        outputs = tone_model.generate(**inputs, max_length=max_length)
    return tone_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# FULL PIPELINE (SPELLING + GRAMMAR + PROFESSIONAL TONE)
# ------------------------
def full_pipeline(sentence):
    grammar_corrected = full_grammar_pipeline(sentence)
    professional_sentence = to_professional(grammar_corrected)
    return professional_sentence

# ------------------------
# STREAMLIT UI
# ------------------------
st.set_page_config(page_title="✨ Grammar & Professional Tone Corrector", page_icon="📝", layout="centered")
st.title("📝 Grammar & Professional Tone Corrector")
st.markdown("### ✨ Correct grammar and rewrite text in Professional tone!")

if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""

user_input = st.text_area("✍️ Paste your text here:", height=150)

if st.button("✅ Correct My Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text!")
    else:
        with st.spinner("🪄 Correcting text and applying Professional tone..."):
            st.session_state.corrected_text = full_pipeline(user_input)
        st.subheader("📖 Corrected & Professional Text")
        st.success(st.session_state.corrected_text)

st.markdown("---")
st.markdown("<center>✨ Built with ❤️ by Prakhar Mathur ✨</center>", unsafe_allow_html=True)
