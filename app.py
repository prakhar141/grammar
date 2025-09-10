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
q_learning_url = "https://huggingface.co/prakhar146/grammar/resolve/main/q_learning_spelling_model.pkl"
q_learning_file = download_file(q_learning_url, "q_learning_spelling_model.pkl")

with open(q_learning_file, "rb") as f:
    model_data = pickle.load(f)

Q = model_data['Q_table']
all_words = model_data['all_words']

def word_to_state(word):
    ngrams = [word[i:i+2] for i in range(len(word)-1)]
    return tuple(sorted(ngrams))

def predict_word(state_word, Q, all_words):
    if state_word is None or len(state_word) == 0:
        return state_word
    state = word_to_state(state_word)
    if state in Q and Q[state]:
        return max(Q[state], key=Q[state].get)
    candidates = get_close_matches(state_word, all_words, n=1)
    return candidates[0] if candidates else state_word
# ------------------------
# T5 Grammar Model
# ------------------------
# ⚠️ Replace this with the actual model repo where your T5 model is uploaded
t5_repo = "prakhar146/grammar"

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(t5_repo, use_fast=True)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_repo)

# Always keep on CPU to avoid "meta tensor" errors
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
# Streamlit UI
# ------------------------
# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="📝 Spelling & Grammar Corrector", page_icon="✨", layout="centered")

st.title("📝 Spelling & Grammar Corrector")
st.markdown("### 🔍 Enter your text and get the final polished sentence instantly!")

# Helper to remove duplicate consecutive words
def remove_duplicate_words(text):
    words = text.split()
    cleaned = []
    for w in words:
        if not cleaned or cleaned[-1].lower() != w.lower():
            cleaned.append(w)
    return " ".join(cleaned)

# User input
user_input = st.text_area("✍️ Enter your text here:")

# Action button
if st.button("✨ Correct My Text"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to correct!")
    else:
        # Step 1: Spelling correction (Q-learning)
        words = user_input.split()
        spelling_corrected = " ".join([predict_word(w, Q, all_words) for w in words])

        # Step 2: Remove duplicates
        cleaned_text = remove_duplicate_words(spelling_corrected)

        # Step 3: Grammar correction (T5)
        fully_corrected = correct_sentence(cleaned_text)

        # Show only final corrected text
        st.subheader("📖 Final Corrected Sentence")
        st.success(fully_corrected)

# Footer
st.markdown("---")
st.caption("✨ Built with ❤️ by Prakhar Mathur")
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>✨ Built with ❤️ by Prakhar Mathur ✨</div>",
    unsafe_allow_html=True
)

