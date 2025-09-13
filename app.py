import streamlit as st
import json
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
q_learning_url = "https://huggingface.co/prakhar146/grammar/resolve/main/q_table.json"
q_learning_file = download_file(q_learning_url, "q_table.json")

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

# ------------------------
# Helpers
# ------------------------
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
cache_dir = "./.hf_cache"

grammar_tokenizer = T5Tokenizer.from_pretrained(
    t5_repo,
    use_fast=True,
    cache_dir=cache_dir
)

grammar_model = T5ForConditionalGeneration.from_pretrained(
    t5_repo,
    cache_dir=cache_dir,
    device_map=None
)

device = torch.device("cpu")
grammar_model.to(device)
grammar_model.eval()

def correct_sentence(sentence, max_length=128):
    input_text = "grammar: " + sentence
    inputs = grammar_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = grammar_model.generate(**inputs, max_length=max_length)
    return grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# Professional Tone Model (Hugging Face files)
# ------------------------
tone_dir = "tone_model"
os.makedirs(tone_dir, exist_ok=True)

tone_files = {
    "model.safetensors": "https://huggingface.co/prakhar146/grammar/resolve/main/model%20(1).safetensors",
    "spiece.model": "https://huggingface.co/prakhar146/grammar/resolve/main/spiece%20(1).model",
    "added_tokens.json": "https://huggingface.co/prakhar146/grammar/resolve/main/added_tokens%20(1).json",
    "tokenizer_config.json": "https://huggingface.co/prakhar146/grammar/resolve/main/tokenizer_config%20(1).json",
    "special_tokens_map.json": "https://huggingface.co/prakhar146/grammar/resolve/main/special_tokens_map%20(1).json",
    "config.json": "https://huggingface.co/prakhar146/grammar/resolve/main/config%20(1).json",
    "generation_config.json": "https://huggingface.co/prakhar146/grammar/resolve/main/generation_config%20(1).json"
}

# Download model files
for name, url in tone_files.items():
    path = os.path.join(tone_dir, name)
    download_file(url, path)

# Load Professional-tone model directly on CPU
tone_tokenizer = T5Tokenizer.from_pretrained(tone_dir)
tone_model = T5ForConditionalGeneration.from_pretrained(tone_dir, device_map=None)
tone_model.to(device)
tone_model.eval()

def to_professional(sentence, max_length=128):
    input_text = f"Professional: {sentence}"
    inputs = tone_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = tone_model.generate(**inputs, max_length=max_length)
    return tone_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------------
# Full Pipeline: Spelling → Grammar → Professional Tone
# ------------------------
def full_pipeline(sentence):
    words = sentence.split()
    spelling_corrected = " ".join([apply_custom_corrections(predict_word(w, Q, all_words)) for w in words])
    cleaned = remove_duplicate_words(spelling_corrected)
    grammar_corrected = correct_sentence(cleaned)
    professional_sentence = to_professional(grammar_corrected)
    return professional_sentence

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="✨GrammarlyPro", page_icon="📝", layout="centered")
st.title("📝 Grammar, Spelling & Professional Tone Corrector")
st.markdown("### ✨ Correct your text and rewrite it in Professional tone!")

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
