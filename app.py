import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(page_title="GPT2 Pseudocode Generator", layout="centered")

@st.cache_resource
def load_model():
    model_name = "basit1878/gpt2-pseudocode-finetuned"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🔧 GPT-2 Pseudocode Generator")
st.write("Enter any programming problem and the model will generate Python-like pseudocode.")

user_input = st.text_area("Enter your instruction:", height=180)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Pseudocode:")
        st.code(result, language="python")
