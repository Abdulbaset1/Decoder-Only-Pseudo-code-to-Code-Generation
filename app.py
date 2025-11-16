import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="GPT2 Pseudocode Generator", layout="centered")

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "basit1878/gpt2-pseudocode-finetuned"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# App Title and Description
# -----------------------------
st.title("🔧 GPT-2 Pseudocode Generator")
st.write("Enter any programming problem and the model will generate Python-like pseudocode.")

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter your instruction:", height=180)

# -----------------------------
# Generate Button
# -----------------------------
if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Encode input
        inputs = tokenizer.encode(user_input, return_tensors="pt")

        # Generate output safely
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,       # Stop at EOS token
            repetition_penalty=1.2,                    # Reduce repeated phrases
            num_return_sequences=1
        )

        # Decode and clean output
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Keep only the generated code after "Python Code:"
        if "Python Code:" in result:
            result = result.split("Python Code:")[-1].strip()

        # Optional: truncate at first double newline to avoid repetition
        result = result.split("\n\n")[0]

        # Display output
        st.subheader("Generated Pseudocode:")
        st.code(result, language="python")
