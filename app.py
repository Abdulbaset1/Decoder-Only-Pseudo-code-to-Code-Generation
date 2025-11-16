import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="GPT2 Pseudocode Generator", layout="centered")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model_name = "basit1878/gpt2-pseudocode-finetuned"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------
# App title & instructions
# -------------------------
st.title("🔧 GPT-2 Pseudocode Generator")
st.write("Enter a programming instruction and the model will generate Python-like pseudocode.")

# -------------------------
# User input
# -------------------------
user_input = st.text_area("Enter your instruction:", height=180)

# -------------------------
# Generate button
# -------------------------
if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Tokenize input
        inputs = tokenizer.encode(user_input, return_tensors="pt")

        # Generate output with safer parameters
        outputs = model.generate(
            inputs,
            max_length=150,            # limit output length
            temperature=0.5,           # lower randomness
            top_p=0.9,                 # nucleus sampling
            top_k=50,                  # avoid unlikely tokens
            do_sample=True,
            repetition_penalty=1.5,    # reduce repetitions
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )

        # Decode safely
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Keep only code after "Python Code:" if present
        if "Python Code:" in result:
            result = result.split("Python Code:")[-1].strip()
        
        # Truncate at first double newline to avoid long garbage
        result = result.split("\n\n")[0]
        
        # Remove empty/garbled lines
        result = "\n".join([line for line in result.split("\n") if line.strip() != ""])

        # Display output
        st.subheader("Generated Pseudocode:")
        st.code(result, language="python")
