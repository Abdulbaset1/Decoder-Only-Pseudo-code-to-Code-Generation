import streamlit as st
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

st.set_page_config(page_title="GPT2 Pseudocode Generator", layout="centered")

@st.cache_resource
def load_model():
    model_name = "basit1878/gpt2-pseudocode-finetuned"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("🔧 GPT-2 Pseudocode Generator")
st.write("Enter a programming instruction and the model will generate Python-like pseudocode.")

user_input = st.text_area("Enter your instruction:", height=180)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer.encode(user_input, return_tensors="pt")

        # --------------------------
        # Generation with stop token
        # --------------------------
        outputs = model.generate(
            inputs,
            max_length=150,
            temperature=0.3,          # lower randomness
            top_p=0.85,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            early_stopping=True        # stop at first EOS
        )

        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --------------------------
        # Clean and extract pseudocode
        # --------------------------
        if "Python Code:" in result:
            result = result.split("Python Code:")[-1].strip()

        # Stop at first blank line or excessive garbage
        result = "\n".join([line for line in result.split("\n") if line.strip() != ""])
        result_lines = []
        for line in result.split("\n"):
            if any(c in line for c in ["{", "}", ";", "#include", "cin", "cout"]):
                break  # stop at non-Python code
            result_lines.append(line)
        result = "\n".join(result_lines)

        st.subheader("Generated Pseudocode:")
        st.code(result if result else "Unable to generate clean pseudocode. Try simpler input.", language="python")
