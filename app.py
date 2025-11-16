import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

st.set_page_config(page_title="GPT2 Pseudocode Generator", layout="centered")

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    model_name = "basit1878/gpt2-pseudocode-finetuned"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()


# ---------------------------------------------------------
# Better Generation Function (fixed repetition + clean output)
# ---------------------------------------------------------

def generate_code(pseudocode, max_length=256):

    prompt = (
        "[PSEUDOCODE]\n"
        f"{pseudocode.strip()}\n\n"
        "[PYTHON]\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only Python section
    if "[PYTHON]" in text:
        code = text.split("[PYTHON]", 1)[1].strip()
    else:
        code = text.strip()

    # Remove accidental continuation
    for stop_word in ["[PSEUDOCODE]", "<|endoftext|>"]:
        if stop_word in code:
            code = code.split(stop_word)[0].strip()

    return code


# ---------------------------------------------------------
# Streamlit UI (same as your previous one)
# ---------------------------------------------------------

st.title("🔧 GPT-2 Pseudocode → Python Generator")
st.write("Enter pseudocode or instructions. The model will generate Python code cleanly.")

user_input = st.text_area("Enter your pseudocode:", height=180)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating..."):
            result = generate_code(user_input)

        st.subheader("Generated Python Code:")
        st.code(result, language="python")
