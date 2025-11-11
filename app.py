import streamlit as st
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import requests
import os
import time

# Set page configuration
st.set_page_config(
    page_title="PseudoCode to Python Code Generator",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff6b6b;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .generated-code {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        font-family: 'Courier New', monospace;
        margin: 1rem 0;
    }
    .pseudocode-input {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer"""
    try:
        # Check if model file exists
        model_path = "http://github.com/Abdulbaset1/Decoder-Only-Pseudo-code-to-Code-Generation/releases/tag/v1"
        
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found.")
            st.info("Please make sure the model file is in the same directory as this script.")
            return None, None
        
        # Initialize tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Initialize model
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Load fine-tuned weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        st.success("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_code(model, tokenizer, pseudocode, max_length=512, temperature=0.7, top_p=0.95):
    """Generate Python code from pseudocode"""
    try:
        # Format the prompt
        prompt = f"Pseudocode:\n{pseudocode}\n\nPython Code:\n"
        
        # Tokenize input
        inputs = tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # Generate code
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs['attention_mask'],
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated code part
        if "Python Code:" in generated_text:
            code = generated_text.split("Python Code:")[-1].strip()
        else:
            code = generated_text
        
        return code
        
    except Exception as e:
        st.error(f"Error during code generation: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🧠 PseudoCode to Python Code Generator</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model... This may take a few moments."):
        model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check if 'gpt2_finetuned.pt' is in the correct location.")
        return
    
    # Sidebar
    st.sidebar.markdown('<div class="sub-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    max_length = st.sidebar.slider("Max Length", min_value=100, max_value=1024, value=512, step=50, 
                                  help="Maximum length of generated text")
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.7, step=0.1,
                                   help="Higher values make output more random, lower values more deterministic")
    top_p = st.sidebar.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.05,
                             help="Nucleus sampling: considers only the top p probability tokens")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="sub-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.sidebar.info(
        "This AI-powered tool converts pseudocode to executable Python code using a fine-tuned GPT-2 model. "
        "Simply enter your algorithm in plain English pseudocode and let the AI generate the corresponding Python code!"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="sub-header">📝 Input Pseudocode</div>', unsafe_allow_html=True)
        
        # Example selection
        example_option = st.selectbox(
            "Choose an example or write your own:",
            ["Custom Input", "Sum of List", "Find Maximum", "Factorial", "Fibonacci", "Palindrome Check"]
        )
        
        # Predefined examples
        examples = {
            "Sum of List": "Initialize sum to 0\nFor each number in the list\n    Add number to sum\nPrint the sum",
            "Find Maximum": "Set max to first element\nFor each number in the list\n    If number > max\n        Set max to number\nPrint max",
            "Factorial": "Read number n\nSet result to 1\nFor i from 1 to n\n    Multiply result by i\nPrint result",
            "Fibonacci": "Read number n\nIf n <= 0\n    Print 0\nElse if n == 1\n    Print 1\nElse\n    a = 0\n    b = 1\n    For i from 2 to n\n        c = a + b\n        a = b\n        b = c\n    Print b",
            "Palindrome Check": "Read string s\nSet left to 0\nSet right to length of s - 1\nWhile left < right\n    If s[left] != s[right]\n        Print 'Not palindrome'\n        Exit\n    left = left + 1\n    right = right - 1\nPrint 'Palindrome'",
            "Custom Input": ""
        }
        
        # Text area for pseudocode input
        default_text = examples[example_option] if example_option != "Custom Input" else ""
        pseudocode = st.text_area(
            "Enter your pseudocode:",
            value=default_text,
            height=250,
            placeholder="Enter your algorithm in pseudocode here...\nExample:\nInitialize sum to 0\nFor each number in list\n    Add number to sum\nPrint sum",
            help="Write your algorithm step by step in plain English"
        )
        
        # Generate button
        generate_btn = st.button(
            "🚀 Generate Python Code",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        st.markdown('<div class="sub-header">🐍 Generated Python Code</div>', unsafe_allow_html=True)
        
        if generate_btn:
            if pseudocode.strip():
                with st.spinner("🔄 Generating Python code... This may take a few seconds."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    generated_code = generate_code(
                        model, 
                        tokenizer, 
                        pseudocode, 
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                if generated_code:
                    st.markdown("### ✅ Generated Code:")
                    st.markdown('<div class="generated-code">', unsafe_allow_html=True)
                    st.code(generated_code, language='python')
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Code metrics
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    with col_metrics1:
                        st.metric("Lines of Code", len(generated_code.split('\n')))
                    with col_metrics2:
                        st.metric("Characters", len(generated_code))
                    with col_metrics3:
                        st.metric("Words", len(generated_code.split()))
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Generated Code",
                        data=generated_code,
                        file_name="generated_code.py",
                        mime="text/x-python",
                        use_container_width=True
                    )
                    
                    # Tips
                    st.markdown("---")
                    with st.expander("💡 Tips for better results"):
                        st.write("""
                        - **Be specific** in your pseudocode
                        - **Use clear variable names** in your description
                        - **Break down complex algorithms** into smaller steps
                        - **Specify data types** when important
                        - **Include edge cases** in your description
                        """)
            else:
                st.warning("⚠️ Please enter some pseudocode to generate Python code.")
        
        else:
            # Placeholder when no generation has been done
            st.markdown("""
            <div class="info-box">
            <h4>👆 Ready to generate code!</h4>
            <p>Enter your pseudocode on the left and click the <strong>Generate Python Code</strong> button.</p>
            <p>You can also select from the example pseudocodes to see how it works.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick examples
            st.markdown("### 🎯 Quick Examples:")
            quick_col1, quick_col2 = st.columns(2)
            
            with quick_col1:
                if st.button("Sum Example", use_container_width=True):
                    st.session_state.example = examples["Sum of List"]
                    st.rerun()
                
                if st.button("Factorial Example", use_container_width=True):
                    st.session_state.example = examples["Factorial"]
                    st.rerun()
            
            with quick_col2:
                if st.button("Max Example", use_container_width=True):
                    st.session_state.example = examples["Find Maximum"]
                    st.rerun()
                
                if st.button("Fibonacci Example", use_container_width=True):
                    st.session_state.example = examples["Fibonacci"]
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with ❤️ using Streamlit and Fine-tuned GPT-2 | "
        "PseudoCode to Python Code Generation"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
