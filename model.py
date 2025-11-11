import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import evaluate
import numpy as np
from tqdm import tqdm
import os
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model_if_needed():
    """Download the model from GitHub releases if not present"""
    model_url = "https://github.com/Abdulbaset1/Decoder-Only-Pseudo-code-to-Code-Generation/releases/tag/v1/gpt2_finetuned.pt"
    model_path = "gpt2_finetuned.pt"
    
    if not os.path.exists(model_path):
        logger.info("Model file not found locally. Downloading from GitHub...")
        try:
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc="Downloading model",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info("Model downloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    return True

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

class PseudoCodeDataset(Dataset):
    """Custom Dataset for Pseudocode to Python Code pairs"""
    
    def __init__(self, tsv_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load TSV file
        df = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

        # Group by probid and subid to reconstruct full code blocks
        self.data = self._preprocess_data(df)
        
    def _preprocess_data(self, df):
        """Reconstruct complete code blocks from line-by-line data"""
        processed_data = []
        
        # Group by problem ID and submission ID
        grouped = df.groupby(['probid', 'subid'])
        
        for (probid, subid), group in grouped:
            # Sort by line number to maintain order
            group = group.sort_values('line')
            
            # Extract pseudocode (text column)
            pseudocode_lines = group['text'].tolist()
            pseudocode = '\n'.join([str(line) for line in pseudocode_lines if pd.notna(line)])
            
            # Extract actual code (code column)
            code_lines = group['code'].tolist()
            code = '\n'.join([str(line) for line in code_lines if pd.notna(line)])
            
            if pseudocode.strip() and code.strip():
                processed_data.append({
                    'pseudocode': pseudocode,
                    'code': code,
                    'probid': probid,
                    'subid': subid
                })
        
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: "Pseudocode:\n{pseudocode}\n\nPython Code:\n{code}<|endoftext|>"
        prompt = f"Pseudocode:\n{item['pseudocode']}\n\nPython Code:\n{item['code']}<|endoftext|>"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

# ============================================================================
# 2. MODEL TRAINING
# ============================================================================

def setup_model_and_tokenizer(model_name='gpt2'):
    """Initialize GPT-2 model and tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def train_model(train_dataset, eval_dataset, model, tokenizer, output_dir='./gpt2-pseudocode-finetuned'):
    """Fine-tune GPT-2 on pseudocode to code task"""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        prediction_loss_only=True
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model in Hugging Face format
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model as .pt file for deployment
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_config': tokenizer.init_kwargs,
        'model_config': model.config
    }, os.path.join(output_dir, 'gpt2_finetuned.pt'))
    
    logger.info(f"Model saved as .pt file: {os.path.join(output_dir, 'gpt2_finetuned.pt')}")
    
    return trainer

# ============================================================================
# 3. MODEL INFERENCE (For Deployment)
# ============================================================================

class PseudoCodeGenerator:
    """Class for generating Python code from pseudocode"""
    
    def __init__(self, model_path='gpt2_finetuned.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer = self._load_model(model_path)
        self.model.to(self.device)
        
    def _load_model(self, model_path):
        """Load the fine-tuned model and tokenizer"""
        # First, download model if needed
        if not os.path.exists(model_path):
            logger.info("Model not found locally. Downloading...")
            if not download_model_if_needed():
                raise FileNotFoundError(f"Could not download model from GitHub releases")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model architecture
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Load fine-tuned weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded fine-tuned model from {model_path}")
        
        return model, tokenizer
    
    def generate_code(self, pseudocode, max_length=512, temperature=0.7, top_p=0.95, num_return_sequences=1):
        """Generate Python code from pseudocode"""
        
        # Format prompt
        prompt = f"Pseudocode:\n{pseudocode}\n\nPython Code:\n"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate code
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs['attention_mask'],
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode generated sequences
        generated_codes = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract only the generated code part
            if "Python Code:" in generated_text:
                code = generated_text.split("Python Code:")[-1].strip()
            else:
                code = generated_text
            
            generated_codes.append(code)
        
        return generated_codes[0] if num_return_sequences == 1 else generated_codes

# ============================================================================
# 4. EVALUATION METRICS
# ============================================================================

def calculate_bleu(predictions, references):
    """Calculate BLEU score"""
    try:
        bleu = evaluate.load("bleu")
        
        # Tokenize predictions and references
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        result = bleu.compute(predictions=pred_tokens, references=ref_tokens)
        return result
    except Exception as e:
        logger.error(f"Error calculating BLEU: {e}")
        return None

def calculate_exact_match(predictions, references):
    """Calculate exact match accuracy"""
    matches = sum([1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip()])
    return matches / len(predictions) if predictions else 0

def calculate_syntax_validity(code_strings):
    """Check syntactic validity of generated Python code"""
    valid_count = 0
    for code in code_strings:
        try:
            compile(code, '<string>', 'exec')
            valid_count += 1
        except:
            pass
    return valid_count / len(code_strings) if code_strings else 0

def evaluate_model(model, tokenizer, test_dataset, device='cuda', num_samples=100):
    """Comprehensive evaluation of the fine-tuned model"""
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    pseudocodes = []
    
    # Sample test data
    test_samples = test_dataset.data[:num_samples]
    
    logger.info(f"Generating predictions for {len(test_samples)} samples...")
    for item in tqdm(test_samples):
        pseudocode = item['pseudocode']
        reference_code = item['code']
        
        # Generate code using the generator class
        generator = PseudoCodeGenerator()
        predicted_code = generator.generate_code(pseudocode, device=device)
        
        predictions.append(predicted_code)
        references.append(reference_code)
        pseudocodes.append(pseudocode)
    
    # Calculate metrics
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    # BLEU Score
    bleu_result = calculate_bleu(predictions, references)
    if bleu_result:
        logger.info(f"BLEU Score: {bleu_result['bleu']:.4f}")
    
    # Exact Match
    exact_match = calculate_exact_match(predictions, references)
    logger.info(f"Exact Match Accuracy: {exact_match:.4f}")
    
    # Syntax Validity
    syntax_valid = calculate_syntax_validity(predictions)
    logger.info(f"Syntax Validity: {syntax_valid:.4f}")
    
    return {
        'bleu': bleu_result,
        'exact_match': exact_match,
        'syntax_validity': syntax_valid,
        'predictions': predictions,
        'references': references
    }

# ============================================================================
# 5. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    # Paths
    TRAIN_PATH = '/kaggle/input/pythoncodgnration/python-spoc-train.tsv'
    EVAL_PATH = '/kaggle/input/pythoncodgnration/python-spoc-train-eval.tsv'
    TEST_PATH = '/kaggle/input/pythoncodgnration/python-spoc-train-test.tsv'
    OUTPUT_DIR = './gpt2-pseudocode-finetuned'
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Step 1: Setup model and tokenizer
    logger.info("\n1. Loading model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer('gpt2')
    model.to(device)
    
    # Step 2: Prepare datasets
    logger.info("\n2. Preparing datasets...")
    train_dataset = PseudoCodeDataset(TRAIN_PATH, tokenizer, max_length=512)
    eval_dataset = PseudoCodeDataset(EVAL_PATH, tokenizer, max_length=512)
    test_dataset = PseudoCodeDataset(TEST_PATH, tokenizer, max_length=512)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Step 3: Fine-tune model
    logger.info("\n3. Fine-tuning model...")
    trainer = train_model(train_dataset, eval_dataset, model, tokenizer, OUTPUT_DIR)
    
    # Step 4: Evaluate model
    logger.info("\n4. Evaluating model on test set...")
    results = evaluate_model(trainer.model, tokenizer, test_dataset, device=device, num_samples=100)
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING AND EVALUATION COMPLETE!")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("="*50)
    
    return trainer, results

if __name__ == "__main__":
    # Run the complete pipeline
    trainer, results = main()
    
    # Optional: Save evaluation results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'bleu': results['bleu'],
            'exact_match': results['exact_match'],
            'syntax_validity': results['syntax_validity']
        }, f, indent=2)
    logger.info("Evaluation results saved to evaluation_results.json")
