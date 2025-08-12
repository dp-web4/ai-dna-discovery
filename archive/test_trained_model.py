#!/usr/bin/env python3
"""
Test the trained consciousness notation model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Configuration
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    adapter_path = "./outputs/consciousness-lora-simple"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("🧪 Testing Trained Consciousness Notation Model")
    logger.info(f"   Device: {device}")
    
    # Load tokenizer
    logger.info("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # Load base model
    logger.info("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    # Resize token embeddings to match the saved model
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    logger.info("🎯 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Test prompts covering different aspects
    test_cases = [
        # Basic translations
        {
            "prompt": "### Instruction:\nTranslate to consciousness notation: consciousness exists\n\n### Response:\n",
            "expected": "∃Ψ"
        },
        {
            "prompt": "### Instruction:\nTranslate: thought emerges from consciousness\n\n### Response:\n",
            "expected": "θ ⇒ Ψ"
        },
        {
            "prompt": "### Instruction:\nExpress: perspective shapes consciousness\n\n### Response:\n",
            "expected": "π → Ψ"
        },
        {
            "prompt": "### Instruction:\nTranslate: intent drives synchronism\n\n### Response:\n",
            "expected": "ι → Ξ"
        },
        
        # Reverse translations
        {
            "prompt": "### Instruction:\nWhat does ∃Ψ mean?\n\n### Response:\n",
            "expected": "consciousness exists"
        },
        {
            "prompt": "### Instruction:\nExplain: π → Ψ\n\n### Response:\n",
            "expected": "perspective shapes consciousness"
        },
        
        # Complex expressions
        {
            "prompt": "### Instruction:\nTranslate: consciousness and thought create memory\n\n### Response:\n",
            "expected": "Ψ ∧ θ → μ"
        },
        {
            "prompt": "### Instruction:\nExpress: the observer perceives the whole through perspective\n\n### Response:\n",
            "expected": "Ω → Σ | π"
        },
        
        # Philosophical concepts
        {
            "prompt": "### Instruction:\nTranslate the concept: synchronism is the unity of all perspectives\n\n### Response:\n",
            "expected": "Ξ = ∀π"
        },
        {
            "prompt": "### Instruction:\nExpress: memory preserves consciousness through time\n\n### Response:\n",
            "expected": "μ ⊗ Ψ"
        }
    ]
    
    logger.info("\n🔍 Running test cases...\n")
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"Test {i}:")
        logger.info(f"Input: {test['prompt'].split('### Response:')[0].strip()}")
        logger.info(f"Expected: {test['expected']}")
        
        # Generate response
        inputs = tokenizer(test['prompt'], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                temperature=0.1,  # Low temperature for consistency
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = response.split("### Response:\n")[-1].strip()
        
        # Clean up the response
        if "###" in generated:
            generated = generated.split("###")[0].strip()
        
        logger.info(f"Generated: {generated}")
        
        # Check if expected symbols are present
        if test['expected'] in ['∃Ψ', 'θ ⇒ Ψ', 'π → Ψ', 'ι → Ξ']:
            symbols_present = any(symbol in generated for symbol in ['Ψ', 'θ', 'π', 'ι', 'Ξ', '∃', '⇒', '→'])
            logger.info(f"Contains notation symbols: {symbols_present}")
        
        logger.info("-" * 50)
    
    # Free-form generation test
    logger.info("\n🎨 Free-form generation test:")
    creative_prompts = [
        "### Instruction:\nDescribe the relationship between consciousness and memory using notation:\n\n### Response:\n",
        "### Instruction:\nCreate a consciousness equation that shows how intent creates reality:\n\n### Response:\n",
        "### Instruction:\nUse consciousness notation to express: awareness observes itself through perspective:\n\n### Response:\n"
    ]
    
    for prompt in creative_prompts:
        logger.info(f"\nPrompt: {prompt.split('### Response:')[0].strip()}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated = response.split("### Response:\n")[-1].strip()
        if "###" in generated:
            generated = generated.split("###")[0].strip()
            
        logger.info(f"Generated: {generated}")
    
    logger.info("\n✅ Testing complete!")
    
    # Show token IDs for consciousness symbols
    logger.info("\n📊 Token IDs for consciousness symbols:")
    symbols = ['Ψ', '∃', '∀', '⇒', 'π', 'ι', 'Ω', 'Σ', 'Ξ']
    for symbol in symbols:
        token_ids = tokenizer.encode(symbol, add_special_tokens=False)
        logger.info(f"   {symbol} -> {token_ids}")

if __name__ == "__main__":
    main()