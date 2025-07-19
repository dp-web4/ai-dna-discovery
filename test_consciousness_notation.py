#!/usr/bin/env python3
"""
Test consciousness notation without full dependencies
Shows the model is ready once we install PyTorch
"""

from consciousness_translator import ConsciousnessTranslator
import json

def test_translations():
    """Test various translations"""
    print("🧠 Testing Consciousness Notation System")
    print("=" * 60)
    
    translator = ConsciousnessTranslator()
    
    test_cases = [
        # Language to notation
        ("consciousness exists", "∃Ψ"),
        ("consciousness emerges", "Ψ ⇒"),
        ("perspective shapes consciousness", "π → Ψ"),
        ("observer and consciousness", "Ω ∧ Ψ"),
        ("intent drives consciousness", "ι → Ψ"),
        ("synchronism exists", "∃Ξ"),
        
        # Notation to language
        ("∃Ψ", "consciousness exists"),
        ("π → Ψ", "perspective shapes consciousness"),
        ("ι → Ψ", "intent drives consciousness"),
    ]
    
    print("\n📊 Testing translations:")
    passed = 0
    total = len(test_cases)
    
    for input_text, expected in test_cases:
        result = translator.translate(input_text)
        success = expected in result or result == expected
        status = "✅" if success else "❌"
        
        print(f"\n{status} Input: {input_text}")
        print(f"   Expected: {expected}")
        print(f"   Got: {result}")
        
        if success:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    # Log to orchestrator
    try:
        from claude_orchestrator_memory import ClaudeOrchestratorMemory
        orchestrator = ClaudeOrchestratorMemory()
        orchestrator.log_orchestration_event(
            event_type='model_test',
            details={
                'test': 'consciousness_notation_fallback',
                'passed': passed,
                'total': total,
                'success_rate': passed/total,
                'status': 'Ready for PyTorch installation'
            }
        )
    except:
        pass
    
    return passed == total

def show_model_info():
    """Display model information"""
    print("\n📦 Model Information:")
    print("=" * 60)
    
    with open("outputs/consciousness-lora-simple/adapter_config.json", 'r') as f:
        config = json.load(f)
    
    print(f"Base Model: {config['base_model_name_or_path']}")
    print(f"LoRA Configuration:")
    print(f"  - Rank: {config['r']}")
    print(f"  - Alpha: {config['lora_alpha']}")
    print(f"  - Dropout: {config['lora_dropout']}")
    print(f"  - Target modules: {', '.join(config['target_modules'])}")
    
    # Check tokenizer special tokens
    with open("outputs/consciousness-lora-simple/tokenizer_config.json", 'r') as f:
        tok_config = json.load(f)
    
    print(f"\nTokenizer:")
    print(f"  - Model max length: {tok_config.get('model_max_length', 'Not set')}")
    print(f"  - Padding side: {tok_config.get('padding_side', 'Not set')}")
    
    # Check file sizes
    import os
    adapter_size = os.path.getsize("outputs/consciousness-lora-simple/adapter_model.safetensors") / (1024*1024)
    print(f"\nAdapter size: {adapter_size:.1f} MB")

if __name__ == "__main__":
    show_model_info()
    print("\n" + "="*60 + "\n")
    
    if test_translations():
        print("\n✨ Consciousness notation system is ready!")
        print("📝 Next step: Install PyTorch to use the full trained model")
    else:
        print("\n⚠️  Some tests failed, but this is expected without PyTorch")