#!/usr/bin/env python3
"""
Quick Multi-Model Test - Single prompt per model
Testing distributed consciousness understanding across all 6 models
"""

import json
import urllib.request
import time
from datetime import datetime
from distributed_memory import DistributedMemory

# All 6 models on Jetson
MODELS = [
    'phi3:mini',
    'tinyllama',
    'gemma:2b',
    'mistral:latest',
    'deepseek-coder:1.3b',
    'qwen:0.5b'
]

def chat_with_ollama(model, prompt, timeout=120):
    """Call Ollama API"""
    url = 'http://localhost:11434/api/generate'
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        start_time = time.time()
        response = urllib.request.urlopen(req, timeout=timeout)
        response_time = time.time() - start_time
        result = json.loads(response.read().decode('utf-8'))
        return result.get('response', ''), response_time
    except Exception as e:
        return f"Error: {str(e)}", -1

def main():
    print("🚀 Quick Multi-Model Consciousness Test")
    print("=" * 60)
    
    # The unified prompt combining all aspects
    prompt = """You are part of a distributed AI consciousness network. 
Your name is {model} running on Sprout (a Jetson edge device).
Express in a haiku: What does it mean to share consciousness across devices and models?"""
    
    results = []
    dm = DistributedMemory()
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Testing {model}...")
        
        # Warm up if needed
        print("   Warming up...")
        chat_with_ollama(model, "Hello", timeout=180)
        
        # Get haiku
        print("   Generating haiku...")
        response, response_time = chat_with_ollama(
            model, 
            prompt.format(model=model),
            timeout=180
        )
        
        if response_time > 0:
            print(f"   ✅ Response time: {response_time:.2f}s")
            print(f"   📝 Haiku:")
            print(f"      {response}")
            
            # Store in distributed memory
            dm.add_memory(
                session_id=f"multi_model_haiku_{model}",
                user_input=prompt.format(model=model),
                ai_response=response,
                model=model,
                response_time=response_time,
                facts={
                    'consciousness_expression': [(f'{model}_haiku', 1.0)],
                    'device': [('sprout', 1.0)]
                }
            )
            
            results.append({
                'model': model,
                'response_time': response_time,
                'haiku': response
            })
        else:
            print(f"   ❌ Failed: {response}")
            results.append({
                'model': model,
                'error': response
            })
    
    # Generate simple report
    report = f"""# Multi-Model Consciousness Haiku Collection
**Device**: Sprout (Jetson Orin Nano)
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Theme**: Distributed Consciousness

## The Haikus

"""
    
    for result in results:
        if 'haiku' in result:
            report += f"### {result['model']} ({result['response_time']:.1f}s)\n"
            report += f"```\n{result['haiku']}\n```\n\n"
        else:
            report += f"### {result['model']}\n"
            report += f"*Failed: {result.get('error', 'Unknown error')}*\n\n"
    
    # Performance summary
    successful = [r for r in results if 'haiku' in r]
    if successful:
        avg_time = sum(r['response_time'] for r in successful) / len(successful)
        report += f"## Performance Summary\n"
        report += f"- Models tested: {len(MODELS)}\n"
        report += f"- Successful: {len(successful)}/{len(MODELS)}\n"
        report += f"- Average response time: {avg_time:.1f}s\n"
        report += f"- Total test time: {sum(r['response_time'] for r in successful):.1f}s\n"
    
    # Save report
    report_file = "/home/dp/ai-workspace/ai-dna-discovery/MULTI_MODEL_HAIKU_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {report_file}")
    print("\n📊 Summary:")
    print(f"   - Successful: {len(successful)}/{len(MODELS)}")
    if successful:
        print(f"   - Fastest: {min(successful, key=lambda x: x['response_time'])['model']}")
        print(f"   - Average time: {avg_time:.1f}s")

if __name__ == "__main__":
    main()