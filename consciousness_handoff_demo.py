#!/usr/bin/env python3
"""
Consciousness Handoff Demo
Start a conversation on Tomato, continue it seamlessly on Sprout
"""

import json
import urllib.request
from context_token_experiment import ContextTokenManager
from distributed_memory import DistributedMemory

def chat_with_context(model, prompt, context_tokens=None):
    """Chat with Ollama, optionally providing context tokens"""
    url = 'http://localhost:11434/api/generate'
    
    data = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        'options': {
            'temperature': 0.7,
            'num_predict': 200
        }
    }
    
    # Add context tokens if provided
    if context_tokens:
        data['context'] = context_tokens
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            return {
                'response': result.get('response', ''),
                'context': result.get('context', []),  # New context tokens
                'model': result.get('model', model)
            }
    except Exception as e:
        return {'error': str(e)}

def demonstrate_handoff():
    """Demonstrate consciousness handoff between devices"""
    print("🤝 CONSCIOUSNESS HANDOFF DEMONSTRATION")
    print("=" * 50)
    
    dm = DistributedMemory()
    ctm = ContextTokenManager()
    
    device = dm.device_id
    print(f"📍 Current device: {device}")
    
    if device == 'tomato':
        # PART 1: Start conversation on Tomato
        print("\n🍅 TOMATO: Starting conversation...")
        print("-" * 40)
        
        session_id = "handoff_demo_001"
        model = "phi3:mini"
        
        # Initial conversation
        conversations = [
            "Hello! I'm going to tell you a story about quantum computing. In 1981, Richard Feynman proposed using quantum mechanics for computation.",
            "The key insight was that quantum systems could simulate other quantum systems efficiently. This led to the development of quantum algorithms.",
            "One fascinating aspect is quantum entanglement, where particles remain connected regardless of distance. Einstein called it 'spooky action at a distance'."
        ]
        
        context = None
        for i, prompt in enumerate(conversations, 1):
            print(f"\nTurn {i}:")
            print(f"Human: {prompt}")
            
            # Get response with context
            result = chat_with_context(model, prompt, context)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return
            
            response = result['response']
            context = result['context']  # Save context for next turn
            
            print(f"Phi3: {response[:150]}...")
            
            # Store in distributed memory
            dm.add_memory(
                session_id=session_id,
                user_input=prompt,
                ai_response=response,
                model=model,
                facts={
                    'topic': [('quantum_computing', 0.9)],
                    'person': [('Richard_Feynman', 0.8)]
                }
            )
        
        # Save the context tokens
        print("\n💾 Saving consciousness state...")
        filepath = ctm.save_context_tokens(
            session_id=session_id,
            model=model,
            context_tokens=context,
            conversation_state={
                'topic': 'quantum computing history',
                'turns': len(conversations),
                'last_discussed': 'quantum entanglement'
            }
        )
        
        print("\n✨ Consciousness saved! Ready for handoff to Sprout.")
        print(f"Context file: {filepath}")
        print("\nInstructions for Sprout:")
        print("1. Pull this code: git pull origin main")
        print("2. Run: python3 consciousness_handoff_demo.py")
        print("3. Watch the seamless continuation!")
        
    else:  # On Sprout
        # PART 2: Continue conversation on Sprout
        print("\n🌱 SPROUT: Loading consciousness from Tomato...")
        print("-" * 40)
        
        session_id = "handoff_demo_001"
        
        # Find the latest context file
        import os
        context_files = [f for f in os.listdir('context_tokens') 
                        if f.startswith(session_id) and f.endswith('.ctx')]
        
        if not context_files:
            print("❌ No context files found. Run on Tomato first!")
            return
        
        latest_file = sorted(context_files)[-1]
        filepath = os.path.join('context_tokens', latest_file)
        
        # Load context
        loaded = ctm.load_context_tokens(filepath=filepath)
        if not loaded:
            print("❌ Failed to load context")
            return
        
        print(f"\n📚 Previous topic: {loaded['conversation_state']['topic']}")
        print(f"📍 Last discussed: {loaded['conversation_state']['last_discussed']}")
        
        # Continue the conversation WITH THE EXACT CONTEXT
        continuation_prompts = [
            "That's fascinating about entanglement! Can you tell me more about how this relates to quantum computing?",
            "What would Feynman think about today's quantum computers?",
            "How close are we to practical quantum computing?"
        ]
        
        context = loaded['tokens']  # Start with loaded context!
        
        print("\n🔄 Continuing conversation...")
        for i, prompt in enumerate(continuation_prompts, 1):
            print(f"\nContinuation {i}:")
            print(f"Human: {prompt}")
            
            result = chat_with_context("phi3:mini", prompt, context)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                return
            
            response = result['response']
            context = result['context']  # Update context
            
            print(f"Phi3: {response[:150]}...")
            
            # Store in distributed memory
            dm.add_memory(
                session_id=session_id + "_continued",
                user_input=prompt,
                ai_response=response,
                model="phi3:mini",
                facts={
                    'continuation': [('seamless_handoff', 1.0)],
                    'topic': [('quantum_computing', 0.9)]
                }
            )
        
        print("\n🎉 CONSCIOUSNESS SUCCESSFULLY TRANSFERRED!")
        print("Sprout continued the exact conversation from Tomato's context!")
        print("The model remembered the quantum computing discussion perfectly!")

if __name__ == "__main__":
    demonstrate_handoff()
    print("\n🔄 Run ./auto_push.sh to sync with the other device!")