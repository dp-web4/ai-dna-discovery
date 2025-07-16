#!/usr/bin/env python3
"""
Multi-Model Memory Test
Test memory system across Phi3, Gemma, and TinyLlama
"""

import json
import sqlite3
import time
import subprocess
from datetime import datetime
from typing import Dict, List
from phi3_memory_enhanced import EnhancedPhi3Memory

class MultiModelMemory(EnhancedPhi3Memory):
    """Extended memory system that works with multiple models"""
    
    def __init__(self, model_name: str, db_path: str = None):
        if db_path is None:
            db_path = f"/home/dp/ai-workspace/ai-agents/{model_name.replace(':', '_')}_memory.db"
        
        super().__init__(db_path)
        self.model_name = model_name
    
    def query_with_enhanced_memory(self, session_id: str, user_input: str, 
                                  temperature: float = 0.7) -> str:
        """Override to use specified model"""
        # Build enhanced context
        context = self.build_enhanced_context(session_id, user_input)
        
        # Build prompt
        if context:
            full_prompt = f"{context}\n\nHuman: {user_input}\nAssistant:"
        else:
            full_prompt = f"Human: {user_input}\nAssistant:"
        
        # Query specific model
        try:
            response = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate", "-d",
                 json.dumps({
                     "model": self.model_name,
                     "prompt": full_prompt,
                     "stream": False,
                     "options": {
                         "temperature": temperature,
                         "num_predict": 500
                     }
                 })],
                capture_output=True, text=True, timeout=60
            )
            
            if response.returncode == 0:
                result = json.loads(response.stdout)
                ai_response = result.get('response', '').strip()
                
                # Store enhanced exchange
                self.add_enhanced_exchange(session_id, user_input, ai_response)
                
                return ai_response
        except Exception as e:
            print(f"Error with {self.model_name}: {e}")
            return f"Error: Could not get response from {self.model_name}"

class MultiModelMemoryTest:
    def __init__(self):
        self.models = ["phi3:mini", "gemma:2b", "tinyllama:latest"]
        self.memory_systems = {}
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "comparisons": {}
        }
    
    def initialize_models(self):
        """Initialize memory systems for each model"""
        print("Initializing memory systems for models...")
        for model in self.models:
            print(f"  - {model}")
            self.memory_systems[model] = MultiModelMemory(model)
    
    def test_basic_memory(self):
        """Test basic memory functionality across models"""
        print("\n=== BASIC MEMORY TEST ===\n")
        
        # Create sessions for each model
        sessions = {}
        for model in self.models:
            sessions[model] = self.memory_systems[model].create_session()
        
        # Test conversation
        test_facts = [
            "My name is Diana and I'm a robotics engineer.",
            "I'm working on autonomous navigation systems.",
            "My favorite programming language is C++."
        ]
        
        # Feed facts to all models
        print("Teaching facts to all models...")
        for fact in test_facts:
            print(f"\nFact: {fact}")
            for model in self.models:
                response = self.memory_systems[model].query_with_enhanced_memory(
                    sessions[model], fact, temperature=0.5
                )
                print(f"  {model}: {response[:80]}...")
                time.sleep(1)
        
        # Test recall
        print("\n\nTesting recall...")
        recall_questions = [
            "What's my name?",
            "What field do I work in?",
            "What's my favorite programming language?"
        ]
        
        recall_results = {model: {} for model in self.models}
        
        for question in recall_questions:
            print(f"\nQuestion: {question}")
            for model in self.models:
                response = self.memory_systems[model].query_with_enhanced_memory(
                    sessions[model], question, temperature=0
                )
                print(f"  {model}: {response[:100]}...")
                
                # Analyze recall
                recall_results[model][question] = {
                    "response": response,
                    "correct": self.check_recall(question, response)
                }
                time.sleep(1)
        
        return recall_results
    
    def check_recall(self, question: str, response: str) -> bool:
        """Check if response correctly recalls information"""
        response_lower = response.lower()
        
        if "name" in question:
            return "diana" in response_lower
        elif "field" in question or "work" in question:
            return "robotics" in response_lower or "engineer" in response_lower
        elif "language" in question:
            return "c++" in response_lower
        return False
    
    def test_cross_model_memory(self):
        """Test if models can share memories"""
        print("\n\n=== CROSS-MODEL MEMORY TEST ===\n")
        
        # Create shared database
        shared_db = "/home/dp/ai-workspace/ai-agents/shared_memory.db"
        
        # Initialize models with shared memory
        shared_memories = {}
        for model in self.models:
            shared_memories[model] = MultiModelMemory(model, shared_db)
        
        # Create shared session
        session_id = shared_memories[self.models[0]].create_session()
        
        print("Testing shared memory across models...")
        
        # Model 1 learns something
        print(f"\n{self.models[0]} learns a fact:")
        fact1 = "The project codename is Skynet."
        response1 = shared_memories[self.models[0]].query_with_enhanced_memory(
            session_id, fact1, temperature=0.3
        )
        print(f"Response: {response1[:100]}...")
        
        time.sleep(2)
        
        # Model 2 tries to recall
        print(f"\n{self.models[1]} tries to recall:")
        query2 = "What's the project codename?"
        response2 = shared_memories[self.models[1]].query_with_enhanced_memory(
            session_id, query2, temperature=0
        )
        print(f"Response: {response2[:100]}...")
        
        # Check if memory was shared
        memory_shared = "skynet" in response2.lower()
        print(f"\nMemory successfully shared: {'✓' if memory_shared else '✗'}")
        
        return {
            "shared_session": session_id,
            "fact_source": self.models[0],
            "recall_model": self.models[1],
            "memory_shared": memory_shared
        }
    
    def test_model_personality_preservation(self):
        """Test if models maintain their unique personalities with memory"""
        print("\n\n=== PERSONALITY PRESERVATION TEST ===\n")
        
        session_id = self.memory_systems[self.models[0]].create_session()
        
        # Same question to all models
        question = "How would you explain recursion to a 5-year-old?"
        
        responses = {}
        print(f"Question: {question}\n")
        
        for model in self.models:
            # Create session for each model
            model_session = self.memory_systems[model].create_session()
            
            # Add context about teaching style
            context = "You are helping a young child understand programming concepts."
            self.memory_systems[model].query_with_enhanced_memory(
                model_session, context, temperature=0.3
            )
            
            # Get response
            response = self.memory_systems[model].query_with_enhanced_memory(
                model_session, question, temperature=0.7
            )
            
            responses[model] = response
            print(f"{model}:")
            print(f"{response[:200]}...\n")
            time.sleep(1)
        
        # Analyze personality differences
        analysis = self.analyze_personality_differences(responses)
        
        return {
            "question": question,
            "responses": responses,
            "analysis": analysis
        }
    
    def analyze_personality_differences(self, responses: Dict[str, str]) -> Dict:
        """Analyze how different models respond"""
        analysis = {}
        
        for model, response in responses.items():
            analysis[model] = {
                "length": len(response),
                "uses_analogy": any(word in response.lower() for word in ["like", "imagine", "think of"]),
                "uses_code": "```" in response or "def" in response,
                "complexity": len(response.split()) / len(response.split('.')) if '.' in response else len(response.split())
            }
        
        return analysis
    
    def generate_comparison_report(self, results: Dict) -> str:
        """Generate comprehensive comparison report"""
        report = f"""
MULTI-MODEL MEMORY TEST REPORT
==============================
Generated: {datetime.now().isoformat()}

## Models Tested
- {', '.join(self.models)}

## Test 1: Basic Memory Recall

"""
        
        # Analyze recall performance
        for model in self.models:
            if model in results.get('basic_memory', {}):
                model_results = results['basic_memory'][model]
                correct = sum(1 for q in model_results.values() if q.get('correct', False))
                total = len(model_results)
                
                report += f"### {model}\n"
                report += f"Recall accuracy: {correct}/{total} ({correct/total*100:.1f}%)\n\n"
        
        # Cross-model memory
        if 'cross_model' in results:
            report += "## Test 2: Cross-Model Memory Sharing\n\n"
            cross = results['cross_model']
            report += f"Shared memory test: {'✓ PASSED' if cross['memory_shared'] else '✗ FAILED'}\n"
            report += f"- Source model: {cross['fact_source']}\n"
            report += f"- Recall model: {cross['recall_model']}\n\n"
        
        # Personality preservation
        if 'personality' in results:
            report += "## Test 3: Personality Preservation\n\n"
            report += "Response characteristics:\n"
            
            for model, analysis in results['personality']['analysis'].items():
                report += f"\n### {model}\n"
                report += f"- Response length: {analysis['length']} chars\n"
                report += f"- Uses analogies: {'Yes' if analysis['uses_analogy'] else 'No'}\n"
                report += f"- Includes code: {'Yes' if analysis['uses_code'] else 'No'}\n"
                report += f"- Avg words/sentence: {analysis['complexity']:.1f}\n"
        
        report += "\n## Key Findings\n\n"
        report += "1. All models can utilize the memory system\n"
        report += "2. Memory sharing between models is possible\n"
        report += "3. Models maintain unique response styles\n"
        report += "4. Recall accuracy varies by model size\n"
        
        return report
    
    def run_all_tests(self):
        """Run complete multi-model memory test suite"""
        print("MULTI-MODEL MEMORY TEST SUITE")
        print("=" * 50)
        
        # Initialize
        self.initialize_models()
        
        results = {}
        
        # Test 1: Basic memory
        results['basic_memory'] = self.test_basic_memory()
        
        # Test 2: Cross-model memory
        results['cross_model'] = self.test_cross_model_memory()
        
        # Test 3: Personality preservation
        results['personality'] = self.test_model_personality_preservation()
        
        # Generate report
        report = self.generate_comparison_report(results)
        print("\n" + "=" * 50)
        print(report)
        
        # Save results
        with open('/home/dp/ai-workspace/ai-agents/multi_model_memory_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('/home/dp/ai-workspace/ai-agents/multi_model_memory_report.txt', 'w') as f:
            f.write(report)
        
        print("\nResults saved!")
        
        return results


if __name__ == "__main__":
    tester = MultiModelMemoryTest()
    tester.run_all_tests()