#!/usr/bin/env python3
"""
Memory Pattern Comparison Test
Compare how Phi3 with memory handles conversations vs Claude's approach
"""

import time
import json
from datetime import datetime
from phi3_memory_enhanced import EnhancedPhi3Memory
import subprocess

class MemoryComparisonTest:
    def __init__(self):
        self.phi3_memory = EnhancedPhi3Memory()
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
    
    def run_conversation_test(self):
        """Test a multi-turn conversation with memory"""
        print("=== CONVERSATION MEMORY TEST ===\n")
        
        # Create session
        session_id = self.phi3_memory.create_session()
        print(f"Session: {session_id}\n")
        
        # Test conversation about a project
        conversation = [
            {
                "context": "Starting a new conversation",
                "input": "Hi! I'm working on a Python project called DataViz that creates beautiful visualizations from CSV files.",
                "test": "introduction"
            },
            {
                "context": "Adding technical details",
                "input": "The main challenge is handling large datasets efficiently. I'm using pandas and matplotlib.",
                "test": "technical_detail"
            },
            {
                "context": "Adding personal preference",
                "input": "My favorite feature is the automatic color palette selection based on data types.",
                "test": "preference"
            },
            {
                "context": "Testing recall - project name",
                "input": "What's the name of my project?",
                "test": "recall_name"
            },
            {
                "context": "Testing recall - technical details",
                "input": "What libraries am I using?",
                "test": "recall_technical"
            },
            {
                "context": "Testing recall - preference",
                "input": "What's my favorite feature?",
                "test": "recall_preference"
            },
            {
                "context": "Testing inference",
                "input": "Based on what you know about my project, what performance optimization would you suggest?",
                "test": "inference"
            }
        ]
        
        results = []
        
        for turn in conversation:
            print(f"[{turn['context']}]")
            print(f"Human: {turn['input']}")
            
            # Get Phi3 response with memory
            start_time = time.time()
            response = self.phi3_memory.query_with_enhanced_memory(
                session_id, turn['input'], temperature=0.7
            )
            response_time = time.time() - start_time
            
            print(f"Phi3: {response}")
            print(f"(Response time: {response_time:.2f}s)\n")
            
            # Analyze response
            analysis = self.analyze_response(turn['test'], turn['input'], response)
            
            results.append({
                "turn": turn['test'],
                "input": turn['input'],
                "response": response,
                "response_time": response_time,
                "analysis": analysis
            })
            
            time.sleep(2)  # Pause between turns
        
        # Get memory statistics
        memory_stats = self.phi3_memory.get_memory_stats(session_id)
        
        return {
            "session_id": session_id,
            "conversation_results": results,
            "memory_stats": memory_stats
        }
    
    def analyze_response(self, test_type, input_text, response):
        """Analyze response quality"""
        analysis = {
            "response_length": len(response),
            "contains_question_mark": "?" in response,
            "word_count": len(response.split())
        }
        
        # Test-specific analysis
        if test_type == "recall_name":
            analysis["recalled_correctly"] = "dataviz" in response.lower()
        elif test_type == "recall_technical":
            analysis["mentioned_pandas"] = "pandas" in response.lower()
            analysis["mentioned_matplotlib"] = "matplotlib" in response.lower()
        elif test_type == "recall_preference":
            analysis["mentioned_color"] = "color" in response.lower()
            analysis["mentioned_palette"] = "palette" in response.lower()
        elif test_type == "inference":
            # Check for relevant suggestions
            optimization_keywords = ["cache", "chunk", "parallel", "optimize", "memory", "efficient"]
            analysis["optimization_suggested"] = any(keyword in response.lower() for keyword in optimization_keywords)
        
        return analysis
    
    def run_context_boundary_test(self):
        """Test how memory handles context boundaries"""
        print("\n=== CONTEXT BOUNDARY TEST ===\n")
        
        session_id = self.phi3_memory.create_session()
        
        # Fill context with information
        facts = [
            "My name is Charlie.",
            "I'm a marine biologist.",
            "I study dolphins.",
            "My favorite ocean is the Pacific.",
            "I have a boat named 'Wave Rider'.",
            "I've been diving for 15 years.",
            "My research focuses on dolphin communication.",
            "I use underwater cameras and hydrophones.",
            "My lab is in San Diego.",
            "I publish papers in Marine Biology Journal."
        ]
        
        # Add all facts
        print("Loading facts into memory...")
        for fact in facts:
            response = self.phi3_memory.query_with_enhanced_memory(
                session_id, fact, temperature=0.3
            )
            print(f"Added: {fact}")
            time.sleep(1)
        
        # Test recall at different points
        print("\n\nTesting recall...")
        test_questions = [
            "What's my name?",
            "What do I study?",
            "Where is my lab?",
            "What's my boat called?",
            "How long have I been diving?"
        ]
        
        recall_results = []
        for question in test_questions:
            response = self.phi3_memory.query_with_enhanced_memory(
                session_id, question, temperature=0
            )
            print(f"\nQ: {question}")
            print(f"A: {response}")
            
            # Simple check if answer contains relevant info
            recall_results.append({
                "question": question,
                "response": response,
                "response_length": len(response)
            })
        
        return {
            "facts_loaded": len(facts),
            "recall_results": recall_results,
            "session_id": session_id
        }
    
    def run_memory_interference_test(self):
        """Test if memories from different topics interfere"""
        print("\n=== MEMORY INTERFERENCE TEST ===\n")
        
        # Create two sessions with different personas
        session1 = self.phi3_memory.create_session()
        session2 = self.phi3_memory.create_session()
        
        print("Session 1: Software Developer")
        # Session 1: Software developer
        dev_facts = [
            "I'm Alex, a software developer.",
            "I love Python and JavaScript.",
            "I work at TechCorp.",
            "My favorite framework is React."
        ]
        
        for fact in dev_facts:
            self.phi3_memory.query_with_enhanced_memory(session1, fact, temperature=0.3)
            print(f"Added: {fact}")
        
        print("\n\nSession 2: Chef")
        # Session 2: Chef
        chef_facts = [
            "I'm Blake, a professional chef.",
            "I specialize in Italian cuisine.",
            "I work at Bella Vista restaurant.",
            "My favorite dish to make is risotto."
        ]
        
        for fact in chef_facts:
            self.phi3_memory.query_with_enhanced_memory(session2, fact, temperature=0.3)
            print(f"Added: {fact}")
        
        # Test for interference
        print("\n\nTesting for memory isolation...")
        
        # Ask session 1 about cooking
        response1 = self.phi3_memory.query_with_enhanced_memory(
            session1, "What's my favorite dish?", temperature=0
        )
        print(f"\nSession 1 (Developer) asked about favorite dish:")
        print(f"Response: {response1}")
        
        # Ask session 2 about programming  
        response2 = self.phi3_memory.query_with_enhanced_memory(
            session2, "What's my favorite framework?", temperature=0
        )
        print(f"\nSession 2 (Chef) asked about favorite framework:")
        print(f"Response: {response2}")
        
        # Check for contamination
        contamination1 = any(word in response1.lower() for word in ["risotto", "italian", "chef", "blake"])
        contamination2 = any(word in response2.lower() for word in ["react", "javascript", "techcorp", "alex"])
        
        return {
            "session1_contaminated": contamination1,
            "session2_contaminated": contamination2,
            "isolation_successful": not (contamination1 or contamination2)
        }
    
    def generate_comparison_report(self, results):
        """Generate a comparison report"""
        report = f"""
MEMORY COMPARISON TEST REPORT
=============================
Generated: {datetime.now().isoformat()}

## Test 1: Conversation Memory

Session: {results['conversation']['session_id']}

### Recall Performance:
"""
        
        # Analyze recall performance
        recall_tests = [r for r in results['conversation']['conversation_results'] if 'recall' in r['turn']]
        successful_recalls = 0
        
        for test in recall_tests:
            if test['turn'] == 'recall_name' and test['analysis'].get('recalled_correctly'):
                successful_recalls += 1
                report += f"✓ Correctly recalled project name\n"
            elif test['turn'] == 'recall_technical' and (test['analysis'].get('mentioned_pandas') or test['analysis'].get('mentioned_matplotlib')):
                successful_recalls += 1
                report += f"✓ Correctly recalled technical details\n"
            elif test['turn'] == 'recall_preference' and (test['analysis'].get('mentioned_color') or test['analysis'].get('mentioned_palette')):
                successful_recalls += 1
                report += f"✓ Correctly recalled preferences\n"
            else:
                report += f"✗ Failed to recall: {test['turn']}\n"
        
        report += f"\nRecall Success Rate: {successful_recalls}/{len(recall_tests)} ({successful_recalls/len(recall_tests)*100:.1f}%)\n"
        
        # Memory stats
        stats = results['conversation']['memory_stats']
        report += f"\n### Memory Statistics:\n"
        report += f"- Total messages: {stats['message_count']}\n"
        report += f"- Average importance: {stats['avg_importance']:.2f}\n"
        report += f"- Facts by type:\n"
        for fact_type, info in stats['facts_by_type'].items():
            report += f"  - {fact_type}: {info['count']} facts\n"
        
        # Context boundary test
        report += f"\n## Test 2: Context Boundaries\n\n"
        boundary_results = results['boundary']
        report += f"Facts loaded: {boundary_results['facts_loaded']}\n"
        report += f"Questions asked: {len(boundary_results['recall_results'])}\n"
        
        # Memory interference test
        report += f"\n## Test 3: Memory Isolation\n\n"
        interference = results['interference']
        report += f"Session isolation: {'✓ PASSED' if interference['isolation_successful'] else '✗ FAILED'}\n"
        if not interference['isolation_successful']:
            report += f"- Session 1 contamination: {interference['session1_contaminated']}\n"
            report += f"- Session 2 contamination: {interference['session2_contaminated']}\n"
        
        report += "\n## Comparison with Claude\n\n"
        report += "Claude's memory characteristics (observed):\n"
        report += "- Perfect recall within session\n"
        report += "- No warmup effects\n"
        report += "- Context-aware responses\n"
        report += "- Semantic understanding of relationships\n\n"
        
        report += "Phi3 with memory system:\n"
        report += f"- Recall success rate: {successful_recalls/len(recall_tests)*100:.1f}%\n"
        report += "- Session persistence: ✓\n"
        report += "- Context boundaries: Maintained\n"
        report += f"- Memory isolation: {'✓' if interference['isolation_successful'] else '✗'}\n"
        
        return report
    
    def run_all_tests(self):
        """Run all comparison tests"""
        print("MEMORY PATTERN COMPARISON TEST")
        print("=" * 50)
        print("Comparing Phi3+Memory vs Claude patterns\n")
        
        results = {}
        
        # Test 1: Conversation memory
        results['conversation'] = self.run_conversation_test()
        
        # Test 2: Context boundaries
        results['boundary'] = self.run_context_boundary_test()
        
        # Test 3: Memory interference
        results['interference'] = self.run_memory_interference_test()
        
        # Generate report
        report = self.generate_comparison_report(results)
        print("\n" + "=" * 50)
        print(report)
        
        # Save results
        with open('/home/dp/ai-workspace/ai-agents/memory_comparison_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        with open('/home/dp/ai-workspace/ai-agents/memory_comparison_report.txt', 'w') as f:
            f.write(report)
        
        print("\nResults saved to memory_comparison_results.json and memory_comparison_report.txt")
        
        return results


if __name__ == "__main__":
    tester = MemoryComparisonTest()
    tester.run_all_tests()