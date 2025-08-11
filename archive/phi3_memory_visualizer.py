#!/usr/bin/env python3
"""
Phi3 Memory System Visualizer
Shows how memory accumulates and influences responses
"""

import json
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import numpy as np

class MemoryVisualizer:
    def __init__(self, db_path="/home/dp/ai-workspace/ai-agents/phi3_memory_enhanced.db"):
        self.db_path = db_path
    
    def visualize_memory_growth(self, session_id: str):
        """Visualize how memory grows over time"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get conversation timeline
        cursor.execute('''
            SELECT timestamp, role, importance_score 
            FROM conversations 
            WHERE session_id = ? 
            ORDER BY timestamp
        ''', (session_id,))
        
        conversations = cursor.fetchall()
        
        # Get facts timeline
        cursor.execute('''
            SELECT last_mentioned, fact_type, confidence
            FROM facts
            WHERE session_id = ?
            ORDER BY last_mentioned
        ''', (session_id,))
        
        facts = cursor.fetchall()
        
        conn.close()
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Phi3 Memory System Visualization', fontsize=16)
        
        # Plot 1: Conversation importance over time
        if conversations:
            timestamps = [c[0] for c in conversations]
            importance = [c[2] for c in conversations]
            colors = ['blue' if c[1] == 'user' else 'green' for c in conversations]
            
            ax1.scatter(range(len(timestamps)), importance, c=colors, alpha=0.6, s=100)
            ax1.set_xlabel('Message Number')
            ax1.set_ylabel('Importance Score')
            ax1.set_title('Message Importance Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Add legend
            blue_patch = patches.Patch(color='blue', label='User')
            green_patch = patches.Patch(color='green', label='Assistant')
            ax1.legend(handles=[blue_patch, green_patch])
        
        # Plot 2: Fact accumulation
        if facts:
            fact_types = list(set(f[1] for f in facts))
            fact_counts = {ft: [] for ft in fact_types}
            
            for i, (_, fact_type, conf) in enumerate(facts):
                for ft in fact_types:
                    if ft == fact_type:
                        if fact_counts[ft]:
                            fact_counts[ft].append(fact_counts[ft][-1] + 1)
                        else:
                            fact_counts[ft].append(1)
                    else:
                        if fact_counts[ft]:
                            fact_counts[ft].append(fact_counts[ft][-1])
                        else:
                            fact_counts[ft].append(0)
            
            for ft, counts in fact_counts.items():
                ax2.plot(range(len(counts)), counts, label=ft, marker='o')
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Cumulative Fact Count')
            ax2.set_title('Fact Accumulation Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory usage visualization
        # Show memory allocation conceptually
        memory_types = [
            ('Short-term', 0.3, 'lightblue'),
            ('Semantic', 0.25, 'lightgreen'),
            ('Episodic', 0.25, 'lightyellow'),
            ('Working', 0.2, 'lightcoral')
        ]
        
        y_pos = 0
        for name, size, color in memory_types:
            ax3.barh(y_pos, size, color=color, edgecolor='black')
            ax3.text(size/2, y_pos, name, ha='center', va='center')
            y_pos += 1
        
        ax3.set_xlim(0, 1)
        ax3.set_xlabel('Memory Allocation')
        ax3.set_title('Conceptual Memory Distribution')
        ax3.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('/home/dp/ai-workspace/ai-agents/memory_visualization.png', dpi=150)
        plt.close()
        
        print("Memory visualization saved to memory_visualization.png")
    
    def generate_memory_report(self, session_id: str):
        """Generate a detailed memory report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get session info
        cursor.execute(
            "SELECT created_at, last_active FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        session_info = cursor.fetchone()
        
        # Get conversation stats
        cursor.execute('''
            SELECT COUNT(*), AVG(importance_score), MAX(importance_score)
            FROM conversations
            WHERE session_id = ?
        ''', (session_id,))
        
        conv_count, avg_imp, max_imp = cursor.fetchone()
        
        # Get fact summary
        cursor.execute('''
            SELECT fact_type, COUNT(*), AVG(confidence)
            FROM facts
            WHERE session_id = ?
            GROUP BY fact_type
        ''', (session_id,))
        
        fact_summary = cursor.fetchall()
        
        # Get most important facts
        cursor.execute('''
            SELECT fact_type, fact_value, confidence
            FROM facts
            WHERE session_id = ?
            ORDER BY confidence DESC, frequency DESC
            LIMIT 5
        ''', (session_id,))
        
        top_facts = cursor.fetchall()
        
        conn.close()
        
        # Generate report
        report = f"""
MEMORY SYSTEM REPORT
Session: {session_id}
==================

Session Timeline:
- Created: {session_info[0] if session_info else 'Unknown'}
- Last Active: {session_info[1] if session_info else 'Unknown'}

Conversation Statistics:
- Total Exchanges: {conv_count or 0}
- Average Importance: {avg_imp or 0:.2f}
- Peak Importance: {max_imp or 0:.2f}

Fact Summary:
"""
        
        for fact_type, count, avg_conf in fact_summary:
            report += f"- {fact_type}: {count} facts (avg confidence: {avg_conf:.2f})\n"
        
        report += "\nTop Facts (by confidence):\n"
        for fact_type, fact_value, confidence in top_facts:
            report += f"- [{fact_type}] {fact_value} (confidence: {confidence:.2f})\n"
        
        report += "\nMemory System Features:\n"
        report += "✓ Persistent storage across sessions\n"
        report += "✓ Fact extraction and categorization\n"
        report += "✓ Importance-based context selection\n"
        report += "✓ Semantic memory for concepts\n"
        report += "✓ Working memory for current context\n"
        
        return report


def demo_visualization():
    """Demo the visualization capabilities"""
    from phi3_memory_enhanced import EnhancedPhi3Memory
    
    print("MEMORY VISUALIZATION DEMO")
    print("=" * 50)
    
    # Create a quick session with some data
    memory = EnhancedPhi3Memory()
    session_id = memory.create_session()
    
    # Add some test exchanges
    test_exchanges = [
        ("Hello, I'm Alice, a data scientist", "Nice to meet you Alice!"),
        ("I love Python and machine learning", "Great interests!"),
        ("Remember I have a meeting at 3pm", "I'll remember that."),
        ("What's my name?", "You're Alice."),
    ]
    
    for user_input, ai_response in test_exchanges:
        memory.add_enhanced_exchange(session_id, user_input, ai_response)
    
    # Generate visualization
    visualizer = MemoryVisualizer()
    visualizer.visualize_memory_growth(session_id)
    
    # Generate report
    report = visualizer.generate_memory_report(session_id)
    print(report)
    
    # Save report
    with open('/home/dp/ai-workspace/ai-agents/memory_report.txt', 'w') as f:
        f.write(report)
    
    print("\nVisualization and report generated!")


if __name__ == "__main__":
    demo_visualization()