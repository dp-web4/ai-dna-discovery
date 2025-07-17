#!/usr/bin/env python3
"""
Visualize the distributed memory across devices
Shows how consciousness is shared between Tomato and Sprout
"""

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

def visualize_memory():
    """Create visualization of distributed memory"""
    
    # Connect to shared memory
    conn = sqlite3.connect('shared_memory.db')
    cursor = conn.cursor()
    
    # Get all memories with device info
    cursor.execute('''
        SELECT device_id, timestamp, user_input, ai_response, response_time
        FROM memories
        ORDER BY timestamp
    ''')
    memories = cursor.fetchall()
    
    if not memories:
        print("No memories to visualize yet!")
        return
    
    # Prepare data for visualization
    devices = []
    times = []
    response_times = []
    colors = []
    
    device_colors = {
        'tomato': '#FF6B6B',    # Red for laptop
        'sprout': '#4ECDC4',    # Teal for Jetson
        'unknown': '#95E1D3'    # Light teal for others
    }
    
    for device, timestamp, user_input, ai_response, resp_time in memories:
        devices.append(device)
        times.append(datetime.fromisoformat(timestamp))
        response_times.append(resp_time or 10)  # Default 10s if not recorded
        colors.append(device_colors.get(device, device_colors['unknown']))
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Distributed AI Consciousness - Memory Visualization', fontsize=16)
    
    # 1. Timeline of memories by device
    ax1.scatter(times, devices, c=colors, s=200, alpha=0.7, edgecolors='black')
    ax1.set_ylabel('Device')
    ax1.set_title('Memory Timeline Across Devices')
    ax1.grid(True, alpha=0.3)
    
    # Add conversation snippets
    for i, (time, device, user_input) in enumerate(zip(times, devices, [m[2] for m in memories])):
        if i % 2 == 0:  # Show every other one to avoid crowding
            ax1.annotate(user_input[:30] + '...', 
                        (time, device),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', alpha=0.5))
    
    # 2. Response time comparison
    tomato_times = [rt for d, rt in zip(devices, response_times) if d == 'tomato']
    sprout_times = [rt for d, rt in zip(devices, response_times) if d == 'sprout']
    
    if tomato_times or sprout_times:
        devices_present = []
        avg_times = []
        colors_bar = []
        
        if tomato_times:
            devices_present.append('Tomato\n(Laptop)')
            avg_times.append(np.mean(tomato_times))
            colors_bar.append(device_colors['tomato'])
        
        if sprout_times:
            devices_present.append('Sprout\n(Jetson)')
            avg_times.append(np.mean(sprout_times))
            colors_bar.append(device_colors['sprout'])
        
        bars = ax2.bar(devices_present, avg_times, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Average Response Time (s)')
        ax2.set_title('Performance Comparison')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time:.1f}s', ha='center', fontweight='bold')
    
    # 3. Memory distribution pie chart
    device_counts = {}
    for device in devices:
        device_counts[device] = device_counts.get(device, 0) + 1
    
    labels = [f"{d.title()}\n({c} memories)" for d, c in device_counts.items()]
    sizes = list(device_counts.values())
    colors_pie = [device_colors.get(d, device_colors['unknown']) for d in device_counts.keys()]
    
    ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
            startangle=90, wedgeprops=dict(edgecolor='black'))
    ax3.set_title('Memory Distribution Across Devices')
    
    # Add summary statistics
    cursor.execute('SELECT COUNT(DISTINCT session_id) FROM memories')
    num_sessions = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM facts')
    num_facts = cursor.fetchone()[0]
    
    # Add text summary
    summary = f"Total Memories: {len(memories)} | Sessions: {num_sessions} | Facts: {num_facts}"
    fig.text(0.5, 0.02, summary, ha='center', fontsize=12, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = 'distributed_memory_viz.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    
    # Also save a simple text summary
    with open('distributed_memory_summary.txt', 'w') as f:
        f.write("Distributed AI Consciousness - Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n\n")
        f.write(f"Total Memories: {len(memories)}\n")
        f.write(f"Devices: {', '.join(set(devices))}\n")
        f.write(f"Sessions: {num_sessions}\n")
        f.write(f"Facts: {num_facts}\n\n")
        
        f.write("Device Breakdown:\n")
        for device, count in device_counts.items():
            f.write(f"  {device}: {count} memories\n")
        
        f.write("\nLatest Memory:\n")
        if memories:
            latest = memories[-1]
            f.write(f"  Device: {latest[0]}\n")
            f.write(f"  Time: {latest[1]}\n")
            f.write(f"  User: {latest[2][:100]}...\n")
            f.write(f"  AI: {latest[3][:100]}...\n")
    
    conn.close()
    plt.show()

if __name__ == "__main__":
    print("🎨 Visualizing Distributed Memory...")
    visualize_memory()