#!/usr/bin/env python3
"""
Quick test of binocular consciousness system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binocular_consciousness import BinocularConsciousness, SimpleCognition

def main():
    print("🧠 Testing Binocular Consciousness System")
    print("="*40)
    print("This is a modular system with:")
    print("- Independent eyes with shared codebase")
    print("- Stereo correlation engine") 
    print("- Hooks for cognition modules")
    print("\nPress 'q' to quit")
    
    # Create consciousness system
    consciousness = BinocularConsciousness()
    
    # Add simple cognition that prints what it sees
    consciousness.correlator.add_cognition_hook(SimpleCognition())
    
    # You could add more cognition modules here:
    # consciousness.correlator.add_cognition_hook(DepthAwareness())
    # consciousness.correlator.add_cognition_hook(MotionTracker())
    # consciousness.correlator.add_cognition_hook(ObjectRecognition())
    
    try:
        consciousness.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()