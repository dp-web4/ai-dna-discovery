#!/usr/bin/env python3
"""
Run all GPT test proposals for the Coherence Engine
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

def run_test(test_name, test_path):
    """Run a single test and capture results."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout per test
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr, file=sys.stderr)
            
        # Check result
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print(f"❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {test_name} TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False

def main():
    """Run all tests in GPT proposals directory."""
    
    print("="*60)
    print("COHERENCE ENGINE TEST SUITE")
    print("Running GPT's Test Proposals")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Define test order (dependencies first)
    tests = [
        # Core components
        ("Trust Curves", "test_trust_curves.py"),
        ("Attention Trace", "test_attention_trace.py"),
        ("Pattern Lifecycle", "test_pattern_lifecycle.py"),
        
        # Advanced features
        ("Conflict Resolution", "test_conflict.py"),
        ("Latency Watchdog", "test_latency_watchdog.py"),
        ("Quality of Service", "test_qos.py"),
    ]
    
    # Track results
    results = {}
    passed = 0
    failed = 0
    
    # Get test directory
    test_dir = Path(__file__).parent / "gpt_proposals_081125"
    
    # Run each test
    for test_name, test_file in tests:
        test_path = test_dir / test_file
        
        if not test_path.exists():
            print(f"⚠️ Test file not found: {test_path}")
            results[test_name] = "NOT FOUND"
            failed += 1
            continue
            
        success = run_test(test_name, test_path)
        results[test_name] = "PASSED" if success else "FAILED"
        
        if success:
            passed += 1
        else:
            failed += 1
            
        # Brief pause between tests
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        symbol = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "⚠️"
        print(f"{symbol} {test_name}: {result}")
    
    print("\n" + "-"*40)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed} ({passed/len(tests)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(tests)*100:.1f}%)")
    
    # Overall result
    print("\n" + "="*60)
    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The Coherence Engine is ready for deployment!")
    else:
        print(f"⚠️ {failed} test(s) need attention")
        print("Review failed tests above for details")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()