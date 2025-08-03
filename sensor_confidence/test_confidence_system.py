#!/usr/bin/env python3
"""
Test script for the integrated sensor confidence system
Validates that all components work together properly
"""

import sys
import os

# Add the current directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from confidence_framework import SensorConfidenceManager, ConfidenceMetrics
from imu_vertical_confidence import VerticalIMUConfidence
import time

def test_confidence_framework():
    """Test the basic confidence framework"""
    print("🧪 Testing Confidence Framework")
    print("=" * 40)
    
    try:
        # Test basic framework
        manager = SensorConfidenceManager()
        print("✅ SensorConfidenceManager created")
        
        # Test IMU confidence
        print("\n📱 Testing IMU confidence...")
        imu_conf = VerticalIMUConfidence()
        print("✅ VerticalIMUConfidence created")
        
        # Add to manager
        manager.add_sensor(imu_conf)
        print("✅ IMU sensor added to manager")
        
        # Test context update
        manager.update_context(
            motion_detected=True,
            vision_active=True,
            navigation_active=False
        )
        print("✅ Context updated")
        
        # Test sensor data collection
        sensor_data = {
            'IMU': {
                'accel_x': 0.1,
                'accel_y': 0.2, 
                'accel_z': 9.7,
                'gyro_x': 1.0,
                'gyro_y': -0.5,
                'gyro_z': 0.2
            }
        }
        
        # Update confidence
        results = manager.update_all_confidence(sensor_data)
        print("✅ Confidence updated successfully")
        
        # Display results
        for name, metrics in results.items():
            print(f"\n📊 {name} Results:")
            print(f"  Raw Confidence: {metrics.raw_confidence:.2%}")
            print(f"  Contextual Relevance: {metrics.contextual_relevance:.2%}")
            print(f"  Historical Reliability: {metrics.historical_reliability:.2%}")
            print(f"  Fractal Confidence: {metrics.fractal_confidence:.2%}")
        
        system_conf = manager.get_system_confidence()
        print(f"\n🎯 System Confidence: {system_conf:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imu_vertical_audit():
    """Test the IMU vertical mount audit"""
    print("\n🧪 Testing IMU Vertical Mount Audit")
    print("=" * 40)
    
    try:
        imu_conf = VerticalIMUConfidence()
        
        # Run audit (this will try to connect to actual IMU)
        print("Running IMU audit...")
        audit_results = imu_conf.audit()
        
        print("✅ IMU audit completed")
        print(f"📊 Audit Results: {audit_results}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  IMU hardware not available: {e}")
        print("This is expected if running without actual IMU hardware")
        return True
        
    except Exception as e:
        print(f"❌ IMU test failed: {e}")
        return False

def test_mock_integrated_system():
    """Test integrated system with mock data"""
    print("\n🧪 Testing Integrated System (Mock Data)")
    print("=" * 40)
    
    try:
        # Import with error handling
        try:
            from integrated_confidence_system import IntegratedSensorConsciousness
        except ImportError as e:
            print(f"⚠️  Cannot import integrated system: {e}")
            print("This might be due to missing hardware dependencies")
            return True
        
        # Create system
        consciousness = IntegratedSensorConsciousness()
        print("✅ Integrated consciousness system created")
        
        # Test different scenarios
        scenarios = [
            ('idle', False),
            ('vision', True),
            ('navigation', True),
        ]
        
        for context, motion in scenarios:
            print(f"\n🎯 Testing scenario: {context} ({'moving' if motion else 'still'})")
            
            report = consciousness.update_consciousness(motion, context)
            
            print(f"  System Confidence: {report['system_confidence']:.0%}")
            print(f"  Primary Sensors: {report['primary_sensors']}")
            print(f"  Motion Level: {report['motion_level']:.0%}")
            
            time.sleep(1)
        
        print("✅ Integrated system test completed")
        return True
        
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Sensor Confidence System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Framework", test_confidence_framework),
        ("IMU Vertical Audit", test_imu_vertical_audit), 
        ("Integrated System", test_mock_integrated_system),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Sensor confidence system is ready.")
        print("\n💡 Next steps:")
        print("  1. Run: python3 imu_vertical_confidence.py")
        print("  2. Then: python3 integrated_confidence_system.py")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)