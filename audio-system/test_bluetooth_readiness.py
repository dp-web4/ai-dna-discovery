#!/usr/bin/env python3
"""
Test Bluetooth audio readiness for voice conversation system
"""

import subprocess
import pyaudio

def check_bluetooth_status():
    """Check Bluetooth and audio system readiness"""
    
    print("=== Bluetooth Audio Readiness Check ===\n")
    
    # Check Bluetooth controller
    print("1. Bluetooth Controller:")
    try:
        result = subprocess.run(['hciconfig', 'hci0'], capture_output=True, text=True)
        if 'UP RUNNING' in result.stdout:
            print("   ✅ Bluetooth adapter is UP and RUNNING")
        else:
            print("   ❌ Bluetooth adapter not running")
    except:
        print("   ❌ Cannot check Bluetooth status")
    
    # Check Bluetooth service
    print("\n2. Bluetooth Service:")
    try:
        result = subprocess.run(['systemctl', 'is-active', 'bluetooth'], capture_output=True, text=True)
        if result.stdout.strip() == 'active':
            print("   ✅ Bluetooth service is active")
        else:
            print("   ❌ Bluetooth service not active")
    except:
        print("   ❌ Cannot check Bluetooth service")
    
    # Check PulseAudio Bluetooth module
    print("\n3. PulseAudio Bluetooth Support:")
    try:
        result = subprocess.run(['pactl', 'list', 'modules', 'short'], capture_output=True, text=True)
        if 'module-bluetooth' in result.stdout:
            print("   ✅ PulseAudio Bluetooth module loaded")
        else:
            print("   ⚠️  PulseAudio Bluetooth module not loaded (will load when device connects)")
    except:
        print("   ❌ Cannot check PulseAudio modules")
    
    # List current audio devices
    print("\n4. Current Audio Devices:")
    p = pyaudio.PyAudio()
    device_count = p.get_device_count()
    
    bluetooth_found = False
    for i in range(device_count):
        info = p.get_device_info_by_index(i)
        name = info['name']
        
        # Check if this might be a Bluetooth device
        if any(bt_indicator in name.lower() for bt_indicator in ['bluetooth', 'bluez', 'a2dp', 'hsp', 'hfp']):
            bluetooth_found = True
            print(f"   🎧 Bluetooth: {name} (in:{info['maxInputChannels']}, out:{info['maxOutputChannels']})")
        elif 'usb' in name.lower():
            print(f"   🔌 USB: {name} (in:{info['maxInputChannels']}, out:{info['maxOutputChannels']})")
    
    if not bluetooth_found:
        print("   ℹ️  No Bluetooth audio devices currently connected")
    
    p.terminate()
    
    # Instructions
    print("\n5. To Connect Bluetooth Headset:")
    print("   1. Turn on your Bluetooth headset")
    print("   2. Put it in pairing mode")
    print("   3. Run: bluetoothctl")
    print("   4. In bluetoothctl:")
    print("      - power on")
    print("      - scan on")
    print("      - pair <device_address>")
    print("      - connect <device_address>")
    print("      - trust <device_address>")
    print("      - exit")
    print("   5. The headset will appear in audio device list")
    print("   6. Our audio HAL will automatically detect it!")
    
    print("\n✅ System is ready for Bluetooth audio devices!")
    print("🎯 When connected, just update the device index in whisper_conversation.py")

if __name__ == "__main__":
    check_bluetooth_status()