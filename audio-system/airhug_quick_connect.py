#!/usr/bin/env python3
"""
AIRHUG Bluetooth Audio Quick Connect
Uses the known paired AIRHUG device for instant connection
"""

import subprocess
import json
import time
import sys

# Known AIRHUG device configuration
AIRHUG_CONFIG = {
    "name": "AIRHUG 01",
    "mac_address": "41:42:5A:A0:6B:ED",
    "pulseaudio_sink": "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"
}

def run_command(cmd, shell=True):
    """Run a shell command and return result"""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_bluetooth_service():
    """Check if Bluetooth service is running"""
    print("🔍 Checking Bluetooth service...")
    success, stdout, stderr = run_command("systemctl is-active bluetooth")
    if success and "active" in stdout:
        print("✅ Bluetooth service is active")
        return True
    else:
        print("❌ Bluetooth service is not active")
        return False

def check_airhug_status():
    """Check current AIRHUG connection status"""
    print(f"🔍 Checking AIRHUG status ({AIRHUG_CONFIG['mac_address']})...")
    
    success, stdout, stderr = run_command(f"bluetoothctl info {AIRHUG_CONFIG['mac_address']}")
    if not success:
        print("❌ AIRHUG device not found in Bluetooth devices")
        return False
        
    status = {
        'paired': 'Paired: yes' in stdout,
        'trusted': 'Trusted: yes' in stdout,
        'connected': 'Connected: yes' in stdout
    }
    
    print(f"  Paired: {'✅' if status['paired'] else '❌'}")
    print(f"  Trusted: {'✅' if status['trusted'] else '❌'}")
    print(f"  Connected: {'✅' if status['connected'] else '❌'}")
    
    return status

def connect_airhug():
    """Connect to AIRHUG device"""
    print(f"🔗 Connecting to {AIRHUG_CONFIG['name']}...")
    
    success, stdout, stderr = run_command(f"bluetoothctl connect {AIRHUG_CONFIG['mac_address']}")
    if success or "Connection successful" in stdout:
        print("✅ Successfully connected to AIRHUG")
        return True
    else:
        print(f"❌ Failed to connect: {stderr}")
        return False

def set_default_audio():
    """Set AIRHUG as default audio output"""
    print("🔊 Setting AIRHUG as default audio output...")
    
    success, stdout, stderr = run_command(f"pactl set-default-sink {AIRHUG_CONFIG['pulseaudio_sink']}")
    if success:
        print("✅ AIRHUG set as default audio sink")
        return True
    else:
        print(f"❌ Failed to set default sink: {stderr}")
        return False

def test_audio():
    """Test audio output"""
    print("🎵 Testing audio output...")
    
    success, stdout, stderr = run_command("speaker-test -t sine -f 1000 -l 1 -c 1", shell=True)
    if success:
        print("✅ Audio test completed successfully")
        return True
    else:
        print(f"⚠️  Audio test had issues: {stderr}")
        return False

def check_pulseaudio_sink():
    """Check if AIRHUG sink is available in PulseAudio"""
    print("🔍 Checking PulseAudio sinks...")
    
    success, stdout, stderr = run_command("pactl list short sinks")
    if success and AIRHUG_CONFIG['pulseaudio_sink'] in stdout:
        print("✅ AIRHUG sink found in PulseAudio")
        return True
    else:
        print("❌ AIRHUG sink not found in PulseAudio")
        return False

def get_audio_device_info():
    """Get PyAudio device information for AIRHUG"""
    print("🎧 Checking PyAudio devices...")
    
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        bluetooth_devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            name = info['name'].lower()
            if any(keyword in name for keyword in ['bluetooth', 'bluez', 'airhug']):
                bluetooth_devices.append({
                    'index': i,
                    'name': info['name'],
                    'inputs': info['maxInputChannels'],
                    'outputs': info['maxOutputChannels']
                })
        
        p.terminate()
        
        if bluetooth_devices:
            print("✅ Found Bluetooth audio devices:")
            for device in bluetooth_devices:
                print(f"  Device {device['index']}: {device['name']} (in:{device['inputs']}, out:{device['outputs']})")
        else:
            print("⚠️  No Bluetooth audio devices found in PyAudio")
            
        return bluetooth_devices
        
    except ImportError:
        print("⚠️  PyAudio not available - cannot check audio device info")
        return []
    except Exception as e:
        print(f"❌ Error checking PyAudio devices: {e}")
        return []

def update_audio_config():
    """Update audio_config.json with current AIRHUG status"""
    config_path = "/home/sprout/ai-workspace/ai-dna-discovery/audio-system/audio_config.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update Bluetooth device info
        config['bluetooth_device']['last_connected'] = time.strftime('%Y-%m-%d')
        config['bluetooth_device']['mac_address'] = AIRHUG_CONFIG['mac_address']
        config['bluetooth_device']['name'] = AIRHUG_CONFIG['name']
        config['bluetooth_device']['pulseaudio_sink'] = AIRHUG_CONFIG['pulseaudio_sink']
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        print("✅ Updated audio_config.json")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not update audio config: {e}")
        return False

def main():
    """Main connection routine"""
    print("🎧 AIRHUG Bluetooth Audio Quick Connect")
    print("=" * 45)
    
    # Check prerequisites
    if not check_bluetooth_service():
        print("\n❌ Bluetooth service not running. Please start it first:")
        print("   sudo systemctl start bluetooth")
        return False
    
    # Check current status
    status = check_airhug_status()
    if not status:
        print("\n❌ AIRHUG device not found. Please pair it first:")
        print("   ./connect_airhug.sh")
        return False
    
    # Connect if not connected
    if not status['connected']:
        if not connect_airhug():
            return False
        time.sleep(2)  # Wait for connection to establish
    
    # Check PulseAudio integration
    if not check_pulseaudio_sink():
        print("\n⚠️  PulseAudio sink not available. Restarting PulseAudio...")
        run_command("systemctl --user restart pulseaudio")
        time.sleep(2)
        
        if not check_pulseaudio_sink():
            print("❌ Still no PulseAudio sink. Check Bluetooth-PulseAudio integration.")
            return False
    
    # Set as default
    if not set_default_audio():
        return False
    
    # Test audio
    if "--test" in sys.argv:
        test_audio()
    
    # Get device info
    get_audio_device_info()
    
    # Update config
    update_audio_config()
    
    print("\n🎉 AIRHUG is ready for use!")
    print(f"   Device: {AIRHUG_CONFIG['name']}")
    print(f"   Sink: {AIRHUG_CONFIG['pulseaudio_sink']}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)