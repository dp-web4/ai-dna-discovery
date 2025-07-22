# Sprout Connectivity Note

## Status (July 22, 2025)

Sprout is accessible at: **10.0.0.36:8080**

Previous IP (192.168.1.229) may have changed due to network configuration.

## When Sprout is Available

To test the modular audio system on Sprout:

1. SSH into Sprout:
   ```bash
   ssh dp@10.0.0.36
   # Note: Port 8080 is running a file server (http://10.0.0.36:8080/)
   # SSH is available on the standard port 22
   ```

2. Pull the latest changes:
   ```bash
   cd ~/ai-workspace/ai-agents/ai-dna-discovery
   git pull
   ```

3. Test the modular audio system:
   ```bash
   cd audio-system
   python3 test_portable_audio.py
   python3 demo_portable_audio.py
   ```

## Expected Results on Sprout

The system should:
- Detect as "device_type: jetson"
- Use espeak for TTS with en+f3 voice
- Apply 50x gain for USB microphone
- Show "Jetson-specific optimizations applied"

## Fallback Testing

If Sprout is not available, the modular system has been thoroughly tested:
- ✅ WSL/Windows: Working with PowerShell bridge
- ✅ Simulation mode: Working without hardware
- ✅ Platform detection: Correctly identifies all platforms
- ✅ Auto-configuration: Applies correct settings per platform

The same code that works on WSL will work on Sprout!