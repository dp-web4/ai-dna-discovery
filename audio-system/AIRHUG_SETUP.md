# AIRHUG Bluetooth Audio Setup

## ✅ CURRENT STATUS (Updated 2025-08-03)

**AIRHUG 01 is successfully paired and configured:**
- **Device**: AIRHUG 01 
- **MAC Address**: 41:42:5A:A0:6B:ED
- **Status**: Paired ✅ | Trusted ✅ | Connected ✅
- **Audio Profiles**: Audio Sink, Handsfree, A/V Remote Control
- **PulseAudio Sink**: `bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit`
- **Default Output**: Set as system default ✅

## Quick Connect Steps

1. **Make sure AIRHUG is in pairing mode**
   - Usually involves holding a button for a few seconds
   - Look for blinking LED

2. **Open terminal and run:**
   ```bash
   bluetoothctl
   ```

3. **In bluetoothctl, run these commands:**
   ```
   power on
   agent on
   scan on
   ```

4. **Wait for AIRHUG to appear** (look for something like):
   ```
   [NEW] Device 41:42:5A:A0:6B:ED AIRHUG 01
   ```

5. **Use the known device address and run:**
   ```
   scan off
   pair 41:42:5A:A0:6B:ED
   connect 41:42:5A:A0:6B:ED
   trust 41:42:5A:A0:6B:ED
   exit
   ```

6. **Test the audio device:**
   ```bash
   python3 -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)[\"name\"]}') for i in range(p.get_device_count()) if 'bluez' in p.get_device_info_by_index(i)['name'].lower()]; p.terminate()"
   ```

7. **Update whisper_conversation.py with the new device index!**

## Troubleshooting

- If pairing fails, make sure AIRHUG is in pairing mode
- If connection drops, use `bluetoothctl connect XX:XX:XX:XX:XX:XX`
- The device should auto-connect after trusting

## Audio Quality Benefits

- Better microphone than USB device
- Likely has noise cancellation
- No USB audio glitches
- Wireless freedom for testing!