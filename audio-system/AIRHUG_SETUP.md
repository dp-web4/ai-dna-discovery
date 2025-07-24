# AIRHUG Bluetooth Audio Setup

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
   [NEW] Device XX:XX:XX:XX:XX:XX AIRHUG 01
   ```

5. **Copy the device address and run:**
   ```
   scan off
   pair XX:XX:XX:XX:XX:XX
   connect XX:XX:XX:XX:XX:XX
   trust XX:XX:XX:XX:XX:XX
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