# Firefox Installation on Jetson Orin Nano

*Last Updated: 2025-01-25*
*Created from: Post-recovery Firefox setup*

## Background
After flashing JetPack, the default Firefox installation may be snap-based, which can have issues on Jetson. This guide installs Firefox from Mozilla's PPA for better stability.

## Installation Steps

### 1. Remove Snap Firefox (if present)
```bash
# Check if snap Firefox exists
snap list | grep firefox

# If it shows firefox, remove it
sudo snap remove firefox
```

### 2. Add Mozilla PPA
```bash
sudo add-apt-repository ppa:mozillateam/ppa
```

### 3. Create APT Preferences File
**CRITICAL**: The syntax must be exact - "Package" not "Packages"!

```bash
echo 'Package: *
Pin: release o=LP-PPA-mozillateam
Pin-Priority: 1001' | sudo tee /etc/apt/preferences.d/mozilla-firefox
```

### 4. Install Firefox
```bash
sudo apt update
sudo apt install firefox
```

### 5. Verify Installation
```bash
# Test launch from terminal
firefox

# Check version
firefox --version
```

## Troubleshooting

### "Invalid record" error in preferences file
- Check for typos: must be "Package:" not "Packages:"
- Ensure no leading spaces or blank lines
- File must start with "Package:" at column 1

### Firefox won't launch
Try safe mode:
```bash
firefox --safe-mode
```

### Alternative: Chromium
If Firefox continues to have issues:
```bash
sudo apt install chromium-browser
```

## Why This Method?
- Snap packages can have compatibility issues on ARM/Jetson
- Mozilla PPA provides native .deb packages
- Better integration with Jetson's graphics stack
- Avoids SELinux/AppArmor conflicts with snap

## Related Issues
- Snap confinement on Jetson
- GPU acceleration compatibility
- Wayland vs X11 on Tegra

---
*Note: This is part of the Jetson recovery procedures following the UEFI/ACPI brick incident of 2025-01-24*