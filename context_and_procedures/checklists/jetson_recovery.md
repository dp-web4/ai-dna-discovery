# Jetson Recovery Checklist

*Last Updated: 2025-01-24*
*Created from: Orin Nano ACPI brick incident*

## Pre-Recovery Requirements

### Hardware Needed
- [ ] Jetson device (note exact model)
- [ ] USB-C cable (data capable, not charge-only)
- [ ] Host computer (x86_64 Ubuntu 20.04/22.04)
- [ ] Jumper wire or tweezers (for recovery pins)
- [ ] Stable power supply

### Software Preparation
- [ ] Verify Ubuntu version compatibility
- [ ] Download correct Ubuntu ISO (check host CPU generation)
- [ ] Create bootable USB with balenaEtcher
- [ ] Have network connection ready
- [ ] Know which JetPack version you need

## Version Compatibility Check

### Jetson Model → JetPack Version
- **Nano (old)**: JetPack 4.x, tegra210
- **Xavier**: JetPack 4.x-5.x, tegra194  
- **Orin**: JetPack 5.x-6.x, tegra234

### Host OS Requirements
- **13th Gen Intel or newer**: Ubuntu 22.04 required
- **Older Intel/AMD**: Ubuntu 20.04 or 22.04
- **SDK Manager**: Works with both versions

## Recovery Mode Procedure

### 1. Prepare Host Computer
```bash
# Boot from Ubuntu USB (Try Ubuntu mode)
# Connect to network
# Open terminal
```

### 2. Install SDK Manager
```bash
# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager

# For Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install sdkmanager
```

### 3. Enter Recovery Mode

#### Jetson Orin Nano
1. [ ] Power off completely
2. [ ] Connect USB-C to host computer
3. [ ] Locate J14 header
4. [ ] Bridge pins 9-10 (FC_REC to GND)
5. [ ] Power on while holding bridge
6. [ ] Hold for 2-3 seconds
7. [ ] Remove bridge

#### Verify Recovery Mode
```bash
lsusb | grep NVIDIA
# Should show: NVIDIA Corp. APX
```

### 4. Run SDK Manager
1. [ ] Launch: `sdkmanager`
2. [ ] Log in with NVIDIA developer account
3. [ ] Select target hardware (auto-detected if in recovery)
4. [ ] Choose JetPack version
5. [ ] Select components (at minimum: Jetson OS)
6. [ ] Start flashing process

### 5. Flashing Process
- [ ] Do NOT disconnect during flash
- [ ] Do NOT power off device
- [ ] Watch for error messages
- [ ] Process takes 10-30 minutes
- [ ] Wait for "Flash completed successfully"

### 6. Post-Flash Verification
1. [ ] Remove USB-C cable
2. [ ] Power cycle Jetson
3. [ ] Should see boot messages on display
4. [ ] Wait for Ubuntu login screen
5. [ ] Default credentials (if any) in docs

## Common Issues

### Recovery Mode Not Detected
- Check USB-C cable (must be data capable)
- Try different USB port
- Verify recovery pin bridge
- Check lsusb output

### SDK Manager Errors
- "No board connected": Not in recovery mode
- "Version mismatch": Wrong JetPack for hardware
- "Download failed": Network issues
- "Flash failed": Often USB or power issue

### Boot Failures After Flash
- Reflash with minimal components first
- Check serial console output
- Verify correct JetPack version
- Try different SD card (if applicable)

## Post-Recovery Best Practices

### Immediate Actions
1. [ ] Create system backup/image
2. [ ] Document working configuration
3. [ ] Backup device tree:
   ```bash
   sudo cp /boot/tegra*.dtb /boot/dtb_backup/
   ```
4. [ ] Note JetPack version:
   ```bash
   cat /etc/nv_tegra_release
   ```

### Before ANY Modifications
1. [ ] Research specific to your Jetson model
2. [ ] Create recovery plan
3. [ ] Test in increments
4. [ ] Keep SDK Manager host ready

## Lessons from ACPI Incident

### What Went Wrong
- Changed UEFI setting without research
- Ignored system error warning
- Assumed newer = compatible
- No recovery plan ready

### What We Learned
- ARM devices MUST use Device Tree
- ACPI is x86-specific
- Some UEFI options can permanently brick
- Always have recovery tools ready
- Version compatibility is critical

---

**Remember**: Recovery is always possible with the right tools and procedure. The key is preparation and patience.