# Hardware Configuration Changes Checklist

*Last Updated: 2025-01-24*
*Lesson Learned: UEFI ACPI change bricked Jetson Orin Nano*

## CRITICAL: Before ANY Hardware Configuration Change

### Research Phase (MANDATORY)
1. [ ] Search for specific hardware + setting combination
2. [ ] Look for "brick", "won't boot", "recovery" in results
3. [ ] Check manufacturer documentation
4. [ ] Find recovery procedures BEFORE making changes
5. [ ] Ask: "Can this setting prevent boot?"

### Documentation Phase
1. [ ] Screenshot current settings
2. [ ] Write down current configuration
3. [ ] Note exact menu path to setting
4. [ ] Document recovery method
5. [ ] Have recovery tools ready

## UEFI/BIOS Changes

### Pre-Change Checklist
1. [ ] Is this an ARM device? (Extra caution!)
2. [ ] What is current boot method?
   - [ ] Device Tree (ARM standard)
   - [ ] ACPI (usually x86)
   - [ ] UEFI
3. [ ] Do I have recovery media prepared?
4. [ ] Is serial console available?
5. [ ] Can I recover if display doesn't work?

### Never Change These Without Research
- ❌ **Hardware Description Method** (Device Tree ↔ ACPI)
- ❌ **Secure Boot** settings on ARM
- ❌ **Memory training** options
- ❌ **CPU/GPU frequency** locks
- ❌ **Power delivery** settings

### Safe Changes (Usually)
- ✅ Boot order (if you have multiple options)
- ✅ Fan curves (but test thermal first)
- ✅ Display output selection
- ✅ USB configuration
- ✅ Network boot options

## Jetson-Specific Warnings

### CRITICAL Settings
1. **Device Tree vs ACPI**
   - ⚠️ Jetson uses Device Tree
   - ⚠️ ACPI will brick the device
   - ⚠️ Requires reflashing to recover

2. **UEFI Version**
   - ⚠️ Don't downgrade UEFI
   - ⚠️ Check compatibility first

3. **Memory Configuration**
   - ⚠️ Don't change memory parameters
   - ⚠️ Can prevent POST

## Making Changes Safely

### Step-by-Step Procedure
1. [ ] Document current state (photos/notes)
2. [ ] Research the specific change
3. [ ] Prepare recovery tools:
   - [ ] SDK Manager ready
   - [ ] Recovery mode pins identified
   - [ ] USB cable tested
   - [ ] Host computer prepared
4. [ ] Make ONE change at a time
5. [ ] Test boot immediately
6. [ ] Document what worked

### After Each Change
1. [ ] Does it boot?
2. [ ] Any error messages?
3. [ ] Is performance affected?
4. [ ] Are all devices detected?
5. [ ] Should I revert?

## Recovery Procedures

### If Device Won't Boot
1. [ ] Don't panic
2. [ ] Try recovery mode:
   - Jetson: Bridge recovery pins
   - PC: Clear CMOS
3. [ ] Connect serial console if available
4. [ ] Use manufacturer recovery tools
5. [ ] Document what failed

### Jetson Recovery Checklist
1. [ ] Power off completely
2. [ ] Connect USB-C to host computer
3. [ ] Bridge pins 9-10 on J14
4. [ ] Power on
5. [ ] Run SDK Manager
6. [ ] Flash QSPI/bootloader
7. [ ] Remove bridge
8. [ ] Test boot

## Common Hardware Tasks

### Adding Storage
1. [ ] Check power requirements
2. [ ] Verify interface compatibility
3. [ ] Update device tree if needed
4. [ ] Test detection before mounting
5. [ ] Add to `/etc/fstab` carefully

### Camera Configuration
1. [ ] Check device tree entries
2. [ ] Verify I2C addresses
3. [ ] Test one camera at a time
4. [ ] Monitor power consumption
5. [ ] Check thermal impact

### Cooling Changes
1. [ ] Monitor temps before changes
2. [ ] Make incremental adjustments
3. [ ] Test under load
4. [ ] Verify fan control works
5. [ ] Set safe defaults

## What We Learned

### From the ACPI Incident
- ARM devices are NOT PCs
- UEFI settings can be permanent
- Always research ARM-specific issues
- Recovery tools must be ready BEFORE
- One wrong setting = hours of recovery

### Best Practices
1. **Research First**: 10 minutes reading saves hours recovering
2. **Document Everything**: Current state is sacred
3. **One Change**: Test between each modification
4. **Recovery Ready**: Tools prepared before touching anything
5. **Ask First**: When unsure, seek guidance

## Prevention Checklist

### Before Opening UEFI/BIOS
1. [ ] Do I really need to change this?
2. [ ] Have I researched this specific setting?
3. [ ] Is recovery media prepared?
4. [ ] Do I have time for potential recovery?
5. [ ] Should I ask for guidance first?

### Red Flags to Stop
- 🚩 "Experimental" or "Advanced" warnings
- 🚩 Settings you don't understand
- 🚩 Changes to fundamental architecture
- 🚩 Anything mentioning "permanent"
- 🚩 Your gut saying "this seems risky"

---

**Remember**: Hardware is unforgiving. Software mistakes are fixable; hardware configuration mistakes might require physical intervention or complete reflashing. When in doubt, DON'T.