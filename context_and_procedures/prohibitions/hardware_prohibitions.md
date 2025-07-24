# Hardware Prohibitions

*Last Updated: 2025-01-24*
*Created After: Jetson Orin Nano ACPI brick incident*

## NEVER Change These Settings Without Research

### UEFI/BIOS Settings That Can Brick Devices

#### ARM Devices (Jetson, Raspberry Pi, etc.)
- ❌ **NEVER** change Device Tree to ACPI
- ❌ **NEVER** disable Device Tree support
- ❌ **NEVER** modify boot method architecture
- ❌ **NEVER** change memory initialization
- ❌ **NEVER** alter power sequencing

**Why**: ARM devices use Device Tree for hardware description. ACPI is x86-specific and will prevent boot.

#### Critical Settings (All Platforms)
- ❌ **NEVER** disable all display outputs
- ❌ **NEVER** set invalid memory frequencies
- ❌ **NEVER** disable all boot devices
- ❌ **NEVER** modify voltage regulators
- ❌ **NEVER** change CPU microcode settings

**Why**: These can prevent POST (Power On Self Test) and require hardware recovery.

### Power and Thermal

- ❌ **NEVER** disable thermal protection
- ❌ **NEVER** set fan speed to 0% permanently
- ❌ **NEVER** exceed voltage specifications
- ❌ **NEVER** disable over-current protection
- ❌ **NEVER** modify power phases

**Why**: Hardware damage is permanent and expensive.

### Storage Operations

- ❌ **NEVER** remove drives during operation
- ❌ **NEVER** change RAID mode with data
- ❌ **NEVER** flash firmware without backup
- ❌ **NEVER** interrupt firmware updates
- ❌ **NEVER** mix drive types in RAID 0

**Why**: Data loss is often unrecoverable.

### GPU/Accelerator Settings

- ❌ **NEVER** flash unsigned vBIOS
- ❌ **NEVER** exceed power limit specifications
- ❌ **NEVER** disable GPU memory ECC without reason
- ❌ **NEVER** modify memory timings blindly
- ❌ **NEVER** overvolt beyond specifications

**Why**: GPUs are expensive and sensitive to configuration.

## Physical Hardware

### Connections and Cables

- ❌ **NEVER** hot-plug non-hot-plug interfaces
- ❌ **NEVER** force connectors
- ❌ **NEVER** mix voltage levels (3.3V/5V)
- ❌ **NEVER** connect without checking pinouts
- ❌ **NEVER** exceed current ratings

**Why**: Wrong connections can destroy components instantly.

### ESD (Electrostatic Discharge)

- ❌ **NEVER** handle boards without grounding
- ❌ **NEVER** work on carpet without protection
- ❌ **NEVER** touch components directly
- ❌ **NEVER** use devices as work surfaces
- ❌ **NEVER** stack powered devices

**Why**: ESD damage is invisible but fatal to electronics.

### Cooling and Airflow

- ❌ **NEVER** block ventilation holes
- ❌ **NEVER** run without heatsinks
- ❌ **NEVER** use liquid cooling without testing
- ❌ **NEVER** ignore thermal warnings
- ❌ **NEVER** stack heat-generating devices

**Why**: Thermal damage is cumulative and shortens lifespan.

## Firmware and Low-Level

### Bootloader/UEFI Updates

- ❌ **NEVER** interrupt bootloader flashing
- ❌ **NEVER** flash incompatible versions
- ❌ **NEVER** skip version prerequisites  
- ❌ **NEVER** flash without recovery plan
- ❌ **NEVER** modify secure boot carelessly

**Why**: Failed bootloader = dead device without hardware programmer.

### Device Tree/ACPI

- ❌ **NEVER** mix Device Tree and ACPI
- ❌ **NEVER** delete device tree entries
- ❌ **NEVER** modify without documentation
- ❌ **NEVER** ignore compiler warnings
- ❌ **NEVER** skip testing changes

**Why**: Incorrect hardware description prevents driver loading.

## Network and Communication

### Network Settings

- ❌ **NEVER** disable all network interfaces
- ❌ **NEVER** set invalid IP configurations
- ❌ **NEVER** modify MAC addresses carelessly
- ❌ **NEVER** disable management interfaces
- ❌ **NEVER** block recovery protocols

**Why**: Can lose remote access permanently on headless systems.

### Serial/Debug Interfaces

- ❌ **NEVER** disable serial console without alternative
- ❌ **NEVER** set invalid baud rates
- ❌ **NEVER** disconnect during recovery
- ❌ **NEVER** use wrong voltage levels
- ❌ **NEVER** short TX to RX while powered

**Why**: Serial console is often the last resort for recovery.

## What Happened to Us

### The ACPI Incident
```
Setting: UEFI → Hardware Description → Device Tree
Changed to: ACPI
Result: No display, no boot, recovery mode required
Time lost: 8+ hours
Lesson: ARM devices MUST use Device Tree
```

### Why This Setting Exists
- UEFI is becoming standard on ARM
- Settings borrowed from x86 systems  
- Not all options are valid for ARM
- Manufacturers don't always protect users

## Recovery Preparedness

### Before ANY Hardware Change
1. ✅ Have recovery tools ready
2. ✅ Know recovery pin locations
3. ✅ Test recovery cable
4. ✅ Document current settings
5. ✅ Have time for potential recovery

### Required Recovery Tools
- SDK Manager (for Jetson)
- USB cables (tested!)
- Serial console adapter
- Recovery mode jumpers
- Another computer

## Golden Rules

1. **Research First**: Every setting, every time
2. **One Change**: Test between modifications
3. **Document**: Photos, notes, everything
4. **Recovery Ready**: Tools prepared BEFORE
5. **Time Buffer**: Never change before deadlines

---

**Remember**: Hardware doesn't have "undo". Every change could be permanent. The 10 minutes you save skipping research could cost days of recovery or expensive replacement.