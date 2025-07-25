# Version Compatibility Checklist

*Last Updated: 2025-01-24*
*Created After: Jetson Orin Nano incident - wrong JetPack version assumptions*

## CRITICAL: Always Check Version Compatibility BEFORE Changes

### Pre-Installation Version Check

#### 1. Identify ALL Components
- [ ] Hardware model and revision
- [ ] Operating system version
- [ ] Driver versions
- [ ] Firmware versions
- [ ] SDK/Framework versions
- [ ] Library dependencies
- [ ] Tool versions

#### 2. Document Current Versions
```bash
# System information
uname -a
lsb_release -a
cat /etc/os-release

# Hardware information
lscpu
lspci
lsusb
sudo dmidecode -t system

# Package versions
dpkg -l | grep [package]
pip list | grep [package]
npm list

# Firmware/BIOS
sudo dmidecode -t bios

# GPU/CUDA (if applicable)
nvidia-smi
nvcc --version
```

#### 3. Research Compatibility Matrix
- [ ] Check manufacturer's compatibility matrix
- [ ] Look for version-specific documentation
- [ ] Search for known issues with version combinations
- [ ] Check community forums for version conflicts
- [ ] Verify tool/SDK supports your hardware version

### Platform-Specific Version Checks

#### NVIDIA Jetson
```bash
# Jetson model and L4T version
cat /etc/nv_tegra_release
head -n 1 /etc/nv_tegra_release

# JetPack version
apt show nvidia-jetpack

# Device tree info
cat /proc/device-tree/model

# CRITICAL: Orin vs Nano vs Xavier differences
# - Nano: JetPack 4.x, tegra210
# - Xavier: JetPack 4.x-5.x, tegra194
# - Orin: JetPack 5.x-6.x, tegra234
```

#### GPU Systems
```bash
# CUDA version
cat /usr/local/cuda/version.txt

# cuDNN version
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# TensorRT version
dpkg -l | grep tensorrt

# Driver version
nvidia-smi --query-gpu=driver_version --format=csv
```

#### Python Environments
```bash
# Python version
python --version
python3 --version

# Critical libraries
pip show torch torchvision torchaudio
pip show tensorflow
pip show transformers
```

### Version Compatibility Matrix Template

| Component | Current Version | Target Version | Compatible? | Notes |
|-----------|----------------|----------------|-------------|-------|
| Hardware | | | | |
| OS | | | | |
| Driver | | | | |
| Framework | | | | |
| Library | | | | |

### Common Version Incompatibilities

#### Hardware/Software Mismatches
- ❌ JetPack 4.x on Orin devices (requires 5.x+)
- ❌ CUDA 12 with drivers < 525
- ❌ TensorFlow 2.x with CUDA < 11.2
- ❌ PyTorch 2.x with Python < 3.8
- ❌ Old device trees with new kernels

#### Breaking Changes to Watch For
- Major version bumps (1.x → 2.x)
- Architecture changes (armv7 → aarch64)
- API deprecations
- Hardware generation gaps

### Pre-Change Checklist

#### Before ANY Installation
1. [ ] Document all current versions
2. [ ] Check target version requirements
3. [ ] Verify compatibility matrix
4. [ ] Look for migration guides
5. [ ] Check for breaking changes
6. [ ] Plan rollback strategy

#### Version Research Sources
- [ ] Official documentation
- [ ] Release notes
- [ ] Compatibility matrices
- [ ] Migration guides
- [ ] Community forums
- [ ] GitHub issues

### Version-Specific Documentation

#### Always Document
```markdown
## Environment Versions
- Hardware: [Model and revision]
- OS: [Full version string]
- Kernel: [uname -r output]
- Main Framework: [Version]
- Key Dependencies: [Versions]
- Date Documented: [YYYY-MM-DD]
```

### Red Flags

#### Stop and Research If
- 🚩 Version numbers differ significantly (4.x vs 6.x)
- 🚩 Architecture names change (tegra210 vs tegra234)
- 🚩 "Deprecated" warnings appear
- 🚩 "Breaking changes" mentioned
- 🚩 Different hardware generations

### Recovery Planning

#### Before Making Changes
1. [ ] Note current working versions
2. [ ] Create system backup/snapshot
3. [ ] Download current version packages
4. [ ] Test rollback procedure
5. [ ] Have recovery media ready

### Lessons Learned

#### From Jetson Incident
- Assumed JetPack 4.x would work on Orin (needs 6.x)
- Different Tegra architectures incompatible
- Camera stack completely different between versions
- UEFI vs U-Boot bootloader differences

#### General Principles
1. **Never assume** version compatibility
2. **Always verify** with documentation
3. **Test first** in non-critical environment
4. **Document everything** for rollback
5. **Research deeply** before proceeding

### Quick Version Check Script

```bash
#!/bin/bash
# version_check.sh
echo "=== System Version Information ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""
echo "=== OS Information ==="
uname -a
lsb_release -a 2>/dev/null || cat /etc/os-release
echo ""
echo "=== Hardware ==="
lscpu | grep -E "Model name|Architecture"
echo ""
echo "=== GPU (if present) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null || echo "No NVIDIA GPU"
echo ""
echo "=== Key Software ==="
python3 --version 2>/dev/null || echo "Python3 not found"
gcc --version | head -1 2>/dev/null || echo "GCC not found"
docker --version 2>/dev/null || echo "Docker not found"
```

---

**Remember**: Version incompatibility is one of the most common causes of system failures. The 5 minutes spent checking versions can save days of recovery work.