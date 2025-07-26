#!/bin/bash
# Backup script before camera connection
# Run with: bash backup_before_camera.sh

echo "🔒 Creating pre-camera configuration backup..."

BACKUP_DIR=~/camera_setup_backups/$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

echo "📁 Backup directory: $BACKUP_DIR"

# 1. Device tree backup
echo "🌳 Backing up device tree..."
sudo dtc -I fs -O dts -o $BACKUP_DIR/device_tree.dts /proc/device-tree 2>/dev/null || echo "Note: dtc warnings are normal"

# 2. Boot configuration
echo "🚀 Backing up boot configuration..."
sudo cp /boot/extlinux/extlinux.conf $BACKUP_DIR/extlinux.conf

# 3. Kernel modules
echo "🔧 Documenting kernel modules..."
lsmod > $BACKUP_DIR/kernel_modules.txt

# 4. System logs baseline
echo "📊 Saving system baselines..."
dmesg > $BACKUP_DIR/dmesg_baseline.txt
sudo journalctl -b > $BACKUP_DIR/journal_baseline.txt

# 5. Device information
echo "🎥 Documenting current video devices..."
ls -la /dev/video* 2>/dev/null > $BACKUP_DIR/video_devices.txt || echo "No video devices (expected)" > $BACKUP_DIR/video_devices.txt

# 6. I2C bus scan
echo "🔌 Scanning I2C buses..."
for i in 0 1 2 7 8 9; do
    echo "I2C Bus $i:" >> $BACKUP_DIR/i2c_baseline.txt
    sudo i2cdetect -y -r $i 2>/dev/null >> $BACKUP_DIR/i2c_baseline.txt || echo "Bus $i not available" >> $BACKUP_DIR/i2c_baseline.txt
done

# 7. Power state
echo "⚡ Recording power state..."
if [ -f /sys/kernel/debug/bpmp/debug/regulator/vdd_in/voltage ]; then
    sudo cat /sys/kernel/debug/bpmp/debug/regulator/vdd_in/voltage > $BACKUP_DIR/power_state.txt 2>/dev/null
fi

# 8. Create quick restore script
echo "🔄 Creating restore script..."
cat > $BACKUP_DIR/quick_restore.sh << 'EOF'
#!/bin/bash
echo "This backup was created on $(date)"
echo "To restore extlinux.conf:"
echo "sudo cp extlinux.conf /boot/extlinux/extlinux.conf"
echo ""
echo "Current video devices:"
cat video_devices.txt
echo ""
echo "Baseline I2C state:"
cat i2c_baseline.txt
EOF
chmod +x $BACKUP_DIR/quick_restore.sh

# 9. Document Jetson info
echo "🤖 Documenting Jetson information..."
cat > $BACKUP_DIR/jetson_info.txt << EOF
Jetson Model: Orin Nano Developer Kit
JetPack: 6.2.1
L4T: R36.4.4
CUDA: $(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)
Camera Type: IMX219 (planned)
Connection: CSI0 only (single camera)
EOF

echo "✅ Backup complete!"
echo ""
echo "📋 Backup contents:"
ls -la $BACKUP_DIR/
echo ""
echo "🚨 REMINDERS:"
echo "1. Connect camera to CSI0 port only"
echo "2. Blue side of ribbon toward pins"
echo "3. Power OFF completely before connecting"
echo "4. If any system errors appear - STOP!"
echo ""
echo "Ready to power down and connect camera!"