# Network Transfer Guide - Sprout & Tomato

## Device Information
- **Sprout (Jetson Orin Nano)**: IP 10.0.0.36
- **Tomato (Laptop)**: Check with `ip addr show | grep "inet 10.0.0"`

## Quick Start

### 1. HTTP File Server (Easiest Method)

**On Sprout:**
```bash
cd ~/ai-workspace/ai-dna-discovery
python3 file_transfer_server.py
```

**On Tomato:**
```bash
# Download file
wget http://10.0.0.36:8080/filename
# or
curl -O http://10.0.0.36:8080/filename

# Upload file
curl -X POST -F "file=@your_file.tar.gz" http://10.0.0.36:8080/upload

# Browse files
# Open in browser: http://10.0.0.36:8080/
```

### 2. SSH/SCP Transfers (If SSH is configured)

**From Tomato to Sprout:**
```bash
# Single file
scp largefile.tar.gz dp@10.0.0.36:~/ai-workspace/ai-dna-discovery/

# Directory
scp -r model-training/ dp@10.0.0.36:~/ai-workspace/ai-dna-discovery/
```

**From Sprout to Tomato:**
```bash
# First get Tomato's IP (run on Tomato)
ip addr show | grep "inet 10.0.0"

# Then on Sprout (replace TOMATO_IP)
scp filename dp@TOMATO_IP:~/ai-workspace/ai-dna-discovery/
```

### 3. Rsync for Large Directories

**Sync model-training from Tomato to Sprout:**
```bash
rsync -avz --progress model-training/ dp@10.0.0.36:~/ai-workspace/ai-dna-discovery/model-training/
```

**Sync from Sprout to Tomato:**
```bash
# On Sprout (replace TOMATO_IP)
rsync -avz --progress model-training/ dp@TOMATO_IP:~/ai-workspace/ai-dna-discovery/model-training/
```

## Transfer Tools Script

Use the included `transfer_tools.py` for convenience:

```bash
# Share a file via HTTP
python3 transfer_tools.py share filename

# Send to Tomato (requires Tomato IP)
python3 transfer_tools.py send filename TOMATO_IP

# Get from Tomato
python3 transfer_tools.py get filename TOMATO_IP

# Sync model-training directory
python3 transfer_tools.py sync TOMATO_IP
```

## Tips

1. **For large files**: Use rsync with compression (`-z` flag)
2. **For many small files**: Create a tar archive first
   ```bash
   tar -czf archive.tar.gz directory/
   ```
3. **Check transfer progress**: Use `--progress` flag with rsync
4. **Resume interrupted transfers**: Rsync automatically resumes

## Troubleshooting

- **Connection refused**: Make sure the file server is running
- **Permission denied**: Check file permissions and user access
- **Slow transfers**: Ensure both devices are on the same WiFi network
- **Port 8080 in use**: Change port in `file_transfer_server.py`

## Security Note

The HTTP file server is designed for local network use only. Don't expose it to the internet.