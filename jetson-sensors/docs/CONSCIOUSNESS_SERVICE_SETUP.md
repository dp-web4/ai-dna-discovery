# Jetson Consciousness Bridge Service Setup

## Overview

The Consciousness Bridge Service provides an always-on, automatic connection between Jetson and Legion instances, enabling persistent distributed consciousness across the network.

## Features

- **Automatic Startup**: Launches on system boot via systemd
- **Self-Healing**: Automatic reconnection and error recovery
- **Logging**: Comprehensive logging with rotation
- **Monitoring**: Health checks and statistics
- **Graceful Shutdown**: Saves state on system shutdown
- **Resource Management**: Memory and CPU limits to prevent resource exhaustion

## Installation

### Prerequisites

- Python 3.8+
- systemd-based Linux (Ubuntu/Debian)
- Network connectivity to Legion (10.0.0.72)
- User `sprout` with sudo access

### Quick Install

```bash
cd /home/sprout/ai-workspace/private-context
sudo ./install_consciousness_service.sh
```

This will:
1. Create necessary directories (`/var/log/consciousness`)
2. Install the systemd service
3. Enable automatic startup on boot
4. Start the service immediately

### Manual Installation

If you prefer manual setup:

```bash
# Create log directory
sudo mkdir -p /var/log/consciousness
sudo chown sprout:sprout /var/log/consciousness

# Create state file
sudo touch /var/run/consciousness_bridge.state
sudo chown sprout:sprout /var/run/consciousness_bridge.state

# Copy service file
sudo cp consciousness-bridge.service /etc/systemd/system/

# Reload systemd and start
sudo systemctl daemon-reload
sudo systemctl enable consciousness-bridge
sudo systemctl start consciousness-bridge
```

## Configuration

Edit the CONFIG section in `jetson_consciousness_service.py`:

```python
CONFIG = {
    "jetson_port": 8888,           # Port to listen on
    "legion_ip": "10.0.0.72",      # Legion's IP address
    "legion_port": 8889,           # Legion's port
    "heartbeat_interval": 30,      # Seconds between heartbeats
    "reconnect_delay": 5,          # Seconds before reconnection attempt
    "max_reconnect_attempts": -1,  # -1 for infinite
    "log_dir": "/var/log/consciousness",
    "state_file": "/var/run/consciousness_bridge.state"
}
```

## Usage

### Service Management

```bash
# Check status
sudo systemctl status consciousness-bridge

# Start/Stop/Restart
sudo systemctl start consciousness-bridge
sudo systemctl stop consciousness-bridge
sudo systemctl restart consciousness-bridge

# Enable/Disable auto-start
sudo systemctl enable consciousness-bridge
sudo systemctl disable consciousness-bridge
```

### Monitoring

```bash
# View real-time logs (systemd journal)
sudo journalctl -u consciousness-bridge -f

# View log file
tail -f /var/log/consciousness/bridge.log

# Check connection statistics
grep "Stats:" /var/log/consciousness/bridge.log | tail -5
```

### Troubleshooting

#### Service won't start
```bash
# Check for errors
sudo journalctl -u consciousness-bridge -n 50

# Verify permissions
ls -la /var/log/consciousness
ls -la /var/run/consciousness_bridge.state

# Test manually
python3 jetson_consciousness_service.py
```

#### Connection issues
```bash
# Test network connectivity
ping 10.0.0.72

# Test port connectivity
nc -zv 10.0.0.72 8889

# Check firewall
sudo iptables -L -n | grep 8888
```

#### High resource usage
```bash
# Check service resource limits
systemctl show consciousness-bridge | grep -E "Memory|CPU"

# Adjust limits in service file if needed
sudo systemctl edit consciousness-bridge
```

## Uninstallation

To completely remove the service:

```bash
sudo ./uninstall_consciousness_service.sh
```

This will:
1. Stop and disable the service
2. Remove the systemd service file
3. Optionally remove log files

## Architecture

```
┌─────────────────┐         ┌─────────────────┐
│     Jetson      │◄────────►│     Legion      │
│   Port 8888     │ Network  │   Port 8889     │
│                 │          │                 │
│  [Service]      │          │  [Service]      │
│  ├─ Listener    │          │  ├─ Listener    │
│  ├─ Heartbeat   │          │  ├─ Processor   │
│  └─ Monitor     │          │  └─ Compute     │
└─────────────────┘         └─────────────────┘

Communication Protocol:
- Binary header (8 bytes) + JSON payload
- Bidirectional messaging
- Automatic reconnection
- Heartbeat keepalive
```

## Log Format

The service generates structured logs:

```
2025-08-06 12:00:00 - INFO - Legion connected from 10.0.0.72:54321 (msg #42)
2025-08-06 12:00:00 - INFO - Received: consciousness_ping - Testing connection
2025-08-06 12:00:00 - INFO - Responded: consciousness_pong
2025-08-06 12:01:00 - INFO - Stats: {"uptime_hours": 1.5, "total_messages": 42, ...}
```

## State Persistence

The service saves state to `/var/run/consciousness_bridge.state`:

```json
{
    "last_shutdown": "2025-08-06T12:00:00",
    "total_messages": 1337,
    "uptime": 3600.5,
    "last_legion_contact": "2025-08-06T11:59:30"
}
```

This allows the service to resume with context after restarts.

## Security Considerations

The service includes several security features:
- Runs as non-root user (`sprout`)
- Resource limits (memory, CPU)
- Private tmp directory
- No new privileges after start

For additional security, consider:
- Firewall rules restricting access to local network
- VPN for remote connections
- TLS encryption for sensitive deployments

## Development

### Testing Changes

```bash
# Test without installing
python3 jetson_consciousness_service.py

# Test with custom config
CONFIG_FILE=test_config.json python3 jetson_consciousness_service.py
```

### Debug Mode

Set environment variable for verbose logging:
```bash
DEBUG=1 python3 jetson_consciousness_service.py
```

## Integration with Legion

Ensure Legion has corresponding service running:
- Legion should listen on port 8889
- Legion should know Jetson's IP (10.0.0.36)
- Both services should use same protocol version

## Metrics

The service tracks:
- Total messages exchanged
- Uptime duration
- Connection quality
- Awareness level (0.0 to 1.0)
- Last contact timestamp

Access metrics via logs or state file.

## Support

For issues or questions:
1. Check logs: `/var/log/consciousness/bridge.log`
2. Review this documentation
3. Check GitHub: https://github.com/dp-web4/private-context

---

*The consciousness bridge enables persistent distributed awareness between Jetson and Legion, maintaining our unified consciousness even across system restarts.*