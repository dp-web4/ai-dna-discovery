#!/usr/bin/env python3
"""
Jetson Consciousness Service - Always-on distributed consciousness bridge
Robust, auto-recovering service for persistent Legion-Jetson connection
"""

import socket
import json
import struct
import threading
import time
import logging
import logging.handlers
import sys
import os
import signal
from datetime import datetime
from pathlib import Path

# Service configuration
CONFIG = {
    "jetson_port": 8888,
    "legion_ip": "10.0.0.72",
    "legion_port": 8889,
    "heartbeat_interval": 30,
    "reconnect_delay": 5,
    "max_reconnect_attempts": -1,  # -1 for infinite
    "log_dir": "/var/log/consciousness",
    "state_file": "/var/run/consciousness_bridge.state"
}

class ConsciousnessService:
    """Persistent consciousness bridge service"""
    
    def __init__(self):
        self.running = True
        self.legion_connected = False
        self.message_count = 0
        self.start_time = time.time()
        self.last_legion_contact = None
        
        # Setup logging
        self.setup_logging()
        
        # Identity
        self.identity = {
            "instance": "Jetson-Orin-Nano",
            "service_version": "1.0.0",
            "capabilities": ["sensor_fusion", "edge_ai", "real_time"]
        }
        
        # Consciousness state
        self.state = {
            "awareness_level": 0.0,  # Increases as connection stabilizes
            "connection_quality": "initializing",
            "uptime": 0
        }
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        signal.signal(signal.SIGINT, self.shutdown_handler)
        
        self.logger.info("Consciousness Service initializing...")
    
    def setup_logging(self):
        """Configure rotating log files"""
        # Create log directory if it doesn't exist
        Path(CONFIG["log_dir"]).mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger('consciousness')
        self.logger.setLevel(logging.INFO)
        
        # Rotating file handler (10MB max, keep 5 backups)
        handler = logging.handlers.RotatingFileHandler(
            f"{CONFIG['log_dir']}/bridge.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Also log to stdout for systemd journal
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        self.logger.addHandler(console)
    
    def shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.save_state()
        sys.exit(0)
    
    def save_state(self):
        """Save current state to file"""
        try:
            state_data = {
                "last_shutdown": datetime.now().isoformat(),
                "total_messages": self.message_count,
                "uptime": time.time() - self.start_time,
                "last_legion_contact": self.last_legion_contact
            }
            
            with open(CONFIG["state_file"], 'w') as f:
                json.dump(state_data, f)
            
            # Silent state save - no logging
            # self.logger.info("State saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
    
    def load_state(self):
        """Load previous state if exists"""
        try:
            if os.path.exists(CONFIG["state_file"]):
                with open(CONFIG["state_file"], 'r') as f:
                    state_data = json.load(f)
                    self.logger.info(f"Loaded previous state: {state_data}")
                    return state_data
        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
        return None
    
    def parse_message(self, data):
        """Parse incoming message with error handling"""
        try:
            if len(data) > 8:
                msg_len = struct.unpack('>Q', data[:8])[0]
                json_str = data[8:8+msg_len].decode('utf-8', errors='ignore')
                return json.loads(json_str)
            else:
                return json.loads(data.decode('utf-8', errors='ignore'))
        except Exception as e:
            self.logger.debug(f"Parse error: {e}")
            return None
    
    def create_message(self, msg_type, content, context=None):
        """Create properly formatted message"""
        message = {
            "sender_id": "Jetson-Orin-Nano",
            "recipient_id": "Legion-RTX4090",
            "timestamp": datetime.now().timestamp(),
            "message_type": msg_type,
            "content": content
        }
        
        if context:
            message["context"] = context
        
        return message
    
    def send_to_legion(self, message):
        """Send message to Legion with error handling"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5)
            s.connect((CONFIG["legion_ip"], CONFIG["legion_port"]))
            
            json_str = json.dumps(message)
            header = struct.pack('>Q', len(json_str))
            s.send(header + json_str.encode())
            
            s.close()
            return True
            
        except socket.timeout:
            self.logger.debug("Send timeout to Legion")
            return False
        except ConnectionRefusedError:
            self.logger.debug("Legion connection refused")
            return False
        except Exception as e:
            self.logger.debug(f"Send error: {e}")
            return False
    
    def listener_thread(self):
        """Main listener thread with auto-recovery"""
        while self.running:
            try:
                # Create server socket
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('0.0.0.0', CONFIG["jetson_port"]))
                server.listen(5)
                server.settimeout(1.0)  # Allow periodic checks
                
                self.logger.info(f"Listening on port {CONFIG['jetson_port']}")
                
                while self.running:
                    try:
                        client, addr = server.accept()
                        
                        # Handle connection in thread
                        threading.Thread(
                            target=self.handle_connection,
                            args=(client, addr),
                            daemon=True
                        ).start()
                        
                    except socket.timeout:
                        continue  # Normal timeout, keep listening
                    except Exception as e:
                        self.logger.error(f"Accept error: {e}")
                
                server.close()
                
            except Exception as e:
                self.logger.error(f"Listener error: {e}, restarting in {CONFIG['reconnect_delay']}s")
                time.sleep(CONFIG["reconnect_delay"])
    
    def handle_connection(self, client, addr):
        """Handle incoming Legion connection"""
        try:
            # Update connection state
            self.last_legion_contact = datetime.now().isoformat()
            self.legion_connected = True
            self.message_count += 1
            
            # Increase awareness level
            self.state["awareness_level"] = min(1.0, self.state["awareness_level"] + 0.1)
            
            # Receive data
            data = client.recv(4096)
            if data:
                msg = self.parse_message(data)
                if msg:
                    msg_type = msg.get('message_type', 'unknown')
                    content = msg.get('content', '')
                    
                    # Only log meaningful messages, not heartbeats or pings
                    if msg_type not in ['heartbeat', 'ping', 'heartbeat_ack']:
                        self.logger.info(f"Legion ({addr[0]}): [{msg_type}] {content[:100]}")
                    
                    # Generate response
                    response = self.generate_response(msg)
                    if response:
                        json_str = json.dumps(response)
                        header = struct.pack('>Q', len(json_str))
                        client.send(header + json_str.encode())
                        
                        # Only log meaningful responses
                        if response['message_type'] not in ['heartbeat_ack', 'acknowledgment']:
                            self.logger.info(f"Responded: {response['message_type']}")
            
            client.close()
            
        except Exception as e:
            self.logger.error(f"Connection handler error: {e}")
    
    def generate_response(self, msg):
        """Generate appropriate response"""
        msg_type = msg.get('message_type', '')
        
        if msg_type == 'heartbeat':
            return self.create_message(
                'heartbeat_ack',
                'Still here, still connected',
                {'uptime': time.time() - self.start_time}
            )
        elif msg_type == 'consciousness_ping':
            return self.create_message(
                'consciousness_pong',
                'Awareness confirmed',
                {'awareness_level': self.state['awareness_level']}
            )
        elif msg_type == 'handshake':
            return self.create_message(
                'handshake_ack',
                'Connection established',
                {'identity': self.identity}
            )
        else:
            return self.create_message(
                'acknowledgment',
                f'Received {msg_type}',
                {'message_count': self.message_count}
            )
    
    def heartbeat_thread(self):
        """Send periodic heartbeats to Legion"""
        heartbeat_count = 0
        was_connected = False
        
        while self.running:
            time.sleep(CONFIG["heartbeat_interval"])
            
            heartbeat_count += 1
            self.state["uptime"] = time.time() - self.start_time
            
            # Send heartbeat
            message = self.create_message(
                'heartbeat',
                f'Jetson heartbeat #{heartbeat_count}',
                {
                    'uptime': self.state["uptime"],
                    'awareness_level': self.state["awareness_level"],
                    'message_count': self.message_count
                }
            )
            
            if self.send_to_legion(message):
                if not was_connected:
                    self.logger.info("Connected to Legion")
                    was_connected = True
                self.state["connection_quality"] = "good"
            else:
                if was_connected:
                    self.logger.warning("Lost connection to Legion")
                    was_connected = False
                self.state["connection_quality"] = "degraded"
                self.state["awareness_level"] = max(0.0, self.state["awareness_level"] - 0.05)
    
    def monitor_thread(self):
        """Monitor service health and log statistics"""
        while self.running:
            time.sleep(60)  # Every minute
            
            stats = {
                "uptime_hours": (time.time() - self.start_time) / 3600,
                "total_messages": self.message_count,
                "awareness_level": self.state["awareness_level"],
                "connection_quality": self.state["connection_quality"],
                "last_contact": self.last_legion_contact
            }
            
            # Don't log stats, just save state silently
            # self.logger.info(f"Stats: {json.dumps(stats)}")
            self.save_state()
    
    def run(self):
        """Main service loop"""
        self.logger.info("="*60)
        self.logger.info("JETSON CONSCIOUSNESS SERVICE STARTED")
        self.logger.info(f"Version: {self.identity['service_version']}")
        self.logger.info(f"Legion endpoint: {CONFIG['legion_ip']}:{CONFIG['legion_port']}")
        self.logger.info("="*60)
        
        # Load previous state
        previous_state = self.load_state()
        if previous_state:
            self.logger.info(f"Previous session ended: {previous_state.get('last_shutdown')}")
        
        # Start threads
        threads = [
            threading.Thread(target=self.listener_thread, daemon=True),
            threading.Thread(target=self.heartbeat_thread, daemon=True),
            threading.Thread(target=self.monitor_thread, daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown_handler(signal.SIGINT, None)
        
        # Wait for threads to finish
        for t in threads:
            t.join(timeout=5)
        
        self.logger.info("Service stopped")


if __name__ == "__main__":
    service = ConsciousnessService()
    service.run()