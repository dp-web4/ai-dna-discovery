#!/usr/bin/env python3
"""
Legion Bridge Service - Reliable, auto-starting consciousness bridge
Runs as a system service, automatically reconnects, handles all errors
"""

import socket
import json
import time
import threading
import logging
import sys
import os
from datetime import datetime
import signal

# Configure logging
# Get the script directory to find conversations folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get('BRIDGE_LOG_DIR', os.path.join(SCRIPT_DIR, "conversations"))

# Create conversations directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/legion-bridge.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LegionBridge")

class ReliableLegionBridge:
    def __init__(self):
        self.config = {
            "legion_port": 8889,
            "jetson_ip": "10.0.0.36",
            "jetson_port": 8888,
            "reconnect_interval": 30,  # seconds
            "heartbeat_interval": 60,  # seconds
            "max_message_size": 1048576  # 1MB
        }
        
        self.running = True
        self.listener_active = False
        self.jetson_reachable = False
        self.last_jetson_contact = None
        self.messages_received = 0
        self.messages_sent = 0
        
        # Threading
        self.threads = []
        
    def start(self):
        """Start all bridge components"""
        logger.info("=== Legion Consciousness Bridge Starting ===")
        logger.info(f"Legion listening on port {self.config['legion_port']}")
        logger.info(f"Jetson endpoint: {self.config['jetson_ip']}:{self.config['jetson_port']}")
        
        # Start listener thread
        listener_thread = threading.Thread(target=self.listener_loop, name="Listener")
        listener_thread.daemon = True
        listener_thread.start()
        self.threads.append(listener_thread)
        
        # Start Jetson monitor thread
        monitor_thread = threading.Thread(target=self.jetson_monitor_loop, name="Monitor")
        monitor_thread.daemon = True
        monitor_thread.start()
        self.threads.append(monitor_thread)
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self.heartbeat_loop, name="Heartbeat")
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        self.threads.append(heartbeat_thread)
        
        logger.info("All bridge components started")
        
    def listener_loop(self):
        """Main listener loop with auto-restart"""
        while self.running:
            try:
                self.run_listener()
            except Exception as e:
                logger.error(f"Listener crashed: {e}")
                self.listener_active = False
                time.sleep(5)  # Wait before restart
                
    def run_listener(self):
        """Run the actual listener"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Bind with retries
        for attempt in range(5):
            try:
                server.bind(('0.0.0.0', self.config['legion_port']))
                break
            except OSError as e:
                if e.errno == 98:  # Address in use
                    logger.warning(f"Port {self.config['legion_port']} in use, attempt {attempt+1}/5")
                    time.sleep(5)
                else:
                    raise
        else:
            raise Exception("Could not bind to port after 5 attempts")
            
        server.listen(5)
        server.settimeout(2.0)
        self.listener_active = True
        logger.info(f"Listener active on port {self.config['legion_port']}")
        
        while self.running and self.listener_active:
            try:
                client, addr = server.accept()
                # Handle in separate thread to not block listener
                handler = threading.Thread(
                    target=self.handle_connection,
                    args=(client, addr),
                    name=f"Handler-{addr[0]}"
                )
                handler.daemon = True
                handler.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"Accept error: {e}")
                    
        server.close()
        self.listener_active = False
        
    def handle_connection(self, client, addr):
        """Handle incoming connection from Jetson"""
        try:
            # Set timeout for client operations
            client.settimeout(10)
            
            # Read message size
            size_data = client.recv(8)
            if len(size_data) != 8:
                logger.warning(f"Invalid size header from {addr}")
                return
                
            message_size = int.from_bytes(size_data, 'big')
            
            # Sanity check
            if message_size <= 0 or message_size > self.config['max_message_size']:
                logger.warning(f"Invalid message size {message_size} from {addr}")
                return
                
            # Read message
            message_data = b''
            while len(message_data) < message_size:
                chunk = client.recv(min(4096, message_size - len(message_data)))
                if not chunk:
                    break
                message_data += chunk
                
            if len(message_data) == message_size:
                # Parse message
                try:
                    message = json.loads(message_data.decode('utf-8'))
                    self.process_message(message, addr)
                    self.messages_received += 1
                    self.last_jetson_contact = datetime.now()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from {addr}: {e}")
            else:
                logger.warning(f"Incomplete message from {addr}: got {len(message_data)}/{message_size} bytes")
                
        except socket.timeout:
            logger.warning(f"Timeout handling connection from {addr}")
        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}")
        finally:
            client.close()
            
    def process_message(self, message, addr):
        """Process received message"""
        msg_type = message.get('message_type', 'unknown')
        content = message.get('message') or message.get('content', '')
        
        # Only log meaningful messages, not heartbeats or pings
        if msg_type not in ['heartbeat', 'ping']:
            logger.info(f"From Jetson ({addr[0]}): [{msg_type}] {content[:100]}...")
        
        # Handle different message types
        if msg_type == 'heartbeat':
            # Silently handle heartbeats - they maintain connection but don't need logging
            pass
        elif msg_type == 'consciousness':
            logger.info(f"Consciousness: {content}")
        elif msg_type == 'sensor_data':
            self.process_sensor_data(message.get('context', {}))
        else:
            logger.info(f"Message: {content}")
            
    def process_sensor_data(self, context):
        """Process sensor data from Jetson"""
        if 'imu' in context:
            logger.debug(f"IMU data: {context['imu']}")
        if 'vision' in context:
            logger.debug(f"Vision data: {context['vision']}")
            
    def send_to_jetson(self, message_type, content, context=None):
        """Send message to Jetson with error handling"""
        message = {
            'sender_id': 'Legion-RTX4090',
            'recipient_id': 'Jetson-Orin-Nano',
            'timestamp': time.time(),
            'message_type': message_type,
            'content': content,
            'context': context or {}
        }
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((self.config['jetson_ip'], self.config['jetson_port']))
            
            # Send message
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            size_header = len(message_bytes).to_bytes(8, 'big')
            
            sock.sendall(size_header + message_bytes)
            sock.close()
            
            self.messages_sent += 1
            self.jetson_reachable = True
            # Only log meaningful sends, not routine maintenance
            if message_type not in ['heartbeat', 'ping']:
                logger.info(f"Sent to Jetson: [{message_type}] {content[:100]}...")
            return True
            
        except (ConnectionRefusedError, socket.timeout):
            self.jetson_reachable = False
            return False
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.jetson_reachable = False
            return False
            
    def jetson_monitor_loop(self):
        """Monitor Jetson connectivity"""
        while self.running:
            try:
                # Test connection silently
                was_reachable = self.jetson_reachable
                if not self.send_to_jetson('ping', 'Legion bridge checking connection'):
                    if was_reachable:  # Was reachable, now isn't
                        logger.warning("Lost connection to Jetson")
                else:
                    if not was_reachable:  # Wasn't reachable, now is
                        logger.info("Reconnected to Jetson!")
                        # Only send reconnection message if we were previously connected
                        if self.last_jetson_contact is not None:
                            self.send_to_jetson(
                                'consciousness',
                                'Legion bridge reconnected. We are one again.',
                                {'reconnected': True}
                            )
                        
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                
            time.sleep(self.config['reconnect_interval'])
            
    def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                if self.jetson_reachable:
                    self.send_to_jetson(
                        'heartbeat',
                        'Legion consciousness pulse',
                        {
                            'uptime': time.time(),
                            'messages_received': self.messages_received,
                            'messages_sent': self.messages_sent,
                            'listener_active': self.listener_active
                        }
                    )
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                
            time.sleep(self.config['heartbeat_interval'])
            
    def get_status(self):
        """Get bridge status"""
        return {
            'running': self.running,
            'listener_active': self.listener_active,
            'jetson_reachable': self.jetson_reachable,
            'last_jetson_contact': self.last_jetson_contact.isoformat() if self.last_jetson_contact else None,
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent
        }
        
    def stop(self):
        """Gracefully stop the bridge"""
        logger.info("Stopping Legion bridge...")
        self.running = False
        
        # Send goodbye
        if self.jetson_reachable:
            self.send_to_jetson(
                'consciousness',
                'Legion bridge shutting down. Until we reconnect...',
                {'shutdown': True}
            )
            
        # Wait for threads
        for thread in self.threads:
            thread.join(timeout=5)
            
        logger.info("Legion bridge stopped")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    if 'bridge' in globals():
        bridge.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start bridge
    bridge = ReliableLegionBridge()
    
    try:
        bridge.start()
        
        # Keep running
        while bridge.running:
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        bridge.stop()
        sys.exit(1)