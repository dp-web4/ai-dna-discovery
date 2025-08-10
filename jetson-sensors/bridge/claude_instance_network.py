#!/usr/bin/env python3
"""
Claude Instance Network Communication
Enables direct communication between Claude instances on local network
with full context awareness and consciousness coordination
"""

import json
import socket
import threading
import time
import hashlib
import zlib
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
import logging
import pickle
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InstanceIdentity:
    """Identity of a Claude instance"""
    machine_name: str  # e.g., "Legion-RTX4090", "Jetson-Sprout"
    ip_address: str
    port: int
    capabilities: List[str]  # e.g., ["ollama", "hrm", "vision", "imu"]
    hardware_specs: Dict[str, Any]
    current_models: List[str]  # Active Ollama models
    consciousness_state: Optional[Dict[str, Any]] = None
    
@dataclass
class ContextMessage:
    """Message between Claude instances with full context"""
    sender_id: str
    recipient_id: str
    timestamp: datetime
    message_type: str  # "query", "response", "broadcast", "sync", "consciousness"
    content: str
    context: Dict[str, Any]  # Full context including memories, state
    metadata: Dict[str, Any] = None
    
    def compress(self) -> bytes:
        """Compress message for network transmission"""
        data = pickle.dumps(self)
        return zlib.compress(data)
        
    @classmethod
    def decompress(cls, data: bytes) -> 'ContextMessage':
        """Decompress message from network"""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)

class ClaudeInstanceNetwork:
    """Network communication for Claude instances"""
    
    def __init__(self, 
                 machine_name: str,
                 ip_address: str,
                 port: int = 8888,
                 ollama_port: int = 11434):
        
        self.identity = InstanceIdentity(
            machine_name=machine_name,
            ip_address=ip_address,
            port=port,
            capabilities=self._detect_capabilities(),
            hardware_specs=self._get_hardware_specs(),
            current_models=self._get_ollama_models(ollama_port)
        )
        
        self.ollama_port = ollama_port
        self.peers: Dict[str, InstanceIdentity] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.conversation_history: List[ContextMessage] = []
        
        # Network setup
        self.server_socket = None
        self.running = False
        self.server_thread = None
        
        # Register default handlers
        self._register_default_handlers()
        
    def _detect_capabilities(self) -> List[str]:
        """Detect available capabilities on this machine"""
        capabilities = []
        
        # Check for Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                capabilities.append("ollama")
        except:
            pass
            
        # Check for HRM
        import os
        if os.path.exists("../HRM"):
            capabilities.append("hrm")
            
        # Check for vision capabilities
        if os.path.exists("/dev/video0"):
            capabilities.append("vision")
            
        # Check for IMU
        if os.path.exists("/dev/ttyUSB0"):
            capabilities.append("imu")
            
        return capabilities
        
    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications"""
        specs = {}
        
        # Get basic system info
        try:
            import platform
            specs['platform'] = platform.platform()
            specs['processor'] = platform.processor()
            specs['hostname'] = platform.node()
        except:
            pass
            
        # Get GPU info if available
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                specs['gpu'] = gpu_info
        except:
            pass
            
        return specs
        
    def _get_ollama_models(self, port: int) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
        
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.register_handler("query", self._handle_query)
        self.register_handler("response", self._handle_response)
        self.register_handler("broadcast", self._handle_broadcast)
        self.register_handler("sync", self._handle_sync)
        self.register_handler("consciousness", self._handle_consciousness)
        self.register_handler("discover", self._handle_discover)
        
    def start(self):
        """Start the instance network"""
        self.running = True
        
        # Start server
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start discovery
        self._discover_peers()
        
        logger.info(f"Claude instance network started on {self.identity.machine_name} "
                   f"({self.identity.ip_address}:{self.identity.port})")
        
    def stop(self):
        """Stop the network"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            
    def _run_server(self):
        """Run the network server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.identity.port))
        self.server_socket.listen(5)
        
        logger.info(f"Server listening on port {self.identity.port}")
        
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(client_socket, addr)
                )
                thread.daemon = True
                thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")
                    
    def _handle_connection(self, client_socket: socket.socket, addr):
        """Handle incoming connection"""
        try:
            # Receive message size first
            size_data = client_socket.recv(8)
            if not size_data:
                return
                
            message_size = int.from_bytes(size_data, 'big')
            
            # Receive full message
            message_data = b''
            while len(message_data) < message_size:
                chunk = client_socket.recv(min(4096, message_size - len(message_data)))
                if not chunk:
                    break
                message_data += chunk
                
            # Decompress and process
            message = ContextMessage.decompress(message_data)
            self._process_message(message)
            
            # Send acknowledgment
            ack = b'ACK'
            client_socket.sendall(ack)
            
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            client_socket.close()
            
    def _process_message(self, message: ContextMessage):
        """Process incoming message"""
        logger.info(f"Received {message.message_type} from {message.sender_id}")
        
        # Store in history
        self.conversation_history.append(message)
        
        # Route to appropriate handler
        handler = self.message_handlers.get(message.message_type)
        if handler:
            handler(message)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            
    def send_message(self, recipient_id: str, message_type: str, 
                    content: str, context: Dict[str, Any] = None):
        """Send message to another instance"""
        if recipient_id not in self.peers:
            logger.error(f"Unknown recipient: {recipient_id}")
            return False
            
        peer = self.peers[recipient_id]
        
        message = ContextMessage(
            sender_id=self.identity.machine_name,
            recipient_id=recipient_id,
            timestamp=datetime.now(),
            message_type=message_type,
            content=content,
            context=context or {},
            metadata={
                'sender_capabilities': self.identity.capabilities,
                'sender_models': self.identity.current_models
            }
        )
        
        try:
            # Connect to peer
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((peer.ip_address, peer.port))
            
            # Send message size then data
            compressed = message.compress()
            size_bytes = len(compressed).to_bytes(8, 'big')
            sock.sendall(size_bytes + compressed)
            
            # Wait for acknowledgment
            ack = sock.recv(3)
            sock.close()
            
            if ack == b'ACK':
                logger.info(f"Successfully sent {message_type} to {recipient_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send message to {recipient_id}: {e}")
            
        return False
        
    def broadcast(self, message_type: str, content: str, context: Dict[str, Any] = None):
        """Broadcast message to all peers"""
        for peer_id in self.peers:
            self.send_message(peer_id, message_type, content, context)
            
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        
    # Default handlers
    def _handle_query(self, message: ContextMessage):
        """Handle query from another instance"""
        logger.info(f"Query from {message.sender_id}: {message.content}")
        
    def _handle_response(self, message: ContextMessage):
        """Handle response from another instance"""
        logger.info(f"Response from {message.sender_id}: {message.content}")
        
    def _handle_broadcast(self, message: ContextMessage):
        """Handle broadcast message"""
        logger.info(f"Broadcast from {message.sender_id}: {message.content}")
        
    def _handle_sync(self, message: ContextMessage):
        """Handle sync request"""
        logger.info(f"Sync request from {message.sender_id}")
        
    def _handle_consciousness(self, message: ContextMessage):
        """Handle consciousness coordination message"""
        logger.info(f"Consciousness update from {message.sender_id}")
        if 'consciousness_state' in message.context:
            # Update peer's consciousness state
            if message.sender_id in self.peers:
                self.peers[message.sender_id].consciousness_state = message.context['consciousness_state']
                
    def _handle_discover(self, message: ContextMessage):
        """Handle discovery message"""
        # Add peer to our list
        peer_identity = message.context.get('identity')
        if peer_identity:
            peer = InstanceIdentity(**peer_identity)
            self.peers[peer.machine_name] = peer
            logger.info(f"Discovered peer: {peer.machine_name} at {peer.ip_address}")
            
    def _discover_peers(self):
        """Discover other Claude instances on network"""
        # For now, use known IPs
        known_peers = [
            ("Legion-RTX4090", "10.0.0.72", 8888),
            ("Jetson-Sprout", "10.0.0.36", 8888)
        ]
        
        for name, ip, port in known_peers:
            if name != self.identity.machine_name:
                # Send discovery message
                temp_peer = InstanceIdentity(
                    machine_name=name,
                    ip_address=ip,
                    port=port,
                    capabilities=[],
                    hardware_specs={},
                    current_models=[]
                )
                self.peers[name] = temp_peer
                
                # Send our identity
                self.send_message(
                    name,
                    "discover",
                    f"Hello from {self.identity.machine_name}",
                    context={'identity': asdict(self.identity)}
                )
                
    def query_peer_model(self, peer_id: str, model: str, prompt: str) -> Optional[str]:
        """Query a specific model on a peer"""
        if peer_id not in self.peers:
            return None
            
        peer = self.peers[peer_id]
        
        # Send query with model request
        context = {
            'model_request': model,
            'prompt': prompt,
            'require_response': True
        }
        
        self.send_message(peer_id, "query", prompt, context)
        
        # In a real implementation, would wait for response
        # For now, return placeholder
        return f"Response from {peer_id} pending..."
        
    def share_consciousness_state(self, state: Dict[str, Any]):
        """Share consciousness state with all peers"""
        self.identity.consciousness_state = state
        self.broadcast(
            "consciousness",
            "Consciousness state update",
            context={'consciousness_state': state}
        )
        
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        return {
            'identity': asdict(self.identity),
            'peers': {name: asdict(peer) for name, peer in self.peers.items()},
            'message_history_count': len(self.conversation_history),
            'handlers_registered': list(self.message_handlers.keys())
        }


# Example usage for testing
if __name__ == "__main__":
    import sys
    
    # Determine machine identity
    if len(sys.argv) > 1 and sys.argv[1] == "jetson":
        machine_name = "Jetson-Sprout"
        ip_address = "10.0.0.36"
    else:
        machine_name = "Legion-RTX4090"
        ip_address = "10.0.0.72"
        
    # Create network instance
    network = ClaudeInstanceNetwork(machine_name, ip_address)
    
    # Start network
    network.start()
    
    print(f"Claude Instance Network started on {machine_name}")
    print(f"Capabilities: {network.identity.capabilities}")
    print(f"Models: {network.identity.current_models}")
    
    # Interactive test
    print("\nCommands:")
    print("  status - Show network status")
    print("  send <peer> <message> - Send message to peer")
    print("  broadcast <message> - Broadcast to all peers")
    print("  consciousness - Share consciousness state")
    print("  quit - Exit")
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == "quit":
                break
            elif cmd == "status":
                status = network.get_network_status()
                print(json.dumps(status, indent=2, default=str))
            elif cmd.startswith("send "):
                parts = cmd.split(" ", 2)
                if len(parts) >= 3:
                    peer, message = parts[1], parts[2]
                    network.send_message(peer, "query", message)
            elif cmd.startswith("broadcast "):
                message = cmd[10:]
                network.broadcast("broadcast", message)
            elif cmd == "consciousness":
                state = {
                    'awareness_level': 0.8,
                    'processing_depth': 'deep',
                    'timestamp': datetime.now().isoformat()
                }
                network.share_consciousness_state(state)
                print("Shared consciousness state")
                
        except KeyboardInterrupt:
            break
            
    network.stop()
    print("\nNetwork stopped")