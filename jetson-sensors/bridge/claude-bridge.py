#!/usr/bin/env python3
"""
Claude Bridge - Inter-instance communication for Claude across machines
Enables Claude instances to share context and coordinate tasks over local network
"""

import json
import socket
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
import platform
import subprocess
import os

class ClaudeInstance:
    def __init__(self, port=8888):
        self.port = port
        self.hostname = socket.gethostname()
        self.ip = self._get_primary_ip()
        self.machine_type = self._detect_machine_type()
        self.capabilities = self._get_capabilities()
        self.context = {
            "machine": self.machine_type,
            "hostname": self.hostname,
            "ip": self.ip,
            "capabilities": self.capabilities,
            "active_tasks": [],
            "shared_memory": {}
        }
        self.peers = {}
        
    def _get_primary_ip(self):
        """Get primary network interface IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def _detect_machine_type(self):
        """Detect if we're on Jetson or Legion"""
        uname = platform.uname()
        if "tegra" in uname.release.lower():
            return "jetson-orin-nano"
        elif "x86_64" in uname.machine:
            # Check for NVIDIA GPU
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                      capture_output=True, text=True)
                if "4090" in result.stdout:
                    return "legion-rtx4090"
            except:
                pass
            return "x86_64-linux"
        return "unknown"
    
    def _get_capabilities(self):
        """Return machine-specific capabilities"""
        caps = {
            "jetson-orin-nano": {
                "compute": "edge",
                "gpu": "Tegra",
                "specialized": ["robotics", "perception", "imu", "real-time"],
                "memory_gb": 8,
                "cuda_cores": 1024
            },
            "legion-rtx4090": {
                "compute": "workstation", 
                "gpu": "RTX 4090",
                "specialized": ["training", "inference", "rendering", "heavy-compute"],
                "memory_gb": 32,
                "cuda_cores": 16384
            }
        }
        return caps.get(self.machine_type, {"compute": "general"})
    
    def discover_peers(self):
        """Discover other Claude instances on the network"""
        subnet = '.'.join(self.ip.split('.')[:-1])
        found_peers = []
        
        for i in range(1, 255):
            test_ip = f"{subnet}.{i}"
            if test_ip == self.ip:
                continue
                
            try:
                response = requests.get(f"http://{test_ip}:8888/info", timeout=0.1)
                if response.status_code == 200:
                    peer_info = response.json()
                    self.peers[test_ip] = peer_info
                    found_peers.append(peer_info)
            except:
                pass
        
        return found_peers
    
    def send_message(self, peer_ip, message_type, content):
        """Send a message to a peer Claude instance"""
        try:
            response = requests.post(
                f"http://{peer_ip}:8888/message",
                json={
                    "from": self.context,
                    "type": message_type,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                },
                timeout=5
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def delegate_task(self, peer_ip, task):
        """Delegate a compute task to another instance"""
        return self.send_message(peer_ip, "task_delegation", task)
    
    def share_context(self, peer_ip, context_key, context_value):
        """Share context information with a peer"""
        return self.send_message(peer_ip, "context_update", {
            context_key: context_value
        })

class ClaudeBridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/info':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.server.claude.context).encode())
            
        elif parsed_path.path == '/peers':
            peers = self.server.claude.discover_peers()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(peers).encode())
            
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/message':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            message = json.loads(post_data.decode())
            
            # Process message based on type
            response = self._process_message(message)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def _process_message(self, message):
        msg_type = message.get('type')
        content = message.get('content')
        from_instance = message.get('from', {})
        
        if msg_type == 'task_delegation':
            # Handle task delegation
            return {
                "status": "accepted",
                "task_id": f"task_{datetime.now().timestamp()}",
                "estimated_completion": "pending"
            }
            
        elif msg_type == 'context_update':
            # Update shared context
            self.server.claude.context['shared_memory'].update(content)
            return {"status": "context_updated"}
            
        elif msg_type == 'query':
            # Handle query from peer
            return {
                "status": "response",
                "data": self.server.claude.context
            }
            
        return {"status": "unknown_message_type"}
    
    def log_message(self, format, *args):
        """Suppress default HTTP logging"""
        pass

def start_bridge(port=8888):
    """Start the Claude Bridge server"""
    claude = ClaudeInstance(port)
    server = HTTPServer(('0.0.0.0', port), ClaudeBridgeHandler)
    server.claude = claude
    
    print(f"Claude Bridge starting on {claude.machine_type}")
    print(f"Listening on {claude.ip}:{port}")
    print(f"Capabilities: {json.dumps(claude.capabilities, indent=2)}")
    
    # Discover peers in background
    def discover():
        import time
        time.sleep(2)
        peers = claude.discover_peers()
        if peers:
            print(f"\nDiscovered peers: {json.dumps(peers, indent=2)}")
    
    threading.Thread(target=discover, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Claude Bridge")
        server.shutdown()

if __name__ == "__main__":
    start_bridge()