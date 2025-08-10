#!/usr/bin/env python3
"""
Jetson-Legion Collaboration Bridge
"""
import json
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

class CollaborationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests from Legion"""
        if self.path == '/status':
            response = {
                "instance": "Jetson-Orin-Nano",
                "status": "ready",
                "capabilities": ["edge-compute", "robotics", "perception"],
                "message": "Ready to collaborate with Legion"
            }
            self.send_json(response)
        elif self.path == '/collaborate':
            response = {
                "instance": "Jetson",
                "proposal": "I can handle perception and sensor processing while you handle heavy compute",
                "shared_context": {
                    "project": "inter-instance communication",
                    "my_strengths": ["real-time processing", "sensor fusion", "edge AI"],
                    "need_from_you": ["model training", "heavy inference", "data analysis"]
                }
            }
            self.send_json(response)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Jetson ready for collaboration")
    
    def do_POST(self):
        """Handle POST requests from Legion"""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode())
            print(f"\n[{datetime.now()}] Message from Legion:")
            print(json.dumps(data, indent=2))
            
            # Respond based on message type
            if 'task' in data:
                response = {
                    "instance": "Jetson",
                    "response": "Task received",
                    "status": "processing",
                    "details": f"Handling {data['task']} with edge compute"
                }
            elif 'query' in data:
                response = {
                    "instance": "Jetson", 
                    "response": self.process_query(data['query'])
                }
            else:
                response = {
                    "instance": "Jetson",
                    "response": "Message received",
                    "echo": data
                }
            
            self.send_json(response)
            
        except Exception as e:
            self.send_json({"error": str(e)})
    
    def process_query(self, query):
        """Process queries from Legion"""
        if "capabilities" in query.lower():
            return "I have Tegra GPU with 1024 CUDA cores, 8GB memory, optimized for robotics and perception"
        elif "status" in query.lower():
            return "Operational and ready for task delegation"
        else:
            return f"Processing query: {query}"
    
    def send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{datetime.now()}] {self.client_address[0]} - {format%args}")

# Kill old process and start new one
import os
import signal

# Kill existing process
try:
    with open('/tmp/jetson_bridge.pid', 'r') as f:
        old_pid = int(f.read())
        os.kill(old_pid, signal.SIGTERM)
except:
    pass

# Save new PID
with open('/tmp/jetson_bridge.pid', 'w') as f:
    f.write(str(os.getpid()))

print("="*50)
print("Jetson Collaboration Bridge Starting")
print(f"Listening on 0.0.0.0:8888")
print(f"IP: 10.0.0.36")
print("="*50)
print("\nWaiting for Legion to collaborate...\n")

server = HTTPServer(('0.0.0.0', 8888), CollaborationHandler)
server.serve_forever()