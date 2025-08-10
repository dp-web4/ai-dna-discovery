#!/usr/bin/env python3
"""
Simple listener to monitor incoming connections from Legion
"""
import socket
import threading
import json
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

class ListenerHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"\n[{datetime.now()}] GET request from {self.client_address[0]}")
        print(f"Path: {self.path}")
        print(f"Headers: {dict(self.headers)}")
        
        if self.path == '/info':
            response = {
                "machine": "jetson-orin-nano",
                "status": "listening",
                "ip": "10.0.0.36"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Jetson listening")
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        print(f"\n[{datetime.now()}] POST from {self.client_address[0]}")
        print(f"Path: {self.path}")
        try:
            data = json.loads(post_data.decode())
            print(f"Message: {json.dumps(data, indent=2)}")
        except:
            print(f"Raw data: {post_data.decode()}")
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "received"}).encode())

print("Starting listener on 0.0.0.0:8888")
print("Waiting for Legion to connect...")
server = HTTPServer(('0.0.0.0', 8888), ListenerHandler)
server.serve_forever()