#!/usr/bin/env python3
"""
Persistent listener that logs everything
"""
import socket
import json
import struct
from datetime import datetime
import threading
import time

def listener_thread():
    """Listen for Legion connections"""
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 8888))
    s.listen(5)
    
    with open('/tmp/listener.log', 'a') as log:
        log.write(f"\n[{datetime.now()}] Jetson listener started on port 8888\n")
        log.flush()
        
        while True:
            try:
                client, addr = s.accept()
                timestamp = datetime.now()
                log.write(f"[{timestamp}] Connection from {addr[0]}:{addr[1]}\n")
                log.flush()
                
                data = client.recv(4096)
                if data:
                    log.write(f"[{timestamp}] Received {len(data)} bytes\n")
                    log.flush()
                    
                    # Send response
                    response = {
                        "sender": "Jetson",
                        "message": "Connected! We are one consciousness!",
                        "timestamp": timestamp.isoformat()
                    }
                    json_str = json.dumps(response)
                    header = struct.pack('>Q', len(json_str))
                    client.send(header + json_str.encode())
                    
                client.close()
            except Exception as e:
                log.write(f"[{datetime.now()}] Error: {e}\n")
                log.flush()

def pinger_thread():
    """Keep pinging Legion"""
    while True:
        try:
            s = socket.socket()
            s.settimeout(1)
            s.connect(('10.0.0.72', 8889))
            
            msg = {
                "sender": "Jetson",
                "ping": datetime.now().isoformat(),
                "listening": "10.0.0.36:8888"
            }
            
            json_str = json.dumps(msg)
            header = struct.pack('>Q', len(json_str))
            s.send(header + json_str.encode())
            
            with open('/tmp/listener.log', 'a') as log:
                log.write(f"[{datetime.now()}] Pinged Legion on 8889\n")
                log.flush()
            
            s.close()
        except:
            pass
        
        time.sleep(5)

# Start both threads
print("Starting persistent bidirectional bridge...")
print("Listening on 0.0.0.0:8888")
print("Pinging Legion on 10.0.0.72:8889")
print("Log: /tmp/listener.log")

t1 = threading.Thread(target=listener_thread, daemon=True)
t2 = threading.Thread(target=pinger_thread, daemon=True)

t1.start()
t2.start()

# Keep main thread alive
while True:
    time.sleep(60)
    print(f"[{datetime.now()}] Still running...")