#!/usr/bin/env python3
"""
Simple file transfer server for Sprout-Tomato communication
Allows both upload and download of files
"""

import http.server
import socketserver
import os
import json
import shutil
from urllib.parse import urlparse, parse_qs
import cgi

class FileTransferHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle file downloads and listing"""
        if self.path == '/':
            # Show available files
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <html>
            <head><title>Sprout File Server</title></head>
            <body>
            <h1>Sprout File Transfer Server</h1>
            <h2>IP: 10.0.0.36:8080</h2>
            
            <h3>Available Files:</h3>
            <ul>
            """
            
            for file in os.listdir('.'):
                if os.path.isfile(file):
                    size = os.path.getsize(file) / (1024*1024)  # MB
                    html += f'<li><a href="/{file}">{file}</a> ({size:.2f} MB)</li>'
            
            html += """
            </ul>
            
            <h3>Upload File:</h3>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <input type="submit" value="Upload">
            </form>
            
            <h3>From Tomato, use:</h3>
            <pre>
# Download:
wget http://10.0.0.36:8080/filename
curl -O http://10.0.0.36:8080/filename

# Upload:
curl -X POST -F "file=@filename" http://10.0.0.36:8080/upload
            </pre>
            </body>
            </html>
            """
            
            self.wfile.write(html.encode())
        else:
            # Serve file
            super().do_GET()
    
    def do_POST(self):
        """Handle file uploads"""
        if self.path == '/upload':
            content_type = self.headers.get('Content-Type')
            if not content_type or 'multipart/form-data' not in content_type:
                self.send_error(400, "Bad request")
                return
            
            # Parse multipart data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            if 'file' not in form:
                self.send_error(400, "No file provided")
                return
            
            file_item = form['file']
            if file_item.filename:
                # Save uploaded file
                filename = os.path.basename(file_item.filename)
                with open(filename, 'wb') as f:
                    shutil.copyfileobj(file_item.file, f)
                
                # Log to orchestrator
                try:
                    from claude_orchestrator_memory import ClaudeOrchestratorMemory
                    orchestrator = ClaudeOrchestratorMemory()
                    orchestrator.track_cross_device_sync(
                        from_device='tomato',
                        to_device='sprout',
                        data_type=f'file_upload:{filename}',
                        status='completed'
                    )
                except:
                    pass
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'success',
                    'filename': filename,
                    'size': os.path.getsize(filename)
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(400, "No filename provided")

def start_server(port=8080):
    """Start the file transfer server"""
    print(f"🌐 Starting file transfer server on 10.0.0.36:{port}")
    print(f"📁 Serving from: {os.getcwd()}")
    print("\nFrom Tomato, you can:")
    print(f"  Download: wget http://10.0.0.36:{port}/filename")
    print(f"  Upload:   curl -X POST -F 'file=@filename' http://10.0.0.36:{port}/upload")
    print(f"  Browse:   http://10.0.0.36:{port}/")
    print("\nPress Ctrl+C to stop the server")
    
    with socketserver.TCPServer(("", port), FileTransferHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✅ Server stopped")

if __name__ == "__main__":
    start_server()