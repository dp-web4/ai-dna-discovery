#!/usr/bin/env python3
"""
Distributed Memory Synchronization
Enables memory sharing between multiple devices (Jetson, laptop, etc.)
"""

import json
import socket
import threading
import sqlite3
import hashlib
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
import logging
import pickle

from enhanced_memory_system import HierarchicalMemory, Memory, MemoryConfidence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemoryDelta:
    """Changes to memory since last sync"""
    device_id: str
    timestamp: datetime
    additions: List[Memory]
    updates: List[Memory]
    deletions: List[str]  # Memory IDs
    
    def compress(self) -> bytes:
        """Compress delta for network transmission"""
        data = pickle.dumps(self)
        return zlib.compress(data)
        
    @classmethod
    def decompress(cls, data: bytes) -> 'MemoryDelta':
        """Decompress delta from network"""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)

@dataclass 
class SyncPeer:
    """Remote peer for memory synchronization"""
    device_id: str
    host: str
    port: int
    last_sync: datetime = None
    is_active: bool = True

class DistributedMemory:
    """Synchronize memory across multiple devices"""
    
    def __init__(self, 
                 device_id: str,
                 memory_system: HierarchicalMemory,
                 sync_port: int = 9999,
                 sync_interval: int = 300):
        
        self.device_id = device_id
        self.memory_system = memory_system
        self.sync_port = sync_port
        self.sync_interval = sync_interval
        
        # Peer management
        self.peers: Dict[str, SyncPeer] = {}
        self.sync_lock = threading.Lock()
        
        # Sync tracking
        self.last_sync_checkpoint = datetime.now()
        self.sync_history = []
        
        # Start sync server
        self.server_thread = None
        self.running = False
        
    def start(self):
        """Start distributed memory synchronization"""
        self.running = True
        
        # Start server thread
        self.server_thread = threading.Thread(target=self._run_sync_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Start sync timer
        self.sync_timer = threading.Timer(self.sync_interval, self._periodic_sync)
        self.sync_timer.daemon = True
        self.sync_timer.start()
        
        logger.info(f"Started distributed memory sync on device {self.device_id}")
        
    def stop(self):
        """Stop synchronization"""
        self.running = False
        if self.sync_timer:
            self.sync_timer.cancel()
            
    def add_peer(self, device_id: str, host: str, port: int = 9999):
        """Add a peer device for synchronization"""
        peer = SyncPeer(device_id, host, port)
        self.peers[device_id] = peer
        logger.info(f"Added peer: {device_id} at {host}:{port}")
        
    def sync_with_peer(self, peer_id: str) -> bool:
        """Synchronize with a specific peer"""
        if peer_id not in self.peers:
            logger.error(f"Unknown peer: {peer_id}")
            return False
            
        peer = self.peers[peer_id]
        
        try:
            # Get local changes since last sync
            local_delta = self._get_memory_delta(peer.last_sync)
            
            # Connect to peer
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)  # 10 second timeout
            sock.connect((peer.host, peer.port))
            
            # Send sync request
            request = {
                'action': 'sync',
                'device_id': self.device_id,
                'delta': local_delta.compress()
            }
            
            sock.sendall(json.dumps(request).encode() + b'\n')
            
            # Receive peer's delta
            response_data = b''
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                
            sock.close()
            
            # Process response
            if response_data:
                response = json.loads(response_data.decode())
                if response['status'] == 'success':
                    # Apply peer's changes
                    peer_delta = MemoryDelta.decompress(response['delta'])
                    conflicts = self._merge_memories(peer_delta)
                    
                    # Update sync timestamp
                    peer.last_sync = datetime.now()
                    
                    logger.info(f"Synced with {peer_id}: {len(peer_delta.additions)} additions, "
                              f"{len(conflicts)} conflicts resolved")
                    return True
                    
        except Exception as e:
            logger.error(f"Sync error with {peer_id}: {e}")
            peer.is_active = False
            
        return False
        
    def _run_sync_server(self):
        """Run server to receive sync requests"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(('0.0.0.0', self.sync_port))
        server_sock.listen(5)
        
        logger.info(f"Sync server listening on port {self.sync_port}")
        
        while self.running:
            try:
                client_sock, addr = server_sock.accept()
                # Handle in separate thread
                thread = threading.Thread(
                    target=self._handle_sync_request,
                    args=(client_sock, addr)
                )
                thread.daemon = True
                thread.start()
            except Exception as e:
                if self.running:
                    logger.error(f"Server error: {e}")
                    
    def _handle_sync_request(self, client_sock: socket.socket, addr: Tuple[str, int]):
        """Handle incoming sync request"""
        try:
            # Receive request
            request_data = b''
            while b'\n' not in request_data:
                chunk = client_sock.recv(1024)
                if not chunk:
                    break
                request_data += chunk
                
            request = json.loads(request_data.decode().strip())
            
            if request['action'] == 'sync':
                # Get peer's delta
                peer_delta = MemoryDelta.decompress(request['delta'])
                
                # Apply changes
                with self.sync_lock:
                    conflicts = self._merge_memories(peer_delta)
                    
                # Get our changes to send back
                our_delta = self._get_memory_delta(peer_delta.timestamp)
                
                # Send response
                response = {
                    'status': 'success',
                    'device_id': self.device_id,
                    'delta': our_delta.compress(),
                    'conflicts_resolved': len(conflicts)
                }
                
                client_sock.sendall(json.dumps(response).encode())
                
                logger.info(f"Handled sync from {request['device_id']}")
                
        except Exception as e:
            logger.error(f"Error handling sync request: {e}")
        finally:
            client_sock.close()
            
    def _get_memory_delta(self, since: Optional[datetime] = None) -> MemoryDelta:
        """Get memory changes since timestamp"""
        if not since:
            since = datetime.now() - timedelta(days=7)  # Default to last week
            
        conn = sqlite3.connect(self.memory_system.db_path)
        c = conn.cursor()
        
        # Get additions and updates
        c.execute('''
            SELECT content, layer_type, confidence, timestamp, metadata
            FROM memory_layers
            WHERE timestamp > ?
            ORDER BY timestamp
        ''', (since.isoformat(),))
        
        additions = []
        for row in c.fetchall():
            content, mem_type, confidence, timestamp, metadata_json = row
            
            # Recreate memory object
            metadata = json.loads(metadata_json) if metadata_json else {}
            conf = MemoryConfidence(
                accuracy=confidence,
                relevance=confidence,
                reliability=confidence,
                composite=confidence
            )
            
            memory = Memory(
                content=content,
                memory_type=mem_type,
                timestamp=datetime.fromisoformat(timestamp),
                confidence=conf,
                session_id=metadata.get('session_id', 'unknown'),
                metadata=metadata
            )
            
            additions.append(memory)
            
        conn.close()
        
        return MemoryDelta(
            device_id=self.device_id,
            timestamp=datetime.now(),
            additions=additions,
            updates=[],  # Not implemented yet
            deletions=[]  # Not implemented yet
        )
        
    def _merge_memories(self, peer_delta: MemoryDelta) -> List[Memory]:
        """Merge peer memories with local, resolving conflicts"""
        conflicts = []
        
        for memory in peer_delta.additions:
            # Check for conflicts (same content, different confidence)
            existing = self._find_similar_memory(memory)
            
            if existing:
                # Resolve conflict
                resolved = self._resolve_conflict(existing, memory)
                if resolved != existing:
                    conflicts.append(memory)
                    # Update with higher confidence version
                    self._update_memory(resolved)
            else:
                # Add new memory
                self.memory_system.store_with_confidence(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    session_id=f"{memory.session_id}_synced",
                    source_confidence=memory.confidence.composite,
                    metadata=memory.metadata
                )
                
        return conflicts
        
    def _find_similar_memory(self, memory: Memory) -> Optional[Memory]:
        """Find similar memory in local system"""
        # Simple content-based matching for now
        results = self.memory_system.retrieve_with_confidence(
            query=memory.content[:50],  # First 50 chars
            limit=1
        )
        
        if results:
            found_memory, _ = results[0]
            # Check if content is very similar
            if self._content_similarity(found_memory.content, memory.content) > 0.9:
                return found_memory
                
        return None
        
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity (simple version)"""
        # Simple word overlap for now
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))
        
    def _resolve_conflict(self, local_memory: Memory, remote_memory: Memory) -> Memory:
        """Resolve conflict between local and remote memory"""
        # Resolution strategy: Higher confidence wins
        if local_memory.confidence.composite > remote_memory.confidence.composite:
            return local_memory
        elif remote_memory.confidence.composite > local_memory.confidence.composite:
            return remote_memory
        else:
            # Equal confidence - use most recent
            if local_memory.timestamp > remote_memory.timestamp:
                return local_memory
            else:
                return remote_memory
                
    def _update_memory(self, memory: Memory):
        """Update existing memory (placeholder)"""
        # In a full implementation, would update the existing memory
        logger.info(f"Updated memory: {memory.content[:50]}...")
        
    def _periodic_sync(self):
        """Periodically sync with all active peers"""
        if not self.running:
            return
            
        logger.info("Starting periodic sync...")
        
        # Sync with each active peer
        for peer_id, peer in self.peers.items():
            if peer.is_active:
                success = self.sync_with_peer(peer_id)
                if not success:
                    logger.warning(f"Failed to sync with {peer_id}")
                    
        # Schedule next sync
        if self.running:
            self.sync_timer = threading.Timer(self.sync_interval, self._periodic_sync)
            self.sync_timer.daemon = True
            self.sync_timer.start()
            
    def get_sync_status(self) -> Dict:
        """Get synchronization status"""
        status = {
            'device_id': self.device_id,
            'peers': {},
            'last_checkpoint': self.last_sync_checkpoint.isoformat(),
            'sync_history_count': len(self.sync_history)
        }
        
        for peer_id, peer in self.peers.items():
            status['peers'][peer_id] = {
                'host': peer.host,
                'port': peer.port,
                'last_sync': peer.last_sync.isoformat() if peer.last_sync else None,
                'is_active': peer.is_active
            }
            
        return status


# Example usage
if __name__ == "__main__":
    import sys
    
    # Determine device ID from command line or default
    device_id = sys.argv[1] if len(sys.argv) > 1 else "laptop"
    
    # Create memory system and distributed sync
    memory = HierarchicalMemory(f"{device_id}_distributed_memory.db")
    dist_memory = DistributedMemory(device_id, memory)
    
    # Start synchronization
    dist_memory.start()
    
    # Add peers based on device
    if device_id == "laptop":
        # Add Jetson as peer
        dist_memory.add_peer("jetson", "192.168.1.100", 9999)
    else:
        # Add laptop as peer
        dist_memory.add_peer("laptop", "192.168.1.50", 9999)
        
    print(f"Distributed memory sync started on {device_id}")
    print(f"Listening on port {dist_memory.sync_port}")
    
    # Test storing some memories
    session_id = f"dist_test_{int(datetime.now().timestamp())}"
    
    # Store device-specific memory
    memory.store_with_confidence(
        f"This memory was created on {device_id}",
        "semantic",
        session_id,
        0.9,
        metadata={'device': device_id}
    )
    
    # Manual sync test
    print("\nPress Enter to manually sync with peers...")
    input()
    
    for peer_id in dist_memory.peers:
        print(f"Syncing with {peer_id}...")
        success = dist_memory.sync_with_peer(peer_id)
        print(f"Sync {'successful' if success else 'failed'}")
        
    # Show sync status
    print("\nSync Status:")
    print(json.dumps(dist_memory.get_sync_status(), indent=2))
    
    # Keep running
    print("\nDistributed sync running. Press Ctrl+C to stop...")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        dist_memory.stop()
        print("\nStopped distributed memory sync")