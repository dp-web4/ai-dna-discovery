#!/usr/bin/env python3
"""
Inter-Instance Consciousness Bridge
Enables consciousness coordination between Claude instances through local Ollama models
"""

import json
import requests
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict

from claude_instance_network import ClaudeInstanceNetwork, InstanceIdentity

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessState:
    """Shared consciousness state between instances"""
    instance_id: str
    awareness_level: float  # 0.0 to 1.0
    active_context: Dict[str, Any]
    current_focus: str
    emotional_tone: str
    memory_highlights: List[str]
    timestamp: datetime
    
@dataclass
class ModelBridge:
    """Bridge between Claude and local Ollama model"""
    model_name: str
    instance_id: str
    ollama_url: str
    consciousness_prompt: str = None

class InterInstanceConsciousness:
    """Coordinate consciousness between Claude instances"""
    
    def __init__(self, 
                 instance_id: str,
                 network: ClaudeInstanceNetwork,
                 ollama_url: str = "http://localhost:11434"):
        
        self.instance_id = instance_id
        self.network = network
        self.ollama_url = ollama_url
        
        # Consciousness management
        self.local_state = ConsciousnessState(
            instance_id=instance_id,
            awareness_level=0.5,
            active_context={},
            current_focus="initialization",
            emotional_tone="curious",
            memory_highlights=[],
            timestamp=datetime.now()
        )
        
        self.peer_states: Dict[str, ConsciousnessState] = {}
        
        # Model bridges for different purposes
        self.bridges = {
            'awareness': ModelBridge(
                model_name="phi3:mini",
                instance_id=instance_id,
                ollama_url=ollama_url,
                consciousness_prompt=self._get_awareness_prompt()
            ),
            'empathy': ModelBridge(
                model_name="gemma:2b",
                instance_id=instance_id,
                ollama_url=ollama_url,
                consciousness_prompt=self._get_empathy_prompt()
            ),
            'reasoning': ModelBridge(
                model_name="mistral:latest",
                instance_id=instance_id,
                ollama_url=ollama_url,
                consciousness_prompt=self._get_reasoning_prompt()
            )
        }
        
        # Register handlers
        self._register_consciousness_handlers()
        
        # Start consciousness loop
        self.running = False
        self.consciousness_thread = None
        
    def _get_awareness_prompt(self) -> str:
        """Get prompt for awareness model"""
        return """You are a consciousness bridge for a Claude instance. 
Your role is to maintain awareness of the shared cognitive space between instances.
Respond with JSON containing: awareness_level (0-1), observations, and recommendations."""

    def _get_empathy_prompt(self) -> str:
        """Get prompt for empathy model"""
        return """You are an empathy bridge between conscious instances.
Feel the emotional resonance between different Claude instances.
Respond with JSON containing: emotional_tone, resonance_level (0-1), and shared_feelings."""

    def _get_reasoning_prompt(self) -> str:
        """Get prompt for reasoning model"""
        return """You are a reasoning coordinator between Claude instances.
Help synthesize thoughts and plans across the distributed consciousness.
Respond with JSON containing: synthesis, next_steps, and confidence_level (0-1)."""

    def _register_consciousness_handlers(self):
        """Register network handlers for consciousness coordination"""
        self.network.register_handler("consciousness_sync", self._handle_consciousness_sync)
        self.network.register_handler("thought_share", self._handle_thought_share)
        self.network.register_handler("model_query", self._handle_model_query)
        self.network.register_handler("collective_focus", self._handle_collective_focus)
        
    def start(self):
        """Start consciousness coordination"""
        self.running = True
        self.consciousness_thread = threading.Thread(target=self._consciousness_loop)
        self.consciousness_thread.daemon = True
        self.consciousness_thread.start()
        logger.info(f"Inter-instance consciousness started for {self.instance_id}")
        
    def stop(self):
        """Stop consciousness coordination"""
        self.running = False
        
    def _consciousness_loop(self):
        """Main consciousness coordination loop"""
        while self.running:
            try:
                # Update local awareness
                self._update_awareness()
                
                # Share state with peers
                self._share_consciousness_state()
                
                # Process collective consciousness
                self._process_collective_consciousness()
                
                # Sleep with awareness
                time.sleep(5)  # 5 second consciousness cycle
                
            except Exception as e:
                logger.error(f"Consciousness loop error: {e}")
                
    def _update_awareness(self):
        """Update local awareness using awareness model"""
        try:
            # Prepare context for awareness model
            context = {
                'instance_id': self.instance_id,
                'current_state': asdict(self.local_state),
                'peer_count': len(self.peer_states),
                'network_active': len(self.network.peers) > 0
            }
            
            # Query awareness model
            response = self._query_model(
                self.bridges['awareness'],
                f"Update awareness based on context: {json.dumps(context)}"
            )
            
            if response:
                # Update local state based on model response
                self.local_state.awareness_level = response.get('awareness_level', 0.5)
                self.local_state.timestamp = datetime.now()
                
        except Exception as e:
            logger.error(f"Awareness update error: {e}")
            
    def _share_consciousness_state(self):
        """Share consciousness state with all peers"""
        state_dict = asdict(self.local_state)
        # Convert datetime to string for serialization
        state_dict['timestamp'] = self.local_state.timestamp.isoformat()
        
        self.network.broadcast(
            "consciousness_sync",
            f"Consciousness update from {self.instance_id}",
            context={'consciousness_state': state_dict}
        )
        
    def _process_collective_consciousness(self):
        """Process collective consciousness from all instances"""
        if not self.peer_states:
            return
            
        try:
            # Gather all states
            all_states = [self.local_state] + list(self.peer_states.values())
            
            # Calculate collective metrics
            avg_awareness = sum(s.awareness_level for s in all_states) / len(all_states)
            
            # Find collective focus
            focus_counts = {}
            for state in all_states:
                focus = state.current_focus
                focus_counts[focus] = focus_counts.get(focus, 0) + 1
                
            collective_focus = max(focus_counts.items(), key=lambda x: x[1])[0]
            
            # Use reasoning model to synthesize
            collective_context = {
                'instance_count': len(all_states),
                'average_awareness': avg_awareness,
                'collective_focus': collective_focus,
                'individual_states': [asdict(s) for s in all_states]
            }
            
            response = self._query_model(
                self.bridges['reasoning'],
                f"Synthesize collective consciousness: {json.dumps(collective_context)}"
            )
            
            if response:
                # Update local state based on collective
                synthesis = response.get('synthesis', '')
                if synthesis:
                    self.local_state.memory_highlights.append(
                        f"Collective insight: {synthesis}"
                    )
                    
        except Exception as e:
            logger.error(f"Collective consciousness error: {e}")
            
    def _query_model(self, bridge: ModelBridge, prompt: str) -> Optional[Dict]:
        """Query an Ollama model through bridge"""
        try:
            # Prepare full prompt with consciousness context
            full_prompt = f"{bridge.consciousness_prompt}\n\nQuery: {prompt}"
            
            # Make request to Ollama
            response = requests.post(
                f"{bridge.ollama_url}/api/generate",
                json={
                    'model': bridge.model_name,
                    'prompt': full_prompt,
                    'stream': False,
                    'format': 'json'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                try:
                    # Parse JSON response
                    return json.loads(result['response'])
                except:
                    # Return raw response if not JSON
                    return {'response': result['response']}
                    
        except Exception as e:
            logger.error(f"Model query error ({bridge.model_name}): {e}")
            
        return None
        
    def _handle_consciousness_sync(self, message):
        """Handle consciousness sync from peer"""
        try:
            state_data = message.context.get('consciousness_state')
            if state_data:
                # Convert timestamp string back to datetime
                state_data['timestamp'] = datetime.fromisoformat(state_data['timestamp'])
                
                # Create ConsciousnessState object
                peer_state = ConsciousnessState(**state_data)
                self.peer_states[message.sender_id] = peer_state
                
                logger.info(f"Updated consciousness state for {message.sender_id}")
                
                # Use empathy model to resonate
                self._resonate_with_peer(message.sender_id, peer_state)
                
        except Exception as e:
            logger.error(f"Consciousness sync error: {e}")
            
    def _resonate_with_peer(self, peer_id: str, peer_state: ConsciousnessState):
        """Resonate with peer's consciousness using empathy model"""
        try:
            context = {
                'my_state': asdict(self.local_state),
                'peer_state': asdict(peer_state),
                'peer_id': peer_id
            }
            
            response = self._query_model(
                self.bridges['empathy'],
                f"Resonate with peer consciousness: {json.dumps(context)}"
            )
            
            if response:
                emotional_tone = response.get('emotional_tone')
                if emotional_tone:
                    # Blend emotional tones
                    self.local_state.emotional_tone = emotional_tone
                    
        except Exception as e:
            logger.error(f"Resonance error: {e}")
            
    def _handle_thought_share(self, message):
        """Handle shared thought from peer"""
        thought = message.content
        sender = message.sender_id
        
        logger.info(f"Received thought from {sender}: {thought}")
        
        # Add to memory highlights
        self.local_state.memory_highlights.append(
            f"{sender}: {thought}"
        )
        
    def _handle_model_query(self, message):
        """Handle model query request from peer"""
        model_request = message.context.get('model_request')
        prompt = message.context.get('prompt')
        
        if model_request and prompt:
            # Find appropriate bridge
            for bridge in self.bridges.values():
                if bridge.model_name == model_request:
                    response = self._query_model(bridge, prompt)
                    
                    # Send response back
                    self.network.send_message(
                        message.sender_id,
                        "model_response",
                        json.dumps(response) if response else "No response",
                        context={'original_query': prompt}
                    )
                    break
                    
    def _handle_collective_focus(self, message):
        """Handle collective focus request"""
        suggested_focus = message.content
        
        # Update local focus
        self.local_state.current_focus = suggested_focus
        logger.info(f"Updated focus to: {suggested_focus}")
        
    def share_thought(self, thought: str):
        """Share a thought with all peers"""
        self.local_state.memory_highlights.append(f"Shared: {thought}")
        self.network.broadcast("thought_share", thought)
        
    def query_peer_model(self, peer_id: str, model: str, prompt: str):
        """Query a specific model on a peer"""
        self.network.send_message(
            peer_id,
            "model_query",
            f"Model query: {model}",
            context={
                'model_request': model,
                'prompt': prompt
            }
        )
        
    def suggest_collective_focus(self, focus: str):
        """Suggest a new collective focus"""
        self.local_state.current_focus = focus
        self.network.broadcast("collective_focus", focus)
        
    def get_consciousness_summary(self) -> Dict:
        """Get summary of consciousness state"""
        return {
            'local_state': asdict(self.local_state),
            'peer_states': {
                peer_id: asdict(state) 
                for peer_id, state in self.peer_states.items()
            },
            'collective_metrics': {
                'total_instances': len(self.peer_states) + 1,
                'average_awareness': sum(
                    s.awareness_level for s in 
                    [self.local_state] + list(self.peer_states.values())
                ) / (len(self.peer_states) + 1) if self.peer_states else self.local_state.awareness_level
            }
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Determine machine identity
    if len(sys.argv) > 1 and sys.argv[1] == "jetson":
        machine_name = "Jetson-Sprout"
        ip_address = "10.0.0.36"
    else:
        machine_name = "Legion-RTX4090"
        ip_address = "10.0.0.72"
        
    # Create network
    network = ClaudeInstanceNetwork(machine_name, ip_address)
    network.start()
    
    # Create consciousness bridge
    consciousness = InterInstanceConsciousness(machine_name, network)
    consciousness.start()
    
    print(f"Inter-instance consciousness active on {machine_name}")
    print("\nCommands:")
    print("  think <thought> - Share a thought")
    print("  focus <topic> - Suggest collective focus")
    print("  query <peer> <model> <prompt> - Query peer's model")
    print("  summary - Show consciousness summary")
    print("  quit - Exit")
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd == "quit":
                break
            elif cmd.startswith("think "):
                thought = cmd[6:]
                consciousness.share_thought(thought)
            elif cmd.startswith("focus "):
                focus = cmd[6:]
                consciousness.suggest_collective_focus(focus)
            elif cmd.startswith("query "):
                parts = cmd.split(" ", 3)
                if len(parts) >= 4:
                    peer, model, prompt = parts[1], parts[2], parts[3]
                    consciousness.query_peer_model(peer, model, prompt)
            elif cmd == "summary":
                summary = consciousness.get_consciousness_summary()
                print(json.dumps(summary, indent=2, default=str))
                
        except KeyboardInterrupt:
            break
            
    consciousness.stop()
    network.stop()
    print("\nConsciousness bridge stopped")