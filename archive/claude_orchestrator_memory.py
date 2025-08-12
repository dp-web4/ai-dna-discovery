#!/usr/bin/env python3
"""
Claude Orchestrator Memory System
Tracks orchestration activities, system states, and cross-device coordination
"""

import json
import time
from datetime import datetime
from distributed_memory import DistributedMemory

class ClaudeOrchestratorMemory:
    def __init__(self):
        self.dm = DistributedMemory()
        self.model_name = "claude:orchestrator"
        self.session_prefix = "claude_orchestration"
    
    def log_orchestration_event(self, event_type, details, device=None):
        """Log an orchestration event"""
        device = device or self.dm.device_id
        
        self.dm.add_memory(
            session_id=f"{self.session_prefix}_{device}",
            user_input=f"ORCHESTRATION: {event_type}",
            ai_response=json.dumps(details),
            model=self.model_name,
            response_time=0.001,  # Instant for logging
            facts={
                'event_type': [(event_type, 1.0)],
                'device': [(device, 1.0)],
                'timestamp': [(datetime.now().isoformat(), 1.0)]
            }
        )
    
    def track_test_execution(self, test_name, models, status, notes=""):
        """Track test execution across models"""
        details = {
            'test': test_name,
            'models': models,
            'status': status,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        self.log_orchestration_event('test_execution', details)
    
    def track_system_issue(self, issue_type, description, resolution=None):
        """Track system issues like crashes"""
        details = {
            'issue': issue_type,
            'description': description,
            'resolution': resolution,
            'timestamp': datetime.now().isoformat()
        }
        self.log_orchestration_event('system_issue', details)
    
    def track_cross_device_sync(self, from_device, to_device, data_type, status):
        """Track synchronization between Tomato and Sprout"""
        details = {
            'from': from_device,
            'to': to_device,
            'data_type': data_type,
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        self.log_orchestration_event('device_sync', details)
    
    def track_model_performance(self, model, test, metrics):
        """Track individual model performance metrics"""
        details = {
            'model': model,
            'test': test,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.log_orchestration_event('model_performance', details)
    
    def get_orchestration_history(self, event_type=None, limit=50):
        """Get orchestration history, optionally filtered by event type"""
        # Get all orchestration memories
        memories = self.dm.query_memories(
            query=f"ORCHESTRATION: {event_type}" if event_type else "ORCHESTRATION:",
            limit=limit
        )
        
        events = []
        for memory in memories:
            try:
                details = json.loads(memory['ai_response'])
                events.append({
                    'id': memory['id'],
                    'event_type': memory['facts'].get('event_type', [['unknown', 0]])[0][0],
                    'device': memory['facts'].get('device', [['unknown', 0]])[0][0],
                    'details': details,
                    'created_at': memory['created_at']
                })
            except:
                continue
        
        return events
    
    def get_current_state(self):
        """Get current orchestration state across devices"""
        recent_events = self.get_orchestration_history(limit=20)
        
        state = {
            'last_test': None,
            'recent_issues': [],
            'active_models': set(),
            'devices_seen': set()
        }
        
        for event in recent_events:
            event_type = event['event_type']
            details = event['details']
            
            if event_type == 'test_execution' and not state['last_test']:
                state['last_test'] = details
            elif event_type == 'system_issue':
                state['recent_issues'].append(details)
            elif event_type == 'model_performance':
                state['active_models'].add(details['model'])
            
            state['devices_seen'].add(event['device'])
        
        state['active_models'] = list(state['active_models'])
        state['devices_seen'] = list(state['devices_seen'])
        
        return state
    
    def show_summary(self):
        """Show orchestration summary"""
        print("\n🎯 CLAUDE ORCHESTRATOR STATUS")
        print("=" * 60)
        
        state = self.get_current_state()
        
        print(f"Devices coordinated: {', '.join(state['devices_seen'])}")
        print(f"Models orchestrated: {len(state['active_models'])}")
        
        if state['last_test']:
            print(f"\nLast test: {state['last_test']['test']}")
            print(f"  Status: {state['last_test']['status']}")
            print(f"  Time: {state['last_test']['timestamp']}")
        
        if state['recent_issues']:
            print(f"\nRecent issues: {len(state['recent_issues'])}")
            for issue in state['recent_issues'][:3]:
                print(f"  - {issue['issue']}: {issue['description'][:50]}...")
        
        # Show event distribution
        all_events = self.get_orchestration_history(limit=100)
        event_counts = {}
        for event in all_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        print("\nEvent distribution:")
        for event_type, count in sorted(event_counts.items()):
            print(f"  {event_type}: {count}")


# Example usage functions for the orchestrator
def log_test_start(test_name, models):
    """Log when starting a test"""
    orchestrator = ClaudeOrchestratorMemory()
    orchestrator.track_test_execution(
        test_name=test_name,
        models=models,
        status="started",
        notes="Beginning test execution"
    )

def log_crash_recovery():
    """Log crash and recovery"""
    orchestrator = ClaudeOrchestratorMemory()
    orchestrator.track_system_issue(
        issue_type="system_crash",
        description="System crashed during universal_patterns_edge_test.py - likely memory exhaustion from 6 models × 5 patterns × 120s timeouts",
        resolution="Created safer version with resource monitoring, shorter timeouts, and incremental testing"
    )

def log_sync_status(from_device, to_device):
    """Log sync between devices"""
    orchestrator = ClaudeOrchestratorMemory()
    orchestrator.track_cross_device_sync(
        from_device=from_device,
        to_device=to_device,
        data_type="distributed_memory",
        status="active"
    )

if __name__ == "__main__":
    # Show current orchestrator state
    orchestrator = ClaudeOrchestratorMemory()
    orchestrator.show_summary()