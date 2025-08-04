#!/usr/bin/env python3
"""
MCP-CPTE Integration Example
Shows how MCP servers act as access points to external CPTE resources
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio

# Simulated MCP client (real one would use actual MCP protocol)
class MCPClient:
    """Simulated MCP client for CPTE access"""
    
    def __init__(self, server_uri: str):
        self.server_uri = server_uri
        self.connected = False
        
    async def connect(self):
        """Connect to MCP server"""
        print(f"🔌 Connecting to MCP server: {self.server_uri}")
        await asyncio.sleep(0.1)  # Simulate network
        self.connected = True
        return self
        
    async def list_tools(self) -> List[str]:
        """List available tools (expertise) from CPTE"""
        if not self.connected:
            raise Exception("Not connected to MCP server")
            
        # Simulate different expertise based on server
        if "math" in self.server_uri:
            return ["solve_ode", "integrate", "differentiate", "explain_theorem"]
        elif "legal" in self.server_uri:
            return ["review_contract", "check_compliance", "draft_clause", "explain_law"]
        elif "medical" in self.server_uri:
            return ["diagnose_symptoms", "suggest_treatment", "drug_interactions", "explain_condition"]
        else:
            return ["general_consultation"]
            
    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool (expertise area)"""
        if not self.connected:
            raise Exception("Not connected to MCP server")
            
        print(f"  → Calling {tool_name} with params: {params}")
        await asyncio.sleep(0.2)  # Simulate processing
        
        # Simulate responses based on tool
        responses = {
            "solve_ode": {
                "solution": "y = x² + C",
                "method_used": "direct integration",
                "confidence": 0.95
            },
            "review_contract": {
                "issues_found": ["Missing termination clause", "Vague IP ownership"],
                "risk_level": "medium",
                "recommendations": ["Add 30-day termination notice", "Clarify work-for-hire"]
            },
            "diagnose_symptoms": {
                "possible_conditions": ["Common cold", "Seasonal allergies"],
                "confidence": 0.7,
                "recommend_tests": ["None needed"],
                "disclaimer": "Consult healthcare provider for medical advice"
            }
        }
        
        return responses.get(tool_name, {"result": f"Processed {tool_name} request"})
        
    async def close(self):
        """Close MCP connection"""
        self.connected = False

@dataclass
class MCPEnabledCPTEMarker:
    """Knowledge marker with MCP server reference"""
    domain: str
    last_used: datetime
    confidence: float
    mcp_uri: str  # MCP server URI
    available_tools: List[str] = None
    
class MCPCPTEManager:
    """CPTE Manager with MCP integration"""
    
    def __init__(self):
        self.knowledge_markers: Dict[str, MCPEnabledCPTEMarker] = {}
        self.mcp_clients: Dict[str, MCPClient] = {}
        
        # Pre-register some external CPTE services
        self._register_external_cptes()
        
    def _register_external_cptes(self):
        """Register known external CPTE MCP servers"""
        # Math expertise
        self.knowledge_markers['differential_equations'] = MCPEnabledCPTEMarker(
            domain='differential_equations',
            last_used=datetime(2020, 5, 15),  # Long time ago
            confidence=0.1,  # Low confidence in internal knowledge
            mcp_uri='mcp://math-experts.ai/calculus'
        )
        
        # Legal expertise
        self.knowledge_markers['contract_law'] = MCPEnabledCPTEMarker(
            domain='contract_law',
            last_used=datetime(2023, 1, 1),
            confidence=0.2,
            mcp_uri='mcp://legal-ai.com/contracts'
        )
        
        # Medical expertise
        self.knowledge_markers['medical_diagnosis'] = MCPEnabledCPTEMarker(
            domain='medical_diagnosis',
            last_used=datetime(2022, 6, 1),
            confidence=0.05,  # Very low - don't self-diagnose!
            mcp_uri='mcp://medical-ai.org/general'
        )
        
    async def discover_capabilities(self, domain: str) -> List[str]:
        """Discover what tools/expertise an external CPTE offers"""
        if domain not in self.knowledge_markers:
            return []
            
        marker = self.knowledge_markers[domain]
        
        # Connect to MCP server if not already connected
        if marker.mcp_uri not in self.mcp_clients:
            client = MCPClient(marker.mcp_uri)
            await client.connect()
            self.mcp_clients[marker.mcp_uri] = client
            
        # Get available tools
        tools = await self.mcp_clients[marker.mcp_uri].list_tools()
        marker.available_tools = tools
        
        return tools
        
    async def consult_external_cpte(self, domain: str, query: str, params: Dict[str, Any] = None) -> Any:
        """Consult external CPTE via MCP"""
        print(f"\n🧠 Consulting external CPTE for '{domain}'")
        
        if domain not in self.knowledge_markers:
            return f"No external CPTE registered for '{domain}'"
            
        marker = self.knowledge_markers[domain]
        print(f"  Internal confidence: {marker.confidence:.2f} (last used: {marker.last_used.date()})")
        print(f"  MCP server: {marker.mcp_uri}")
        
        # Ensure we have a connection
        if marker.mcp_uri not in self.mcp_clients:
            client = MCPClient(marker.mcp_uri)
            await client.connect()
            self.mcp_clients[marker.mcp_uri] = client
            
        client = self.mcp_clients[marker.mcp_uri]
        
        # Discover capabilities if not cached
        if not marker.available_tools:
            marker.available_tools = await client.list_tools()
            print(f"  Available expertise: {marker.available_tools}")
            
        # Determine which tool to use based on query
        tool_to_use = self._select_tool(query, marker.available_tools)
        
        if not tool_to_use:
            return "No appropriate tool found for query"
            
        # Prepare parameters
        if params is None:
            params = {"query": query}
            
        # Call the external CPTE
        result = await client.call_tool(tool_to_use, params)
        
        return result
        
    def _select_tool(self, query: str, available_tools: List[str]) -> Optional[str]:
        """Simple tool selection based on query keywords"""
        query_lower = query.lower()
        
        # Simple keyword matching
        tool_keywords = {
            'solve_ode': ['solve', 'differential', 'ode'],
            'review_contract': ['review', 'contract', 'check'],
            'diagnose_symptoms': ['diagnose', 'symptoms', 'sick'],
            'integrate': ['integrate', 'integral'],
            'draft_clause': ['draft', 'write', 'create']
        }
        
        for tool, keywords in tool_keywords.items():
            if tool in available_tools and any(kw in query_lower for kw in keywords):
                return tool
                
        # Default to first available tool
        return available_tools[0] if available_tools else None

async def demo_mcp_cpte():
    """Demonstrate MCP-CPTE integration"""
    print("=== MCP-CPTE Integration Demo ===\n")
    
    manager = MCPCPTEManager()
    
    # Scenario 1: Math problem - no internal knowledge
    print("📐 Scenario 1: Solving differential equation")
    print("User asks: 'How do I solve dy/dx = 2x?'")
    
    # Discover capabilities
    math_tools = await manager.discover_capabilities('differential_equations')
    print(f"  Discovered tools: {math_tools}")
    
    # Consult external CPTE
    result = await manager.consult_external_cpte(
        'differential_equations',
        'solve dy/dx = 2x',
        {'equation': 'dy/dx = 2x'}
    )
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    # Scenario 2: Legal question
    print("\n⚖️ Scenario 2: Contract review")
    print("User asks: 'Can you review this employment contract?'")
    
    result = await manager.consult_external_cpte(
        'contract_law',
        'review employment contract',
        {'contract_type': 'employment', 'jurisdiction': 'US-CA'}
    )
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    # Scenario 3: Medical query (with strong disclaimer)
    print("\n🏥 Scenario 3: Medical question")
    print("User asks: 'I have a headache and runny nose, what could it be?'")
    
    result = await manager.consult_external_cpte(
        'medical_diagnosis',
        'diagnose symptoms: headache and runny nose',
        {'symptoms': ['headache', 'runny nose']}
    )
    print(f"  Result: {json.dumps(result, indent=2)}")
    
    # Show the power of MCP integration
    print("\n✨ Benefits of MCP-CPTE Integration:")
    print("  1. Standardized access to any external expertise")
    print("  2. Discovery of capabilities without hardcoding")
    print("  3. Context-aware tool selection")
    print("  4. Secure, authenticated access to expert knowledge")
    print("  5. Easy to add new expert domains - just register MCP URI")

if __name__ == "__main__":
    # Run async demo
    asyncio.run(demo_mcp_cpte())