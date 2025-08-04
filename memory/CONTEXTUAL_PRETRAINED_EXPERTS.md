# Contextual Pretrained Experts (CPTEs)

## Core Insight

HRM is 'wise' (world-aware and able to reason efficiently) but not 'learned' (small sample set, conceptual rather than precise).

This mirrors human cognition:
- **Learning**: "I learned differential equations in college"
- **Aging/Purging**: "Haven't used it since, knowledge aged and purged"
- **Marker Remains**: "Only 'diff eq' marker remains"
- **Re-learning Options**:
  - Re-learn it myself if needed
  - Ask someone who kept the knowledge current

## CPTE Definition

A **Contextual Pretrained Expert** is an agentic expert in a very specific context that can:
1. Accept contextual input
2. Interpret using specialized knowledge
3. Apply domain expertise
4. Produce contextually relevant and accurate output

**Key Distinction**: CPTEs are NOT RAG (not a database) - they are actual agents with specialized training.

## External vs Internal CPTEs

### External CPTEs
- "I ask someone else"
- Separate entities with maintained expertise
- Examples:
  - Math CPTE for differential equations
  - Legal CPTE for contract analysis
  - Medical CPTE for diagnosis

### Internal CPTEs
- "I use it often enough that I learned it myself"
- Internalized knowledge through frequent use
- Examples:
  - Basic arithmetic (everyone has this internal)
  - Programming syntax (developers internalize)
  - Native language grammar

## Web4 Integration

CPTEs map naturally to Web4 entities:
- Each CPTE has its own identity (LCT)
- T3/V3 profiles determine expertise quality
- Can form networks of expertise
- Value flows based on knowledge utility

## MCP Servers as CPTE Access Points

**Key Insight**: MCP (Model Context Protocol) servers are the natural implementation for accessing external CPTE resources.

### Architecture
```
Internal CPTE ←→ CPTE Manager ←→ MCP Client ←→ MCP Server ←→ External CPTE
                      ↓
                Knowledge Marker
                (contains MCP URI)
```

### Benefits of MCP for CPTEs
1. **Standardized Protocol**: Uniform interface for all external experts
2. **Tool Exposure**: Each CPTE exposes its capabilities as MCP tools
3. **Context Handling**: MCP handles context passing naturally
4. **Discovery**: MCP servers can advertise available expertise
5. **Authentication**: Built-in security for accessing expert knowledge

### Example Implementation
```python
# Knowledge marker with MCP reference
marker = CPTEMarker(
    domain='differential_equations',
    last_used='2025-02-04',
    confidence=0.48,
    external_ref='mcp://math-experts.com/diff-eq'
)

# MCP client connects to external CPTE
async with mcp.ClientSession(marker.external_ref) as session:
    # List available tools (expertise areas)
    tools = await session.list_tools()
    # Example: ['solve_ode', 'classify_de', 'numerical_methods']
    
    # Consult the expert
    result = await session.call_tool(
        'solve_ode',
        {'equation': 'dy/dx = 2x', 'method': 'analytical'}
    )
```

### MCP Server Example for Math CPTE
```python
# External CPTE exposed as MCP server
class MathCPTEServer:
    @mcp.tool()
    async def solve_ode(self, equation: str, method: str = 'analytical'):
        """Solve ordinary differential equation"""
        # Specialized knowledge application
        return self.ode_solver.solve(equation, method)
        
    @mcp.tool()
    async def explain_concept(self, concept: str, level: str = 'undergraduate'):
        """Explain mathematical concept at specified level"""
        return self.knowledge_base.explain(concept, level)
```

## Memory System Integration

### Knowledge Lifecycle
```
1. Initial Learning → Full Knowledge (High Memory Cost)
2. Low Usage → Knowledge Decay (Confidence Decreases)
3. Aged Out → Marker Only (Low Memory Cost)
4. Need Arises → Either:
   - Consult External CPTE
   - Re-learn (Recreate Internal CPTE)
```

### Implementation in Enhanced Memory

```python
class CPTEMarker:
    """Lightweight marker for aged-out knowledge"""
    def __init__(self, domain: str, last_used: datetime, confidence: float):
        self.domain = domain
        self.last_used = last_used
        self.confidence = confidence
        self.external_cpte_ref = None  # Link to external expert
        
class CPTEManager:
    """Manages internal/external expertise"""
    
    def query_knowledge(self, domain: str, query: str):
        # Check if we have internal CPTE
        if self.has_internal_cpte(domain):
            return self.internal_cpte[domain].process(query)
            
        # Check for marker
        marker = self.get_knowledge_marker(domain)
        if marker and marker.external_cpte_ref:
            # Delegate to external CPTE
            return self.consult_external_cpte(marker.external_cpte_ref, query)
            
        # No expertise available
        return None
        
    def maintain_expertise(self):
        """Age out unused knowledge to markers"""
        for domain, cpte in self.internal_cptes.items():
            if cpte.usage_frequency < threshold:
                # Convert to marker
                marker = CPTEMarker(
                    domain=domain,
                    last_used=cpte.last_accessed,
                    confidence=cpte.decay_confidence()
                )
                self.knowledge_markers[domain] = marker
                del self.internal_cptes[domain]
```

## Examples

### Math CPTE Scenario
```python
# Years ago: Full differential equations knowledge
internal_cpte['diff_eq'] = DiffEqExpert(confidence=0.9)

# After years of non-use: Decayed to marker
knowledge_marker['diff_eq'] = CPTEMarker(
    domain='differential_equations',
    last_used='2010-05-15',
    confidence=0.1,
    external_cpte_ref='math_expert_service'
)

# When needed: Consult external CPTE
result = cpte_manager.query_knowledge(
    'diff_eq', 
    'solve dy/dx = 2x'
)  # Routes to external expert
```

### Programming CPTE Scenario
```python
# Frequently used: Maintained as internal
internal_cpte['python'] = PythonExpert(confidence=0.95)

# Direct internal consultation
result = cpte_manager.query_knowledge(
    'python',
    'how to async/await'
)  # Answered immediately from internal knowledge
```

## Benefits

1. **Memory Efficiency**: Only keep active knowledge in full form
2. **Expertise Networks**: Leverage distributed specialized knowledge
3. **Natural Learning**: Mirrors human knowledge acquisition/decay
4. **Scalability**: Can have thousands of markers, few full CPTEs
5. **Web4 Native**: Each CPTE is a value-creating entity

## Integration with Current Architecture

1. **Memory Layers**:
   - Markers in Semantic Memory (lightweight)
   - Active CPTEs in Consciousness Layer
   - Usage patterns in Episodic Memory

2. **Confidence Scoring**:
   - Internal CPTE confidence based on usage frequency
   - External CPTE confidence based on reputation/T3V3
   - Marker confidence decays over time

3. **Distributed Sync**:
   - Share markers across devices
   - External CPTE references synchronized
   - Usage patterns inform which CPTEs to internalize

## Next Steps

1. Design CPTE marker schema for memory system
2. Create CPTE manager interface
3. Implement aging/decay algorithms
4. Build external CPTE consultation protocol
5. Integrate with Web4 entity framework