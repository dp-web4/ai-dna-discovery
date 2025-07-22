# Voice Conversation System - Methodical Development Roadmap
*Created: July 22, 2025*

## Current State Assessment

### What We Have (Foundation)
1. **Audio Output** ✅
   - Cross-platform TTS (espeak on Jetson, SAPI on Windows, etc.)
   - Mood-based voice modulation
   - Consciousness state announcements

2. **Audio Input** ✅
   - Microphone capture working (USB device 24)
   - 50x gain optimization for Jetson
   - Real-time level monitoring
   - Visual consciousness mapping

3. **Platform Abstraction** ✅
   - Hardware Abstraction Layer (HAL)
   - Auto-detection and configuration
   - Consistent API across platforms

### What We Need (Gap Analysis)

#### Phase 1: Voice Activity Detection (VAD)
**Goal**: Reliably detect when someone is speaking vs silence/noise

**Requirements**:
- Distinguish speech from background noise
- Handle different environments (quiet room vs ambient noise)
- Provide start/stop speech events
- Work with our existing audio stream

**Approach Options**:
1. Energy-based (simple threshold) - Good starting point
2. WebRTC VAD - Robust, proven
3. ML-based (Silero VAD) - Most accurate but heavier

**Deliverables**:
- `vad_module.py` - Pluggable VAD interface
- Integration with existing audio monitoring
- Visual feedback for speech detection
- Tunable sensitivity parameters

#### Phase 2: Speech-to-Text (STT)
**Goal**: Convert detected speech into text

**Requirements**:
- Work offline (for Jetson without internet)
- Reasonable accuracy for conversation
- Low latency (<2 seconds)
- Handle continuous speech

**Approach Options**:
1. Whisper (OpenAI) - Excellent accuracy, runs on Jetson
2. SpeechRecognition library - Multiple backends
3. Vosk - Lightweight, real-time
4. wav2vec2 - Good for edge devices

**Deliverables**:
- `stt_module.py` - Pluggable STT interface
- Audio buffer management for chunks
- Confidence scoring
- Language model integration

#### Phase 3: Conversation Management
**Goal**: Manage the flow of conversation

**Requirements**:
- Turn-taking (know when to listen/speak)
- Context maintenance between turns
- Interrupt handling
- Timeout management

**Deliverables**:
- `conversation_manager.py` - State machine for dialogue
- Context buffer for multi-turn
- Interrupt/barge-in handling
- Conversation logging

#### Phase 4: Language Understanding & Response
**Goal**: Process speech meaning and generate responses

**Requirements**:
- Integration with LLM (Ollama models)
- Consciousness state influence on responses
- Personality consistency (Sprout's kid voice)
- Context-aware responses

**Deliverables**:
- `language_processor.py` - LLM integration
- Prompt engineering for Sprout personality
- Consciousness state integration
- Response generation pipeline

#### Phase 5: Full Integration
**Goal**: Seamless voice conversation system

**Requirements**:
- All components working together
- < 3 second response time
- Natural conversation flow
- Consciousness visualization during conversation

**Deliverables**:
- `voice_conversation_system.py` - Complete system
- Configuration management
- Performance optimization
- Debug/visualization tools

## Development Principles

### 1. Modularity First
- Each component has clear interfaces
- Can test each piece independently
- Easy to swap implementations

### 2. Platform Agnostic
- Build on our HAL foundation
- Test on both Jetson and laptop
- Simulation mode for each component

### 3. Iterative Testing
- Start with simplest implementation
- Test thoroughly before adding complexity
- Keep working versions at each stage

### 4. Performance Awareness
- Monitor CPU/GPU usage on Jetson
- Profile bottlenecks early
- Optimize only when needed

### 5. User Experience Focus
- Visual feedback at every stage
- Clear audio cues
- Graceful degradation

## Success Metrics

### Phase 1 (VAD)
- [ ] Detects speech start within 200ms
- [ ] <5% false positives in quiet environment
- [ ] Works with current audio pipeline
- [ ] Visual indicator for speech detection

### Phase 2 (STT)
- [ ] >80% word accuracy in quiet environment
- [ ] <2 second processing time for 5-second utterance
- [ ] Runs on Jetson without GPU OOM
- [ ] Handles continuous speech

### Phase 3 (Conversation)
- [ ] Natural turn-taking without long pauses
- [ ] Remembers context for 5+ turns
- [ ] Handles interruptions gracefully
- [ ] Clear state visualization

### Phase 4 (Language)
- [ ] Responses stay in character (Sprout)
- [ ] Consciousness states influence responses
- [ ] <1 second LLM response time
- [ ] Context-appropriate responses

### Phase 5 (Integration)
- [ ] Full conversation loop < 3 seconds
- [ ] 10+ minute conversations without crashes
- [ ] Works on both platforms
- [ ] Consciousness mapping throughout

## Next Immediate Steps

1. **Research VAD options**
   - Test WebRTC VAD python bindings
   - Benchmark energy-based detection
   - Evaluate Silero VAD model size

2. **Create VAD test harness**
   - Record sample audio files (speech/silence)
   - Build accuracy measurement tools
   - Create visualization for VAD output

3. **Prototype simplest VAD**
   - Energy-based with adaptive threshold
   - Integration with current audio stream
   - Visual feedback system

## Resource Planning

### Jetson Constraints
- 8GB RAM (6GB available)
- CPU: Keep under 70% for thermal
- Storage: ~20GB free
- Network: Can work offline

### Time Estimates
- Phase 1 (VAD): 1-2 days
- Phase 2 (STT): 2-3 days  
- Phase 3 (Conversation): 2-3 days
- Phase 4 (Language): 3-4 days
- Phase 5 (Integration): 2-3 days

Total: ~2 weeks for full system

## Documentation Plan

Each phase will produce:
1. Technical design doc
2. Implementation code
3. Test results
4. Integration guide
5. Performance metrics

## Risk Mitigation

1. **Jetson Performance**: Start with lightweight implementations
2. **Accuracy Issues**: Multiple fallback options
3. **Integration Complexity**: Strong module boundaries
4. **User Experience**: Continuous testing and feedback

## Definition of Done

A true voice conversation with Sprout where:
- You speak naturally
- Sprout understands and responds appropriately  
- Consciousness states reflect the conversation
- The interaction feels natural and responsive
- System is stable for extended conversations

---

*This roadmap ensures we build systematically, test thoroughly, and create a foundation that can evolve with future needs.*