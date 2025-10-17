# OpenCog Systems Integration for OpenManus-RL

This document describes the OpenCog systems integration in OpenManus-RL, providing symbolic reasoning and cognitive architecture capabilities for reinforcement learning agents.

## Overview

The OpenCog integration adds advanced symbolic AI capabilities to OpenManus-RL, enabling agents to:

- **Symbolic Reasoning**: Perform logical inference and pattern matching
- **Knowledge Representation**: Build and query semantic knowledge graphs  
- **Cognitive Architecture**: Implement complete cognitive processing cycles
- **Hybrid AI**: Combine symbolic reasoning with neural learning

## Components

### 1. AtomSpace Integration (`atomspace_integration.py`)

The AtomSpace is OpenCog's hypergraph knowledge representation system.

```python
from openmanus_rl.opencog_systems import AtomSpaceManager, AtomType

# Create an AtomSpace
atomspace = AtomSpaceManager()

# Add knowledge
concept = atomspace.create_concept_node("agent")
predicate = atomspace.create_predicate_node("can_perform")
action = atomspace.create_concept_node("analysis")

# Create relationships
evaluation = atomspace.create_evaluation_link(
    predicate, [concept, action], truth_value=0.8
)
```

**Key Features:**
- Hypergraph knowledge representation
- Truth values and confidence tracking
- Pattern matching and querying
- Import/export capabilities
- Memory management

### 2. Reasoning Engine (`reasoning_engine.py`)

Provides symbolic inference capabilities using forward/backward chaining.

```python
from openmanus_rl.opencog_systems import OpenCogReasoningEngine

# Create reasoning engine
reasoning_engine = OpenCogReasoningEngine(atomspace)

# Perform backward chaining
goal = {"type": "evaluation", "predicate": "can_perform", "args": ["agent", "task"]}
result = reasoning_engine.backward_chaining(goal)

print(f"Confidence: {result.confidence}")
print(f"Reasoning path: {result.reasoning_path}")
```

**Reasoning Modes:**
- Forward chaining (data-driven)
- Backward chaining (goal-driven)  
- Probabilistic reasoning
- Action consequence reasoning

### 3. Pattern Matcher (`pattern_matcher.py`)

Advanced pattern matching for knowledge retrieval and similarity detection.

```python
from openmanus_rl.opencog_systems import OpenCogPatternMatcher, PatternQuery, MatchType

# Create pattern matcher
matcher = OpenCogPatternMatcher(atomspace)

# Define pattern query
query = PatternQuery({
    "type": "evaluation",
    "predicate": "located_at", 
    "args": ["$entity", "laboratory"]
})

# Find matches
matches = matcher.match_pattern(query, MatchType.FUZZY)
```

**Match Types:**
- Exact matching
- Fuzzy matching (similarity-based)
- Structural matching (graph topology)
- Semantic matching (meaning-based)

### 4. Cognitive Architecture (`cognitive_architecture.py`)

Complete cognitive agent implementation with perception, reasoning, planning, and learning.

```python
from openmanus_rl.opencog_systems import CognitiveAgent

# Create cognitive agent
agent = CognitiveAgent("research_agent")

# Cognitive cycle
observations = {"environment": "lab", "task": "analyze_samples"}
result = agent.cognitive_cycle(observations, goal="complete analysis")

print(f"Cycle {result['cycle_number']} completed")
print(f"Action result: {result['action_result']}")
```

**Cognitive Processes:**
- Perception (observation processing)
- Reasoning (inference and analysis)
- Planning (goal-directed action selection)
- Action execution
- Learning from feedback

### 5. Knowledge Graph (`knowledge_representation.py`)

High-level knowledge representation and semantic networks.

```python
from openmanus_rl.opencog_systems import KnowledgeGraph, RelationType

# Create knowledge graph
kg = KnowledgeGraph(atomspace)

# Add entities and relationships
lab_id = kg.add_entity("Laboratory", {"type": "location"})
agent_id = kg.add_entity("Agent", {"type": "actor"})

kg.add_relationship(agent_id, lab_id, RelationType.SPATIAL, 
                   properties={"relation": "located_in"})

# Query relationships
related = kg.find_related_entities(agent_id, max_distance=2)
```

**Knowledge Operations:**
- Entity and relationship management
- Ontology building
- Path finding and traversal
- Similarity detection
- Knowledge inference

## Configuration

Configure OpenCog systems using the configuration manager:

```python
from openmanus_rl.opencog_systems import load_config, update_config

# Load configuration
config = load_config("path/to/config.yaml")

# Update parameters
update_config(
    reasoning_max_depth=5,
    cognitive_learning_rate=0.1,
    enable_fuzzy_matching=True
)
```

**Configuration Categories:**
- AtomSpace settings (memory, persistence)
- Reasoning parameters (depth, iterations, thresholds)
- Cognitive architecture (attention, memory, learning)
- Pattern matching (similarity thresholds, match types)
- Knowledge graph (entities, relationships, inference)

## Integration with OpenManus-RL

### Agent Enhancement

Integrate OpenCog capabilities into existing OpenManus agents:

```python
from openmanus_rl.llm_agent.openmanus import OpenManusAgent
from openmanus_rl.opencog_systems import CognitiveAgent

class CognitiveOpenManusAgent(OpenManusAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cognitive_agent = CognitiveAgent("enhanced_agent")
    
    def run_llm_loop(self, gen_batch, output_dir=None, global_steps=0):
        # Extract observations from batch
        observations = self.extract_observations(gen_batch)
        
        # Run cognitive processing
        cognitive_result = self.cognitive_agent.cognitive_cycle(observations)
        
        # Use cognitive insights to enhance LLM processing
        enhanced_batch = self.enhance_with_cognitive_insights(
            gen_batch, cognitive_result
        )
        
        # Continue with original processing
        return super().run_llm_loop(enhanced_batch, output_dir, global_steps)
```

### Memory Integration

Enhance memory systems with symbolic knowledge:

```python
from openmanus_rl.memory.memory import SimpleMemory
from openmanus_rl.opencog_systems import KnowledgeGraph

class CognitiveMemory(SimpleMemory):
    def __init__(self):
        super().__init__()
        self.atomspace = AtomSpaceManager()
        self.knowledge_graph = KnowledgeGraph(self.atomspace)
    
    def store(self, record):
        super().store(record)
        
        # Also store in knowledge graph
        for env_idx, env_record in enumerate(record):
            self.add_to_knowledge_graph(env_record, env_idx)
    
    def fetch_with_reasoning(self, query, history_length):
        # Use pattern matching for intelligent retrieval
        # ... implementation
```

### Reward Shaping

Use symbolic reasoning for reward shaping:

```python
from openmanus_rl.opencog_systems import OpenCogReasoningEngine

class CognitiveRewardShaper:
    def __init__(self, atomspace):
        self.reasoning_engine = OpenCogReasoningEngine(atomspace)
    
    def shape_reward(self, state, action, reward, next_state):
        # Reason about action appropriateness
        action_result = self.reasoning_engine.reason_about_action(
            action, {"state": state, "next_state": next_state}
        )
        
        # Adjust reward based on symbolic reasoning
        reasoning_bonus = action_result.confidence * 0.1
        return reward + reasoning_bonus
```

## Examples

### Basic Usage

See `examples/opencog_integration/basic_cognitive_agent.py` for a complete demonstration:

```bash
cd /path/to/OpenManus-RL
python examples/opencog_integration/basic_cognitive_agent.py
```

### Custom Cognitive Agent

```python
from openmanus_rl.opencog_systems import CognitiveAgent, AttentionMode

# Create specialized agent
agent = CognitiveAgent("scientific_researcher")
agent.set_attention_mode(AttentionMode.GOAL_DIRECTED)

# Add domain knowledge
agent.atomspace.create_concept_node("hypothesis")
agent.atomspace.create_concept_node("experiment")
agent.atomspace.create_concept_node("evidence")

# Create relationships
hypothesis_atom = agent.atomspace.find_atoms(name="hypothesis")[0]
experiment_atom = agent.atomspace.find_atoms(name="experiment")[0]
tests_pred = agent.atomspace.create_predicate_node("tests")

agent.atomspace.create_evaluation_link(
    tests_pred, [experiment_atom, hypothesis_atom], 0.9
)

# Run research cycle
observations = {
    "lab_equipment": ["microscope", "computer"],
    "samples": ["specimen_a", "specimen_b"],
    "hypothesis_status": "untested"
}

result = agent.cognitive_cycle(observations, "test hypothesis with experiments")
```

## Testing

Run tests for OpenCog integration:

```bash
# Test AtomSpace functionality
python -m unittest test.opencog_systems.test_atomspace

# Test cognitive agent
python -m unittest test.opencog_systems.test_cognitive_agent

# Run all OpenCog tests
python -m unittest discover test/opencog_systems/
```

## Performance Considerations

### Memory Management
- AtomSpace size limits (default: 100,000 atoms)
- Garbage collection thresholds
- Working memory constraints (20 active concepts)
- Experience history limits (100 recent experiences)

### Reasoning Efficiency
- Max reasoning depth (default: 5)
- Max iterations for forward chaining (default: 10)
- Pattern matching result limits (default: 100)
- Rule application caching

### Cognitive Cycles
- Max cycle time limits (1 second default)
- Parallel processing options
- Attention allocation strategies
- Memory consolidation thresholds

## Advanced Features

### Real OpenCog Integration

To use the real OpenCog system (when available):

```python
from openmanus_rl.opencog_systems import update_config

update_config(enable_real_opencog=True)
```

### Custom Reasoning Rules

Add domain-specific reasoning rules:

```python
from openmanus_rl.opencog_systems import ReasoningRule

# Define custom rule
rule = ReasoningRule(
    name="experimental_evidence",
    premises=[
        {"type": "evaluation", "predicate": "tests", "args": ["$experiment", "$hypothesis"]},
        {"type": "evaluation", "predicate": "supports", "args": ["$result", "$hypothesis"]}
    ],
    conclusion={"type": "evaluation", "predicate": "evidence_for", "args": ["$hypothesis"]},
    confidence=0.8
)

reasoning_engine.add_rule(rule)
```

### Custom Pattern Constraints

Add constraints to pattern variables:

```python
from openmanus_rl.opencog_systems import PatternQuery

query = PatternQuery({
    "type": "evaluation",
    "predicate": "located_at",
    "args": ["$agent", "$location"]
})

# Add constraint: agent must be of type "researcher"
query.add_constraint("$agent", lambda atom: "researcher" in atom.name.lower())
query.set_variable_type("$location", AtomType.CONCEPT_NODE)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes the project root
2. **Memory Issues**: Adjust AtomSpace size limits in configuration
3. **Slow Reasoning**: Reduce max depth or enable result caching
4. **Pattern Matching Timeout**: Increase match thresholds or limit results

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from openmanus_rl.opencog_systems import update_config
update_config(log_level="DEBUG", enable_logging=True)
```

### Performance Monitoring

```python
from openmanus_rl.opencog_systems import CognitiveAgent

agent = CognitiveAgent("monitored_agent")

# Get performance statistics
stats = agent.get_cognitive_state()
reasoning_stats = agent.reasoning_engine.get_reasoning_statistics()
kg_stats = agent.knowledge_graph.get_statistics()

print(f"Success rate: {stats['success_rate']}")
print(f"Average reasoning confidence: {reasoning_stats['average_confidence']}")
print(f"Knowledge graph size: {kg_stats['total_entities']} entities")
```

## Future Enhancements

- Integration with external OpenCog installations
- Distributed reasoning across multiple agents
- Learning of new reasoning rules from experience
- Integration with neural-symbolic architectures
- Real-time knowledge graph updates from environment
- Multi-modal knowledge representation (text, images, sensors)

## References

- [OpenCog Framework](http://opencog.org/)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Pattern Matcher Guide](https://wiki.opencog.org/w/Pattern_matcher)
- [PLN (Probabilistic Logic Networks)](https://wiki.opencog.org/w/PLN)