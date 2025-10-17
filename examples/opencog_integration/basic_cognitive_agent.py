#!/usr/bin/env python3
"""
Basic OpenCog Cognitive Agent Example for OpenManus-RL.

This example demonstrates how to use the OpenCog systems integration
to create a cognitive agent with symbolic reasoning capabilities.
"""

import logging
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from openmanus_rl.opencog_systems import (
    CognitiveAgent, AttentionMode, AtomSpaceManager, 
    OpenCogReasoningEngine, KnowledgeGraph
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_basic_cognitive_agent():
    """Demonstrate basic cognitive agent functionality."""
    
    print("=" * 60)
    print("OpenCog Cognitive Agent Demonstration")
    print("=" * 60)
    
    # Create a cognitive agent
    agent = CognitiveAgent("demo_agent")
    
    print(f"\n1. Agent initialized: {agent.agent_name}")
    print(f"   Initial AtomSpace size: {agent.atomspace.size()}")
    print(f"   Initial state: {agent.state.value}")
    
    # Demonstrate perception
    print(f"\n2. Perception Phase")
    observations = {
        "environment": "laboratory",
        "objects": ["computer", "microscope", "samples"],
        "task": "analyze_samples",
        "urgency": "high"
    }
    
    agent.perceive(observations)
    print(f"   Perceived: {observations}")
    print(f"   AtomSpace size after perception: {agent.atomspace.size()}")
    print(f"   Active concepts: {len(agent.memory.active_concepts)}")
    
    # Demonstrate reasoning
    print(f"\n3. Reasoning Phase")
    reasoning_result = agent.reason("What should I do with the samples?")
    print(f"   Reasoning query: 'What should I do with the samples?'")
    print(f"   Reasoning confidence: {reasoning_result.confidence:.3f}")
    print(f"   Reasoning steps: {len(reasoning_result.reasoning_path)}")
    
    if reasoning_result.reasoning_path:
        print("   Reasoning path:")
        for i, step in enumerate(reasoning_result.reasoning_path[:3]):  # Show first 3 steps
            print(f"     {i+1}. {step}")
    
    # Demonstrate planning
    print(f"\n4. Planning Phase")
    goal = "analyze the samples efficiently"
    plan = agent.plan(goal, {"equipment": ["microscope", "computer"]})
    
    print(f"   Goal: {goal}")
    print(f"   Generated plan with {len(plan)} actions:")
    
    for i, action in enumerate(plan):
        print(f"     {i+1}. {action.action_type} (confidence: {action.confidence:.3f})")
        print(f"        Expected outcome: {action.expected_outcome}")
    
    # Demonstrate action execution
    print(f"\n5. Action Execution Phase")
    if plan:
        action_result = agent.act(plan[0])
        print(f"   Executed: {plan[0].action_type}")
        print(f"   Action result: {action_result}")
        print(f"   Success rate: {agent.success_rate:.3f}")
    
    # Demonstrate learning
    print(f"\n6. Learning Phase")
    feedback = {
        "reward": 0.8,
        "success": True,
        "correction": None,
        "outcome": "samples analyzed successfully"
    }
    
    agent.learn(feedback)
    print(f"   Provided feedback: {feedback}")
    print(f"   Updated success rate: {agent.success_rate:.3f}")
    
    # Demonstrate full cognitive cycle
    print(f"\n7. Complete Cognitive Cycle")
    new_observations = {
        "environment": "laboratory",
        "new_samples": "bacterial_culture",
        "equipment_status": "ready",
        "previous_results": "positive"
    }
    
    cycle_result = agent.cognitive_cycle(new_observations, "process new bacterial samples")
    
    print(f"   Cycle {cycle_result['cycle_number']} completed in {cycle_result['cycle_time']:.3f}s")
    print(f"   Current attention focus: {cycle_result.get('attention_focus', 'None')}")
    print(f"   Active concepts: {cycle_result['active_concepts']}")
    
    # Show cognitive state
    print(f"\n8. Current Cognitive State")
    state = agent.get_cognitive_state()
    
    print(f"   State: {state['state']}")
    print(f"   Attention mode: {state['attention_mode']}")
    print(f"   Total cycles: {state['cycle_count']}")
    print(f"   AtomSpace size: {state['atomspace_size']}")
    print(f"   Recent experiences: {state['recent_experiences']}")
    
    return agent


def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph capabilities."""
    
    print("\n" + "=" * 60)
    print("OpenCog Knowledge Graph Demonstration")
    print("=" * 60)
    
    # Create AtomSpace and Knowledge Graph
    atomspace = AtomSpaceManager()
    kg = KnowledgeGraph(atomspace)
    
    print(f"\n1. Knowledge Graph initialized")
    print(f"   Initial entities: {len(kg.entities)}")
    print(f"   Initial relationships: {len(kg.relationships)}")
    
    # Add domain-specific knowledge
    print(f"\n2. Adding Domain Knowledge")
    
    # Add laboratory entities
    lab_id = kg.add_entity("Laboratory", {"type": "location", "purpose": "research"})
    computer_id = kg.add_entity("Computer", {"type": "equipment", "function": "data_processing"})
    microscope_id = kg.add_entity("Microscope", {"type": "equipment", "function": "observation"})
    sample_id = kg.add_entity("Sample", {"type": "material", "state": "unknown"})
    
    # Add agent
    agent_id = kg.add_entity("ResearchAgent", {"type": "agent", "role": "researcher"})
    
    print(f"   Added 5 entities")
    
    # Add relationships
    from openmanus_rl.opencog_systems.knowledge_representation import RelationType
    
    # Location relationships
    kg.add_relationship(computer_id, lab_id, RelationType.SPATIAL, properties={"relation": "located_in"})
    kg.add_relationship(microscope_id, lab_id, RelationType.SPATIAL, properties={"relation": "located_in"})
    kg.add_relationship(sample_id, lab_id, RelationType.SPATIAL, properties={"relation": "located_in"})
    
    # Agent relationships  
    kg.add_relationship(agent_id, computer_id, RelationType.ASSOCIATION, properties={"relation": "uses"})
    kg.add_relationship(agent_id, microscope_id, RelationType.ASSOCIATION, properties={"relation": "uses"})
    
    # Task relationships
    analyze_task_id = kg.add_entity("AnalyzeTask", {"type": "task", "priority": "high"})
    kg.add_relationship(agent_id, analyze_task_id, RelationType.ASSOCIATION, properties={"relation": "performs"})
    kg.add_relationship(analyze_task_id, sample_id, RelationType.ASSOCIATION, properties={"relation": "targets"})
    
    print(f"   Added {len(kg.relationships)} relationships")
    
    # Query the knowledge graph
    print(f"\n3. Querying Knowledge Graph")
    
    # Find entities related to the agent
    related = kg.find_related_entities(agent_id, max_distance=2)
    print(f"   Entities related to ResearchAgent:")
    for entity, distance in related[:5]:  # Show top 5
        print(f"     - {entity.name} (distance: {distance:.2f})")
    
    # Get neighborhood of laboratory
    neighborhood = kg.get_entity_neighborhood(lab_id, radius=1)
    print(f"   Laboratory neighborhood contains {len(neighborhood['entities'])} entities")
    
    # Infer new relationships
    print(f"\n4. Knowledge Inference")
    new_rels = kg.infer_relationships()
    print(f"   Inferred {len(new_rels)} new relationships")
    
    # Show statistics
    stats = kg.get_statistics()
    print(f"\n5. Knowledge Graph Statistics")
    print(f"   Total entities: {stats['total_entities']}")
    print(f"   Total relationships: {stats['total_relationships']}")
    print(f"   Entity types: {list(stats['entity_types'].keys())}")
    print(f"   Relation types: {list(stats['relation_types'].keys())}")
    
    return kg


def demonstrate_reasoning_engine():
    """Demonstrate reasoning engine capabilities."""
    
    print("\n" + "=" * 60)
    print("OpenCog Reasoning Engine Demonstration")
    print("=" * 60)
    
    # Create AtomSpace and Reasoning Engine
    atomspace = AtomSpaceManager()
    reasoning_engine = OpenCogReasoningEngine(atomspace)
    
    print(f"\n1. Reasoning Engine initialized")
    print(f"   Number of rules: {len(reasoning_engine.rules)}")
    
    # Add some domain knowledge
    print(f"\n2. Adding Domain Knowledge")
    
    # Create basic facts about the laboratory domain
    lab_atom = atomspace.create_concept_node("laboratory")
    agent_atom = atomspace.create_concept_node("agent")
    sample_atom = atomspace.create_concept_node("sample")
    analysis_atom = atomspace.create_concept_node("analysis")
    
    # Create relationships
    located_in_pred = atomspace.create_predicate_node("located_in")
    can_perform_pred = atomspace.create_predicate_node("can_perform")
    requires_pred = atomspace.create_predicate_node("requires")
    
    # Agent is located in laboratory
    atomspace.create_evaluation_link(located_in_pred, [agent_atom, lab_atom], 0.9)
    
    # Agent can perform analysis
    atomspace.create_evaluation_link(can_perform_pred, [agent_atom, analysis_atom], 0.8)
    
    # Analysis requires sample
    atomspace.create_evaluation_link(requires_pred, [analysis_atom, sample_atom], 0.95)
    
    print(f"   Added domain facts")
    print(f"   AtomSpace size: {atomspace.size()}")
    
    # Demonstrate forward chaining
    print(f"\n3. Forward Chaining Reasoning")
    derived_atoms = reasoning_engine.forward_chaining(max_iterations=3)
    print(f"   Derived {len(derived_atoms)} new atoms through forward chaining")
    
    # Demonstrate backward chaining
    print(f"\n4. Backward Chaining Reasoning")
    
    # Query: Can the agent perform analysis?
    goal = {
        "type": "evaluation",
        "predicate": "can_perform",
        "args": ["agent", "analysis"]
    }
    
    result = reasoning_engine.backward_chaining(goal, max_depth=3)
    print(f"   Query: Can agent perform analysis?")
    print(f"   Result confidence: {result.confidence:.3f}")
    print(f"   Reasoning steps: {len(result.reasoning_path)}")
    
    if result.reasoning_path:
        print("   Reasoning path:")
        for step in result.reasoning_path[:3]:
            print(f"     - {step}")
    
    # Demonstrate action reasoning
    print(f"\n5. Action Reasoning")
    action_result = reasoning_engine.reason_about_action(
        "perform_analysis",
        {"location": "laboratory", "equipment": "microscope"}
    )
    
    print(f"   Action: perform_analysis")
    print(f"   Context: laboratory with microscope")
    print(f"   Reasoning confidence: {action_result.confidence:.3f}")
    
    # Show explanation
    if action_result.conclusion:
        explanation = reasoning_engine.explain_reasoning(action_result)
        print(f"\n6. Reasoning Explanation")
        print(explanation)
    
    # Get statistics
    stats = reasoning_engine.get_reasoning_statistics()
    print(f"\n7. Reasoning Statistics")
    print(f"   Total rules: {stats['total_rules']}")
    print(f"   AtomSpace size: {stats['atomspace_size']}")
    print(f"   Reasoning history: {stats['reasoning_history_length']}")
    print(f"   Successful reasonings: {stats['successful_reasonings']}")
    print(f"   Average confidence: {stats['average_confidence']:.3f}")
    
    return reasoning_engine


def main():
    """Main demonstration function."""
    
    print("OpenManus-RL OpenCog Systems Integration Demo")
    print("=" * 60)
    
    try:
        # Demonstrate each component
        agent = demonstrate_basic_cognitive_agent()
        kg = demonstrate_knowledge_graph()
        reasoning_engine = demonstrate_reasoning_engine()
        
        print("\n" + "=" * 60)
        print("Integration Demonstration Complete!")
        print("=" * 60)
        
        print(f"\nSummary:")
        print(f"- Cognitive Agent: {agent.cycle_count} cycles, {agent.success_rate:.3f} success rate")
        print(f"- Knowledge Graph: {kg.get_statistics()['total_entities']} entities, {kg.get_statistics()['total_relationships']} relationships")
        print(f"- Reasoning Engine: {len(reasoning_engine.reasoning_history)} reasoning operations")
        
        # Save cognitive state for inspection
        state_file = "/tmp/cognitive_agent_state.json"
        try:
            with open(state_file, 'w') as f:
                json.dump(agent.get_cognitive_state(), f, indent=2, default=str)
            print(f"\nCognitive agent state saved to: {state_file}")
        except Exception as e:
            print(f"\nCould not save state: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)