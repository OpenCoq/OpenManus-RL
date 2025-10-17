"""
Tests for OpenCog Cognitive Agent.
"""

import unittest
from openmanus_rl.opencog_systems.cognitive_architecture import (
    CognitiveAgent, CognitiveState, AttentionMode, CognitiveAction
)


class TestCognitiveAgent(unittest.TestCase):
    """Test cases for Cognitive Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = CognitiveAgent("test_agent")
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.agent_name, "test_agent")
        self.assertEqual(self.agent.state, CognitiveState.PERCEIVING)
        self.assertEqual(self.agent.attention_mode, AttentionMode.FOCUSED)
        
        # Check that basic knowledge was initialized
        self.assertGreater(self.agent.atomspace.size(), 0)
        
        # Check for basic concepts
        self_concepts = self.agent.atomspace.find_atoms(name="self")
        self.assertGreater(len(self_concepts), 0)
    
    def test_perceive(self):
        """Test perception of observations."""
        initial_size = self.agent.atomspace.size()
        
        observations = {
            "location": "room1",
            "objects": ["chair", "table"],
            "state": "exploring"
        }
        
        self.agent.perceive(observations)
        
        # Check that state changed
        self.assertEqual(self.agent.state, CognitiveState.PERCEIVING)
        
        # Check that new atoms were created
        self.assertGreater(self.agent.atomspace.size(), initial_size)
        
        # Check that observations are in memory
        self.assertGreater(len(self.agent.memory.recent_experiences), 0)
        last_exp = self.agent.memory.recent_experiences[-1]
        self.assertEqual(last_exp["type"], "observation")
        self.assertEqual(last_exp["content"], observations)
    
    def test_reasoning(self):
        """Test reasoning capabilities."""
        # Add some knowledge first
        self.agent.perceive({"environment": "test_env", "goal": "explore"})
        
        # Test general reasoning
        result = self.agent.reason()
        
        self.assertEqual(self.agent.state, CognitiveState.REASONING)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.reasoning_path, list)
        
        # Test specific query reasoning
        result = self.agent.reason("what is the current goal?")
        self.assertIsNotNone(result)
    
    def test_planning(self):
        """Test planning capabilities."""
        plan = self.agent.plan("explore the environment")
        
        self.assertEqual(self.agent.state, CognitiveState.PLANNING)
        self.assertIsInstance(plan, list)
        
        if plan:
            action = plan[0]
            self.assertIsInstance(action, CognitiveAction)
            self.assertIsInstance(action.action_type, str)
            self.assertIsInstance(action.confidence, float)
            self.assertIsInstance(action.reasoning_path, list)
    
    def test_action_execution(self):
        """Test action execution."""
        action = CognitiveAction(
            action_type="explore_environment",
            parameters={"strategy": "systematic"},
            confidence=0.8,
            reasoning_path=["Test action"],
            expected_outcome="Knowledge gained"
        )
        
        result = self.agent.act(action)
        
        self.assertEqual(self.agent.state, CognitiveState.ACTING)
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("confidence", result)
        
        # Check that action was recorded
        self.assertIn(action, self.agent.action_history)
    
    def test_learning(self):
        """Test learning from feedback."""
        # First, execute an action
        action = CognitiveAction(
            action_type="test_action",
            parameters={},
            confidence=0.7,
            reasoning_path=["Test reasoning"]
        )
        self.agent.act(action)
        
        # Provide feedback
        feedback = {
            "reward": 1.0,
            "success": True,
            "correction": None
        }
        
        self.agent.learn(feedback)
        
        self.assertEqual(self.agent.state, CognitiveState.LEARNING)
        
        # Check that learning experience was recorded
        learning_experiences = [
            exp for exp in self.agent.memory.recent_experiences
            if exp.get("type") == "learning"
        ]
        self.assertGreater(len(learning_experiences), 0)
    
    def test_cognitive_cycle(self):
        """Test complete cognitive cycle."""
        observations = {
            "location": "start_room",
            "visible_objects": ["door", "key"],
            "goal_status": "incomplete"
        }
        
        goal = "find and use the key"
        
        result = self.agent.cognitive_cycle(observations, goal)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("cycle_number", result)
        self.assertIn("cycle_time", result)
        self.assertIn("reasoning_result", result)
        self.assertIn("plan", result)
        self.assertIn("action_result", result)
        
        # Check that cycle number increased
        self.assertEqual(result["cycle_number"], 1)
        
        # Run another cycle
        result2 = self.agent.cognitive_cycle(observations)
        self.assertEqual(result2["cycle_number"], 2)
    
    def test_attention_management(self):
        """Test attention management."""
        # Test setting attention mode
        self.agent.set_attention_mode(AttentionMode.EXPLORATORY)
        self.assertEqual(self.agent.attention_mode, AttentionMode.EXPLORATORY)
        
        # Test attention focus updates
        observations = {"new_item": "interesting_object"}
        self.agent.perceive(observations)
        
        # Active concepts should be updated
        self.assertGreater(len(self.agent.memory.active_concepts), 0)
    
    def test_memory_management(self):
        """Test memory operations."""
        # Add some experiences
        for i in range(5):
            self.agent.perceive({"step": i, "data": f"test_data_{i}"})
        
        # Check memory has experiences
        self.assertGreater(len(self.agent.memory.recent_experiences), 0)
        
        # Test relevant experience retrieval
        relevant = self.agent.memory.get_relevant_experiences("step", limit=3)
        self.assertLessEqual(len(relevant), 3)
        
        # Test memory clearing
        self.agent.clear_memory()
        self.assertEqual(len(self.agent.memory.recent_experiences), 0)
        self.assertEqual(len(self.agent.memory.current_goals), 0)
    
    def test_cognitive_state_reporting(self):
        """Test cognitive state reporting."""
        # Add some activity
        self.agent.perceive({"test": "data"})
        self.agent.plan("test goal")
        
        state = self.agent.get_cognitive_state()
        
        self.assertIsInstance(state, dict)
        self.assertIn("state", state)
        self.assertIn("attention_mode", state)
        self.assertIn("active_concepts", state)
        self.assertIn("atomspace_size", state)
        self.assertIn("success_rate", state)
        self.assertIn("cycle_count", state)
        
        # Check data types
        self.assertIsInstance(state["active_concepts"], int)
        self.assertIsInstance(state["atomspace_size"], int)
        self.assertIsInstance(state["success_rate"], float)
        self.assertIsInstance(state["cycle_count"], int)
    
    def test_knowledge_persistence(self):
        """Test that knowledge persists across operations."""
        initial_size = self.agent.atomspace.size()
        
        # Add knowledge through perception
        self.agent.perceive({"fact1": "value1", "fact2": "value2"})
        
        # Knowledge should persist
        size_after_perception = self.agent.atomspace.size()
        self.assertGreater(size_after_perception, initial_size)
        
        # Reasoning might add some knowledge but shouldn't remove existing
        self.agent.reason("test query")
        self.assertGreaterEqual(self.agent.atomspace.size(), size_after_perception)
        
        # Planning adds goal knowledge
        self.agent.plan("test goal")
        self.assertGreaterEqual(self.agent.atomspace.size(), size_after_perception)


if __name__ == '__main__':
    unittest.main()