"""
OpenCog Cognitive Architecture for OpenManus-RL agents.

This module implements a cognitive architecture using OpenCog components,
providing integrated symbolic reasoning, learning, and decision-making capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

from .atomspace_integration import AtomSpaceManager, Atom, AtomType
from .reasoning_engine import OpenCogReasoningEngine, ReasoningResult
from .pattern_matcher import OpenCogPatternMatcher, PatternQuery, MatchType
from .knowledge_representation import KnowledgeGraph


class CognitiveState(Enum):
    """States of the cognitive processing cycle."""
    PERCEIVING = "perceiving"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTING = "acting"
    LEARNING = "learning"


class AttentionMode(Enum):
    """Modes of attention allocation."""
    FOCUSED = "focused"
    DISTRIBUTED = "distributed"
    EXPLORATORY = "exploratory"
    GOAL_DIRECTED = "goal_directed"


@dataclass
class CognitiveMemory:
    """Working memory for the cognitive agent."""
    current_goals: List[Atom] = field(default_factory=list)
    active_concepts: List[Atom] = field(default_factory=list)
    recent_experiences: List[Dict[str, Any]] = field(default_factory=list)
    attention_focus: Optional[Atom] = None
    confidence_levels: Dict[str, float] = field(default_factory=dict)
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add a new experience to memory."""
        self.recent_experiences.append({
            **experience,
            'timestamp': time.time()
        })
        # Keep only recent experiences (last 100)
        self.recent_experiences = self.recent_experiences[-100:]
    
    def get_relevant_experiences(self, context: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get experiences relevant to a context."""
        # Simplified relevance matching - would be more sophisticated in practice
        relevant = []
        for exp in reversed(self.recent_experiences):
            if context.lower() in str(exp).lower():
                relevant.append(exp)
                if len(relevant) >= limit:
                    break
        return relevant


@dataclass 
class CognitiveAction:
    """Represents an action in the cognitive space."""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning_path: List[str]
    expected_outcome: Optional[str] = None
    
    def __str__(self):
        return f"CognitiveAction({self.action_type}, conf={self.confidence:.3f})"


class CognitiveAgent:
    """
    OpenCog-based cognitive agent for complex reasoning and decision-making.
    
    Integrates AtomSpace, reasoning engine, and pattern matcher to provide
    a comprehensive cognitive architecture for RL agents.
    """
    
    def __init__(self, agent_name: str = "cognitive_agent"):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
        # Core OpenCog components
        self.atomspace = AtomSpaceManager()
        self.reasoning_engine = OpenCogReasoningEngine(self.atomspace)
        self.pattern_matcher = OpenCogPatternMatcher(self.atomspace)
        self.knowledge_graph = KnowledgeGraph(self.atomspace)
        
        # Cognitive state management
        self.state = CognitiveState.PERCEIVING
        self.memory = CognitiveMemory()
        self.attention_mode = AttentionMode.FOCUSED
        
        # Learning and adaptation
        self.learning_rate = 0.1
        self.exploration_factor = 0.2
        
        # Performance tracking
        self.action_history: List[CognitiveAction] = []
        self.success_rate = 0.0
        self.cycle_count = 0
        
        # Initialize basic knowledge
        self._initialize_basic_knowledge()
    
    def _initialize_basic_knowledge(self):
        """Initialize the agent with basic knowledge and concepts."""
        
        # Basic concepts
        self_concept = self.atomspace.create_concept_node("self")
        agent_concept = self.atomspace.create_concept_node("agent")
        environment_concept = self.atomspace.create_concept_node("environment")
        
        # Self-identification
        self.atomspace.create_inheritance_link(self_concept, agent_concept, 1.0)
        
        # Basic actions
        action_concept = self.atomspace.create_concept_node("action")
        think_action = self.atomspace.create_concept_node("think")
        observe_action = self.atomspace.create_concept_node("observe")
        
        self.atomspace.create_inheritance_link(think_action, action_concept, 1.0)
        self.atomspace.create_inheritance_link(observe_action, action_concept, 1.0)
        
        # Goal concepts
        goal_concept = self.atomspace.create_concept_node("goal")
        success_concept = self.atomspace.create_concept_node("success")
        
        # Add to knowledge graph
        self.knowledge_graph.add_entity("self", {"type": "agent", "name": self.agent_name})
        self.knowledge_graph.add_entity("environment", {"type": "context"})
        
        self.logger.info("Initialized basic knowledge structures")
    
    def perceive(self, observations: Dict[str, Any]) -> None:
        """
        Process new observations and update internal state.
        
        Args:
            observations: Dictionary of observed information
        """
        self.state = CognitiveState.PERCEIVING
        self.logger.debug(f"Perceiving: {observations}")
        
        # Convert observations to atoms
        observation_atoms = []
        for key, value in observations.items():
            # Create observation atoms
            obs_atom = self.atomspace.create_concept_node(f"obs_{key}_{value}")
            observation_atoms.append(obs_atom)
            
            # Create evaluation link for the observation
            obs_predicate = self.atomspace.create_predicate_node("observed")
            self.atomspace.create_evaluation_link(obs_predicate, [obs_atom], 
                                                truth_value=0.9)
        
        # Update working memory
        self.memory.active_concepts.extend(observation_atoms)
        self.memory.add_experience({
            "type": "observation",
            "content": observations,
            "atoms": [atom.atom_id for atom in observation_atoms]
        })
        
        # Update attention based on observations
        self._update_attention(observation_atoms)
    
    def reason(self, query: Optional[str] = None) -> ReasoningResult:
        """
        Perform reasoning about the current situation.
        
        Args:
            query: Optional specific query to reason about
            
        Returns:
            ReasoningResult with conclusions and reasoning path
        """
        self.state = CognitiveState.REASONING
        self.logger.debug(f"Reasoning about: {query or 'current situation'}")
        
        if query:
            # Specific query reasoning
            query_pattern = self._parse_query_to_pattern(query)
            result = self.reasoning_engine.backward_chaining(query_pattern)
        else:
            # General situation reasoning
            result = self._reason_about_current_situation()
        
        # Update memory with reasoning results
        self.memory.add_experience({
            "type": "reasoning",
            "query": query,
            "result": result,
            "confidence": result.confidence
        })
        
        # Update confidence levels
        if result.conclusion:
            self.memory.confidence_levels[result.conclusion.name] = result.confidence
        
        return result
    
    def plan(self, goal: str, context: Dict[str, Any] = None) -> List[CognitiveAction]:
        """
        Create a plan to achieve a goal.
        
        Args:
            goal: Goal description
            context: Additional context information
            
        Returns:
            List of cognitive actions forming a plan
        """
        self.state = CognitiveState.PLANNING
        self.logger.debug(f"Planning for goal: {goal}")
        
        # Create goal atom
        goal_atom = self.atomspace.create_concept_node(f"goal_{goal}")
        self.memory.current_goals.append(goal_atom)
        
        # Use knowledge graph to find relevant actions
        relevant_actions = self.knowledge_graph.find_related_entities(
            "action", max_distance=2
        )
        
        plan = []
        
        # Simple planning: create actions based on goal and context
        if "explore" in goal.lower():
            action = CognitiveAction(
                action_type="explore_environment",
                parameters={"strategy": "systematic"},
                confidence=0.8,
                reasoning_path=["Goal requires exploration", "Systematic exploration is effective"],
                expected_outcome="Increased knowledge of environment"
            )
            plan.append(action)
        
        elif "learn" in goal.lower():
            action = CognitiveAction(
                action_type="analyze_patterns", 
                parameters={"focus": context.get("domain", "general")},
                confidence=0.7,
                reasoning_path=["Learning goal identified", "Pattern analysis aids learning"],
                expected_outcome="Improved understanding"
            )
            plan.append(action)
        
        else:
            # Default action: gather more information
            action = CognitiveAction(
                action_type="gather_information",
                parameters={"target": goal},
                confidence=0.6,
                reasoning_path=["Goal not fully understood", "More information needed"],
                expected_outcome="Better goal understanding"
            )
            plan.append(action)
        
        # Add plan to memory
        self.memory.add_experience({
            "type": "planning",
            "goal": goal,
            "plan": plan,
            "context": context
        })
        
        return plan
    
    def act(self, action: CognitiveAction) -> Dict[str, Any]:
        """
        Execute a cognitive action.
        
        Args:
            action: CognitiveAction to execute
            
        Returns:
            Dictionary with action results
        """
        self.state = CognitiveState.ACTING
        self.logger.debug(f"Executing action: {action}")
        
        result = {"success": False, "output": None, "confidence": 0.0}
        
        try:
            if action.action_type == "explore_environment":
                result = self._execute_exploration(action)
            elif action.action_type == "analyze_patterns":
                result = self._execute_pattern_analysis(action)
            elif action.action_type == "gather_information":
                result = self._execute_information_gathering(action)
            else:
                # Generic action execution
                result = self._execute_generic_action(action)
            
            # Record action execution
            self.action_history.append(action)
            
            # Update success rate
            if result["success"]:
                self.success_rate = (self.success_rate * len(self.action_history) + 1) / (len(self.action_history) + 1)
            
            # Add to memory
            self.memory.add_experience({
                "type": "action_execution",
                "action": action,
                "result": result
            })
            
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
            result["error"] = str(e)
        
        return result
    
    def learn(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback and update knowledge.
        
        Args:
            feedback: Feedback information including rewards, corrections, etc.
        """
        self.state = CognitiveState.LEARNING
        self.logger.debug(f"Learning from feedback: {feedback}")
        
        # Extract learning signals
        reward = feedback.get("reward", 0.0)
        correction = feedback.get("correction")
        success = feedback.get("success", False)
        
        # Update knowledge based on feedback
        if success and self.action_history:
            # Reinforce successful actions
            last_action = self.action_history[-1]
            self._reinforce_action_knowledge(last_action, reward)
        
        if correction:
            # Learn from corrections
            self._learn_from_correction(correction)
        
        # Update confidence levels
        recent_experiences = self.memory.get_relevant_experiences("action", limit=5)
        if recent_experiences:
            successful_actions = 0
            for exp in recent_experiences:
                if isinstance(exp, dict) and "result" in exp:
                    result = exp.get("result", {})
                    if isinstance(result, dict) and result.get("success", False):
                        successful_actions += 1
            
            avg_success = successful_actions / len(recent_experiences) if recent_experiences else 0
            self.memory.confidence_levels["general_performance"] = avg_success
        
        # Add learning experience to memory
        self.memory.add_experience({
            "type": "learning",
            "feedback": feedback,
            "adjustment": "knowledge_updated"
        })
    
    def cognitive_cycle(self, observations: Dict[str, Any], 
                       goal: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a complete cognitive cycle: perceive -> reason -> plan -> act.
        
        Args:
            observations: Current observations
            goal: Optional goal to work towards
            
        Returns:
            Dictionary with cycle results
        """
        self.cycle_count += 1
        cycle_start = time.time()
        
        # Perceive
        self.perceive(observations)
        
        # Reason about current situation
        reasoning_result = self.reason()
        
        # Plan if we have a goal
        plan = []
        if goal:
            plan = self.plan(goal, {"observations": observations})
        
        # Act on the first planned action or a default action
        action_result = None
        if plan:
            action_result = self.act(plan[0])
        else:
            # Default action: analyze current situation
            default_action = CognitiveAction(
                action_type="analyze_situation",
                parameters={"observations": observations},
                confidence=0.5,
                reasoning_path=["No specific plan", "Analyzing situation"]
            )
            action_result = self.act(default_action)
        
        cycle_time = time.time() - cycle_start
        
        cycle_result = {
            "cycle_number": self.cycle_count,
            "cycle_time": cycle_time,
            "reasoning_result": reasoning_result,
            "plan": plan,
            "action_result": action_result,
            "attention_focus": self.memory.attention_focus.name if self.memory.attention_focus else None,
            "active_concepts": len(self.memory.active_concepts),
            "success_rate": self.success_rate
        }
        
        self.logger.info(f"Completed cognitive cycle {self.cycle_count} in {cycle_time:.3f}s")
        return cycle_result
    
    def _update_attention(self, new_atoms: List[Atom]):
        """Update attention focus based on new information."""
        if not new_atoms:
            return
        
        if self.attention_mode == AttentionMode.FOCUSED:
            # Focus on the most salient atom (simplified)
            self.memory.attention_focus = new_atoms[0]
        elif self.attention_mode == AttentionMode.EXPLORATORY:
            # Rotate attention among new atoms
            if len(new_atoms) > 1:
                self.memory.attention_focus = new_atoms[1]
        
        # Update active concepts (limited working memory)
        self.memory.active_concepts.extend(new_atoms)
        self.memory.active_concepts = self.memory.active_concepts[-20:]  # Keep last 20
    
    def _parse_query_to_pattern(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query into a pattern for reasoning."""
        # Simplified query parsing - would be more sophisticated in practice
        query_lower = query.lower()
        
        if "what" in query_lower:
            return {"type": "evaluation", "predicate": "has_property", "args": ["$entity", "$property"]}
        elif "who" in query_lower:
            return {"type": "concept", "name": "$person"}
        elif "where" in query_lower:
            return {"type": "evaluation", "predicate": "located_at", "args": ["$entity", "$location"]}
        else:
            return {"type": "concept", "name": "$unknown"}
    
    def _reason_about_current_situation(self) -> ReasoningResult:
        """Reason about the current situation based on active concepts."""
        # Create a general situational query
        situation_query = {
            "type": "evaluation",
            "predicate": "current_situation",
            "args": ["$state"]
        }
        
        return self.reasoning_engine.backward_chaining(situation_query, max_depth=3)
    
    def _execute_exploration(self, action: CognitiveAction) -> Dict[str, Any]:
        """Execute exploration action."""
        strategy = action.parameters.get("strategy", "random")
        
        # Use pattern matcher to find unexplored areas
        query = PatternQuery({"type": "concept", "name": "$unknown"})
        matches = self.pattern_matcher.match_pattern(query, MatchType.FUZZY)
        
        result = {
            "success": True,
            "output": f"Explored using {strategy} strategy",
            "confidence": action.confidence,
            "discoveries": len(matches)
        }
        
        return result
    
    def _execute_pattern_analysis(self, action: CognitiveAction) -> Dict[str, Any]:
        """Execute pattern analysis action."""
        focus = action.parameters.get("focus", "general")
        
        # Find patterns in recent experiences
        relevant_experiences = self.memory.get_relevant_experiences(focus, limit=10)
        
        # Simple pattern detection
        pattern_count = len(set(exp.get("type") for exp in relevant_experiences))
        
        result = {
            "success": True,
            "output": f"Analyzed patterns in {focus} domain",
            "confidence": action.confidence,
            "patterns_found": pattern_count,
            "experience_count": len(relevant_experiences)
        }
        
        return result
    
    def _execute_information_gathering(self, action: CognitiveAction) -> Dict[str, Any]:
        """Execute information gathering action."""
        target = action.parameters.get("target", "general")
        
        # Query atomspace for relevant information
        target_atoms = self.atomspace.find_atoms(name=target)
        related_info = []
        
        for atom in target_atoms:
            incoming = self.atomspace.get_incoming_set(atom)
            outgoing = self.atomspace.get_outgoing_set(atom)
            related_info.extend(incoming + outgoing)
        
        result = {
            "success": len(related_info) > 0,
            "output": f"Gathered information about {target}",
            "confidence": action.confidence,
            "information_pieces": len(related_info)
        }
        
        return result
    
    def _execute_generic_action(self, action: CognitiveAction) -> Dict[str, Any]:
        """Execute a generic action."""
        result = {
            "success": True,
            "output": f"Executed {action.action_type}",
            "confidence": action.confidence * 0.8  # Lower confidence for generic actions
        }
        
        return result
    
    def _reinforce_action_knowledge(self, action: CognitiveAction, reward: float):
        """Reinforce knowledge about successful actions."""
        # Create or update action-outcome relationships
        action_atom = self.atomspace.create_concept_node(f"action_{action.action_type}")
        outcome_atom = self.atomspace.create_concept_node("positive_outcome")
        
        # Create evaluation link with confidence based on reward
        leads_to_predicate = self.atomspace.create_predicate_node("leads_to")
        self.atomspace.create_evaluation_link(
            leads_to_predicate,
            [action_atom, outcome_atom],
            truth_value=min(1.0, max(0.1, reward))
        )
    
    def _learn_from_correction(self, correction: str):
        """Learn from correction feedback."""
        # Simple correction learning - create negative examples
        correction_atom = self.atomspace.create_concept_node(f"correction_{correction}")
        mistake_atom = self.atomspace.create_concept_node("mistake")
        
        self.atomspace.create_inheritance_link(correction_atom, mistake_atom, 0.8)
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state information."""
        return {
            "state": self.state.value,
            "attention_mode": self.attention_mode.value,
            "attention_focus": self.memory.attention_focus.name if self.memory.attention_focus else None,
            "active_concepts": len(self.memory.active_concepts),
            "current_goals": [goal.name for goal in self.memory.current_goals],
            "recent_experiences": len(self.memory.recent_experiences),
            "confidence_levels": self.memory.confidence_levels.copy(),
            "atomspace_size": self.atomspace.size(),
            "success_rate": self.success_rate,
            "cycle_count": self.cycle_count
        }
    
    def set_attention_mode(self, mode: AttentionMode):
        """Set the attention allocation mode."""
        self.attention_mode = mode
        self.logger.debug(f"Attention mode set to: {mode.value}")
    
    def clear_memory(self):
        """Clear working memory while preserving long-term knowledge."""
        self.memory = CognitiveMemory()
        self.logger.info("Working memory cleared")