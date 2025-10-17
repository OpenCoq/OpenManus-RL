"""
OpenCog Reasoning Engine for symbolic reasoning in OpenManus-RL.

This module provides reasoning capabilities using OpenCog's AtomSpace,
including forward chaining, backward chaining, and probabilistic reasoning.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

from .atomspace_integration import AtomSpaceManager, Atom, AtomType


class ReasoningMode(Enum):
    """Different modes of reasoning."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    PROBABILISTIC = "probabilistic"
    PATTERN_MATCHING = "pattern_matching"


@dataclass
class ReasoningRule:
    """Represents a reasoning rule with premise and conclusion patterns."""
    name: str
    premises: List[Dict[str, Any]]
    conclusion: Dict[str, Any]
    confidence: float = 1.0
    
    def __str__(self):
        return f"Rule({self.name}): {self.premises} -> {self.conclusion}"


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    conclusion: Optional[Atom]
    confidence: float
    reasoning_path: List[str]
    used_rules: List[str]
    intermediate_results: List[Atom]


class OpenCogReasoningEngine:
    """
    OpenCog-based reasoning engine for symbolic inference.
    
    Provides various reasoning modes including forward/backward chaining
    and probabilistic reasoning using the AtomSpace knowledge base.
    """
    
    def __init__(self, atomspace: AtomSpaceManager):
        self.atomspace = atomspace
        self.rules: List[ReasoningRule] = []
        self.logger = logging.getLogger(__name__)
        self.reasoning_history: List[ReasoningResult] = []
        
        # Initialize with some basic reasoning rules
        self._initialize_basic_rules()
    
    def _initialize_basic_rules(self):
        """Initialize the engine with basic reasoning rules."""
        
        # Transitivity rule for inheritance
        transitivity_rule = ReasoningRule(
            name="inheritance_transitivity",
            premises=[
                {"type": "inheritance", "child": "$X", "parent": "$Y"},
                {"type": "inheritance", "child": "$Y", "parent": "$Z"}
            ],
            conclusion={"type": "inheritance", "child": "$X", "parent": "$Z"},
            confidence=0.9
        )
        
        # Similarity symmetry rule
        similarity_symmetry = ReasoningRule(
            name="similarity_symmetry", 
            premises=[
                {"type": "similarity", "a": "$X", "b": "$Y"}
            ],
            conclusion={"type": "similarity", "a": "$Y", "b": "$X"},
            confidence=1.0
        )
        
        # Action consequence rule
        action_consequence = ReasoningRule(
            name="action_consequence",
            premises=[
                {"type": "evaluation", "predicate": "action_taken", "args": ["$action"]},
                {"type": "evaluation", "predicate": "action_leads_to", "args": ["$action", "$outcome"]}
            ],
            conclusion={"type": "evaluation", "predicate": "expected_outcome", "args": ["$outcome"]},
            confidence=0.8
        )
        
        self.add_rule(transitivity_rule)
        self.add_rule(similarity_symmetry)
        self.add_rule(action_consequence)
    
    def add_rule(self, rule: ReasoningRule):
        """Add a reasoning rule to the engine."""
        self.rules.append(rule)
        self.logger.debug(f"Added reasoning rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove a reasoning rule by name."""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]
        removed = len(self.rules) < original_len
        if removed:
            self.logger.debug(f"Removed reasoning rule: {rule_name}")
        return removed
    
    def forward_chaining(self, max_iterations: int = 10) -> List[Atom]:
        """
        Perform forward chaining inference to derive new conclusions.
        
        Args:
            max_iterations: Maximum number of inference iterations
            
        Returns:
            List of newly derived atoms
        """
        derived_atoms = []
        iteration = 0
        
        while iteration < max_iterations:
            initial_count = self.atomspace.size()
            new_atoms_this_iteration = []
            
            for rule in self.rules:
                new_atoms = self._apply_rule_forward(rule)
                new_atoms_this_iteration.extend(new_atoms)
                derived_atoms.extend(new_atoms)
            
            iteration += 1
            
            # Stop if no new atoms were derived
            if self.atomspace.size() == initial_count:
                break
                
            self.logger.debug(f"Forward chaining iteration {iteration}: "
                            f"derived {len(new_atoms_this_iteration)} new atoms")
        
        self.logger.info(f"Forward chaining completed after {iteration} iterations, "
                        f"derived {len(derived_atoms)} total atoms")
        return derived_atoms
    
    def backward_chaining(self, goal: Dict[str, Any], max_depth: int = 5) -> ReasoningResult:
        """
        Perform backward chaining to prove a goal.
        
        Args:
            goal: Goal pattern to prove
            max_depth: Maximum reasoning depth
            
        Returns:
            ReasoningResult with proof information
        """
        reasoning_path = []
        used_rules = []
        intermediate_results = []
        
        def prove_goal(current_goal, depth):
            if depth > max_depth:
                return None, 0.0
            
            reasoning_path.append(f"Trying to prove: {current_goal}")
            
            # Check if goal is already in atomspace
            existing_atoms = self._find_matching_atoms(current_goal)
            if existing_atoms:
                best_atom = max(existing_atoms, key=lambda a: a.truth_value or 0)
                reasoning_path.append(f"Found existing fact: {best_atom.name}")
                return best_atom, best_atom.truth_value or 1.0
            
            # Try to prove using rules
            for rule in self.rules:
                if self._goal_matches_rule_conclusion(current_goal, rule.conclusion):
                    reasoning_path.append(f"Trying rule: {rule.name}")
                    used_rules.append(rule.name)
                    
                    # Try to prove all premises
                    premise_confidence = 1.0
                    premise_atoms = []
                    
                    for premise in rule.premises:
                        premise_atom, confidence = prove_goal(premise, depth + 1)
                        if premise_atom is None:
                            premise_confidence = 0.0
                            break
                        premise_atoms.append(premise_atom)
                        premise_confidence *= confidence
                    
                    if premise_confidence > 0:
                        # Create conclusion atom
                        conclusion_atom = self._create_atom_from_pattern(
                            rule.conclusion, premise_atoms
                        )
                        if conclusion_atom:
                            final_confidence = premise_confidence * rule.confidence
                            conclusion_atom.truth_value = final_confidence
                            intermediate_results.append(conclusion_atom)
                            reasoning_path.append(f"Proved using {rule.name} with confidence {final_confidence}")
                            return conclusion_atom, final_confidence
            
            reasoning_path.append(f"Failed to prove: {current_goal}")
            return None, 0.0
        
        conclusion, confidence = prove_goal(goal, 0)
        
        return ReasoningResult(
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=reasoning_path,
            used_rules=used_rules,
            intermediate_results=intermediate_results
        )
    
    def probabilistic_reasoning(self, query: Dict[str, Any], 
                              evidence: List[Dict[str, Any]]) -> float:
        """
        Perform probabilistic reasoning given evidence.
        
        Args:
            query: Query to evaluate
            evidence: List of evidence facts
            
        Returns:
            Probability of the query being true
        """
        # Simplified probabilistic reasoning
        # In a full implementation, this would use PLN (Probabilistic Logic Networks)
        
        base_probability = 0.1  # Prior probability
        
        # Add evidence to atomspace temporarily
        evidence_atoms = []
        for fact in evidence:
            atom = self._create_atom_from_pattern(fact)
            if atom:
                evidence_atoms.append(atom)
        
        # Try backward chaining with evidence
        result = self.backward_chaining(query)
        
        if result.conclusion:
            # Evidence supports the query
            probability = min(0.95, base_probability + result.confidence * 0.8)
        else:
            # Look for contradictory evidence
            probability = base_probability
        
        # Clean up temporary evidence atoms
        for atom in evidence_atoms:
            if atom.atom_id in self.atomspace:
                self.atomspace.remove_atom(atom.atom_id)
        
        return probability
    
    def explain_reasoning(self, result: ReasoningResult) -> str:
        """
        Generate a human-readable explanation of the reasoning process.
        
        Args:
            result: ReasoningResult to explain
            
        Returns:
            Formatted explanation string
        """
        explanation_lines = []
        explanation_lines.append("=== Reasoning Explanation ===")
        
        if result.conclusion:
            explanation_lines.append(f"Conclusion: {result.conclusion.name}")
            explanation_lines.append(f"Confidence: {result.confidence:.3f}")
        else:
            explanation_lines.append("No conclusion could be reached")
        
        explanation_lines.append("\nReasoning Path:")
        for step in result.reasoning_path:
            explanation_lines.append(f"  • {step}")
        
        if result.used_rules:
            explanation_lines.append("\nRules Applied:")
            for rule_name in result.used_rules:
                explanation_lines.append(f"  • {rule_name}")
        
        if result.intermediate_results:
            explanation_lines.append("\nIntermediate Results:")
            for atom in result.intermediate_results:
                explanation_lines.append(f"  • {atom.name} (confidence: {atom.truth_value:.3f})")
        
        return "\n".join(explanation_lines)
    
    def reason_about_action(self, action: str, context: Dict[str, Any]) -> ReasoningResult:
        """
        Reason about the consequences and appropriateness of an action.
        
        Args:
            action: Action to reason about
            context: Context information for the reasoning
            
        Returns:
            ReasoningResult with action analysis
        """
        # Create atoms for the action and context
        action_atom = self.atomspace.create_concept_node(f"action_{action}")
        
        # Add context information
        for key, value in context.items():
            context_atom = self.atomspace.create_concept_node(f"context_{key}_{value}")
            self.atomspace.create_evaluation_link(
                self.atomspace.create_predicate_node("has_context"),
                [action_atom, context_atom]
            )
        
        # Query for action consequences
        consequence_query = {
            "type": "evaluation",
            "predicate": "action_leads_to",
            "args": [action, "$outcome"]
        }
        
        result = self.backward_chaining(consequence_query)
        
        # Add action reasoning to history
        self.reasoning_history.append(result)
        
        return result
    
    def _apply_rule_forward(self, rule: ReasoningRule) -> List[Atom]:
        """Apply a rule in forward chaining mode."""
        new_atoms = []
        
        # Find all possible variable bindings for the rule premises
        bindings_list = self._find_rule_bindings(rule)
        
        for bindings in bindings_list:
            # Check if conclusion already exists
            conclusion_pattern = self._substitute_variables(rule.conclusion, bindings)
            existing = self._find_matching_atoms(conclusion_pattern)
            
            if not existing:
                # Create new conclusion atom
                conclusion_atom = self._create_atom_from_pattern(conclusion_pattern)
                if conclusion_atom:
                    conclusion_atom.confidence = rule.confidence
                    new_atoms.append(conclusion_atom)
                    self.logger.debug(f"Applied rule {rule.name}: created {conclusion_atom.name}")
        
        return new_atoms
    
    def _find_rule_bindings(self, rule: ReasoningRule) -> List[Dict[str, str]]:
        """Find all possible variable bindings for a rule's premises."""
        # Simplified binding finder - would be more sophisticated in full implementation
        return []
    
    def _find_matching_atoms(self, pattern: Dict[str, Any]) -> List[Atom]:
        """Find atoms matching a pattern."""
        if pattern.get("type") == "concept":
            return self.atomspace.find_atoms(AtomType.CONCEPT_NODE, pattern.get("name"))
        elif pattern.get("type") == "evaluation":
            return self.atomspace.find_atoms(AtomType.EVALUATION_LINK)
        return []
    
    def _goal_matches_rule_conclusion(self, goal: Dict[str, Any], 
                                    conclusion: Dict[str, Any]) -> bool:
        """Check if a goal matches a rule's conclusion pattern."""
        return goal.get("type") == conclusion.get("type")
    
    def _create_atom_from_pattern(self, pattern: Dict[str, Any], 
                                context_atoms: List[Atom] = None) -> Optional[Atom]:
        """Create an atom from a pattern description."""
        pattern_type = pattern.get("type")
        
        if pattern_type == "concept":
            return self.atomspace.create_concept_node(pattern.get("name", "unknown"))
        elif pattern_type == "evaluation":
            predicate = self.atomspace.create_predicate_node(
                pattern.get("predicate", "unknown_predicate")
            )
            # Simplified - would handle arguments properly in full implementation
            args = [self.atomspace.create_concept_node("arg")]
            return self.atomspace.create_evaluation_link(predicate, args)
        elif pattern_type == "inheritance":
            child = self.atomspace.create_concept_node(pattern.get("child", "child"))
            parent = self.atomspace.create_concept_node(pattern.get("parent", "parent"))
            return self.atomspace.create_inheritance_link(child, parent)
        
        return None
    
    def _substitute_variables(self, pattern: Dict[str, Any], 
                            bindings: Dict[str, str]) -> Dict[str, Any]:
        """Substitute variables in a pattern with their bindings."""
        # Simplified variable substitution
        substituted = pattern.copy()
        for key, value in substituted.items():
            if isinstance(value, str) and value.startswith("$"):
                substituted[key] = bindings.get(value, value)
        return substituted
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about reasoning operations."""
        return {
            "total_rules": len(self.rules),
            "atomspace_size": self.atomspace.size(),
            "reasoning_history_length": len(self.reasoning_history),
            "successful_reasonings": sum(1 for r in self.reasoning_history if r.conclusion),
            "average_confidence": sum(r.confidence for r in self.reasoning_history) / max(1, len(self.reasoning_history))
        }