"""
OpenCog Pattern Matcher for symbolic pattern recognition and matching.

This module provides pattern matching capabilities using OpenCog's AtomSpace,
enabling complex query processing and knowledge retrieval.
"""

from typing import List, Dict, Any, Optional, Set, Union, Callable
import re
from dataclasses import dataclass
from enum import Enum

from .atomspace_integration import AtomSpaceManager, Atom, AtomType


class MatchType(Enum):
    """Types of pattern matching."""
    EXACT = "exact"
    FUZZY = "fuzzy"  
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass
class PatternVariable:
    """Represents a pattern variable with constraints."""
    name: str
    atom_type: Optional[AtomType] = None
    constraints: Optional[List[Callable]] = None
    
    def matches(self, atom: Atom) -> bool:
        """Check if an atom satisfies this variable's constraints."""
        if self.atom_type and atom.atom_type != self.atom_type:
            return False
        
        if self.constraints:
            for constraint in self.constraints:
                if not constraint(atom):
                    return False
        
        return True


@dataclass
class MatchResult:
    """Result of a pattern matching operation."""
    bindings: Dict[str, Atom]
    confidence: float
    match_type: MatchType
    
    def __str__(self):
        return f"Match(confidence={self.confidence:.3f}, bindings={len(self.bindings)})"


class PatternQuery:
    """Represents a pattern query with variables and constraints."""
    
    def __init__(self, pattern_dict: Dict[str, Any]):
        self.pattern = pattern_dict
        self.variables: Dict[str, PatternVariable] = {}
        self._parse_variables()
    
    def _parse_variables(self):
        """Extract variables from the pattern."""
        self._extract_variables_recursive(self.pattern)
    
    def _extract_variables_recursive(self, obj):
        """Recursively extract variables from a nested structure."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith('$'):
                    if value not in self.variables:
                        self.variables[value] = PatternVariable(value)
                elif isinstance(value, (dict, list)):
                    self._extract_variables_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                self._extract_variables_recursive(item)
    
    def add_constraint(self, var_name: str, constraint: Callable[[Atom], bool]):
        """Add a constraint to a variable."""
        if var_name in self.variables:
            if self.variables[var_name].constraints is None:
                self.variables[var_name].constraints = []
            self.variables[var_name].constraints.append(constraint)
    
    def set_variable_type(self, var_name: str, atom_type: AtomType):
        """Set the required atom type for a variable."""
        if var_name in self.variables:
            self.variables[var_name].atom_type = atom_type


class OpenCogPatternMatcher:
    """
    OpenCog Pattern Matcher for complex symbolic pattern recognition.
    
    Provides sophisticated pattern matching capabilities including exact matching,
    fuzzy matching, and structural similarity detection.
    """
    
    def __init__(self, atomspace: AtomSpaceManager):
        self.atomspace = atomspace
        self.match_threshold = 0.7
        self.max_results = 100
    
    def match_pattern(self, query: PatternQuery, 
                     match_type: MatchType = MatchType.EXACT) -> List[MatchResult]:
        """
        Match a pattern query against the atomspace.
        
        Args:
            query: Pattern query to match
            match_type: Type of matching to perform
            
        Returns:
            List of match results sorted by confidence
        """
        if match_type == MatchType.EXACT:
            return self._exact_match(query)
        elif match_type == MatchType.FUZZY:
            return self._fuzzy_match(query)
        elif match_type == MatchType.STRUCTURAL:
            return self._structural_match(query)
        elif match_type == MatchType.SEMANTIC:
            return self._semantic_match(query)
        else:
            raise ValueError(f"Unknown match type: {match_type}")
    
    def _exact_match(self, query: PatternQuery) -> List[MatchResult]:
        """Perform exact pattern matching."""
        results = []
        
        # Get all possible starting atoms based on the query
        candidate_atoms = self._get_candidate_atoms(query.pattern)
        
        for atom in candidate_atoms:
            bindings = {}
            if self._try_match_atom(atom, query.pattern, bindings, query.variables):
                result = MatchResult(
                    bindings=bindings.copy(),
                    confidence=1.0,
                    match_type=MatchType.EXACT
                )
                results.append(result)
        
        return results[:self.max_results]
    
    def _fuzzy_match(self, query: PatternQuery) -> List[MatchResult]:
        """Perform fuzzy pattern matching with similarity scoring."""
        results = []
        
        candidate_atoms = self._get_candidate_atoms(query.pattern)
        
        for atom in candidate_atoms:
            bindings = {}
            similarity = self._calculate_fuzzy_similarity(atom, query.pattern, bindings, query.variables)
            
            if similarity >= self.match_threshold:
                result = MatchResult(
                    bindings=bindings,
                    confidence=similarity,
                    match_type=MatchType.FUZZY
                )
                results.append(result)
        
        # Sort by confidence (similarity)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:self.max_results]
    
    def _structural_match(self, query: PatternQuery) -> List[MatchResult]:
        """Match based on structural similarity (link structure)."""
        results = []
        
        # Focus on link atoms for structural matching
        link_atoms = [atom for atom in self.atomspace.atoms.values() 
                     if atom.outgoing]
        
        for atom in link_atoms:
            bindings = {}
            structural_score = self._calculate_structural_similarity(
                atom, query.pattern, bindings, query.variables
            )
            
            if structural_score >= self.match_threshold:
                result = MatchResult(
                    bindings=bindings,
                    confidence=structural_score,
                    match_type=MatchType.STRUCTURAL
                )
                results.append(result)
        
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:self.max_results]
    
    def _semantic_match(self, query: PatternQuery) -> List[MatchResult]:
        """Match based on semantic similarity."""
        results = []
        
        # Simplified semantic matching - in a full implementation,
        # this would use word embeddings or semantic networks
        
        candidate_atoms = self._get_candidate_atoms(query.pattern)
        
        for atom in candidate_atoms:
            bindings = {}
            semantic_score = self._calculate_semantic_similarity(
                atom, query.pattern, bindings, query.variables
            )
            
            if semantic_score >= self.match_threshold:
                result = MatchResult(
                    bindings=bindings,
                    confidence=semantic_score,
                    match_type=MatchType.SEMANTIC
                )
                results.append(result)
        
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:self.max_results]
    
    def _get_candidate_atoms(self, pattern: Dict[str, Any]) -> List[Atom]:
        """Get candidate atoms that could potentially match the pattern."""
        pattern_type = pattern.get("type")
        
        if pattern_type == "concept":
            return self.atomspace.find_atoms(AtomType.CONCEPT_NODE)
        elif pattern_type == "predicate":
            return self.atomspace.find_atoms(AtomType.PREDICATE_NODE)
        elif pattern_type == "evaluation":
            return self.atomspace.find_atoms(AtomType.EVALUATION_LINK)
        elif pattern_type == "inheritance":
            return self.atomspace.find_atoms(AtomType.INHERITANCE_LINK)
        elif pattern_type == "similarity":
            return self.atomspace.find_atoms(AtomType.SIMILARITY_LINK)
        else:
            # Return all atoms if type not specified
            return list(self.atomspace.atoms.values())
    
    def _try_match_atom(self, atom: Atom, pattern: Dict[str, Any], 
                       bindings: Dict[str, Atom], 
                       variables: Dict[str, PatternVariable]) -> bool:
        """Try to match an atom against a pattern."""
        
        # Handle variable patterns
        if isinstance(pattern, str) and pattern.startswith('$'):
            return self._bind_variable(pattern, atom, bindings, variables)
        
        # Handle dictionary patterns
        if isinstance(pattern, dict):
            pattern_type = pattern.get("type")
            
            # Check atom type matches
            if pattern_type == "concept" and atom.atom_type != AtomType.CONCEPT_NODE:
                return False
            elif pattern_type == "evaluation" and atom.atom_type != AtomType.EVALUATION_LINK:
                return False
            
            # Check name if specified
            pattern_name = pattern.get("name")
            if pattern_name:
                if isinstance(pattern_name, str):
                    if pattern_name.startswith('$'):
                        if not self._bind_variable(pattern_name, atom, bindings, variables):
                            return False
                    elif pattern_name != atom.name:
                        return False
            
            # Check outgoing set for links
            pattern_outgoing = pattern.get("outgoing")
            if pattern_outgoing and len(pattern_outgoing) != len(atom.outgoing):
                return False
            
            if pattern_outgoing:
                for i, out_pattern in enumerate(pattern_outgoing):
                    if not self._try_match_atom(atom.outgoing[i], out_pattern, bindings, variables):
                        return False
            
            return True
        
        return False
    
    def _bind_variable(self, var_name: str, atom: Atom, 
                      bindings: Dict[str, Atom], 
                      variables: Dict[str, PatternVariable]) -> bool:
        """Bind a variable to an atom if constraints are satisfied."""
        
        # Check if variable already bound
        if var_name in bindings:
            return bindings[var_name] == atom
        
        # Check variable constraints
        if var_name in variables:
            variable = variables[var_name]
            if not variable.matches(atom):
                return False
        
        # Bind the variable
        bindings[var_name] = atom
        return True
    
    def _calculate_fuzzy_similarity(self, atom: Atom, pattern: Dict[str, Any],
                                   bindings: Dict[str, Atom],
                                   variables: Dict[str, PatternVariable]) -> float:
        """Calculate fuzzy similarity between an atom and pattern."""
        
        # Simplified fuzzy matching - would be more sophisticated in practice
        base_score = 0.0
        
        if isinstance(pattern, dict):
            pattern_type = pattern.get("type")
            
            # Type similarity
            if pattern_type == "concept" and atom.atom_type == AtomType.CONCEPT_NODE:
                base_score += 0.3
            elif pattern_type == "evaluation" and atom.atom_type == AtomType.EVALUATION_LINK:
                base_score += 0.3
            
            # Name similarity
            pattern_name = pattern.get("name")
            if pattern_name and not pattern_name.startswith('$'):
                name_similarity = self._string_similarity(atom.name, pattern_name)
                base_score += 0.4 * name_similarity
            
            # Structural similarity for links
            pattern_outgoing = pattern.get("outgoing")
            if pattern_outgoing and atom.outgoing:
                struct_similarity = min(len(pattern_outgoing), len(atom.outgoing)) / max(len(pattern_outgoing), len(atom.outgoing))
                base_score += 0.3 * struct_similarity
        
        return min(1.0, base_score)
    
    def _calculate_structural_similarity(self, atom: Atom, pattern: Dict[str, Any],
                                       bindings: Dict[str, Atom],
                                       variables: Dict[str, PatternVariable]) -> float:
        """Calculate structural similarity based on graph structure."""
        
        if not atom.outgoing:
            return 0.1  # Low score for nodes in structural matching
        
        pattern_outgoing = pattern.get("outgoing", [])
        if not pattern_outgoing:
            return 0.1
        
        # Compare outgoing structures
        similarity = 0.0
        
        # Arity similarity
        arity_similarity = min(len(atom.outgoing), len(pattern_outgoing)) / max(len(atom.outgoing), len(pattern_outgoing))
        similarity += 0.4 * arity_similarity
        
        # Type distribution similarity
        atom_types = [out.atom_type for out in atom.outgoing]
        pattern_types = []
        for out_pattern in pattern_outgoing:
            if isinstance(out_pattern, dict):
                ptype = out_pattern.get("type")
                if ptype == "concept":
                    pattern_types.append(AtomType.CONCEPT_NODE)
                elif ptype == "predicate":
                    pattern_types.append(AtomType.PREDICATE_NODE)
        
        if pattern_types:
            type_intersection = len(set(atom_types) & set(pattern_types))
            type_union = len(set(atom_types) | set(pattern_types))
            type_similarity = type_intersection / max(1, type_union)
            similarity += 0.6 * type_similarity
        
        return min(1.0, similarity)
    
    def _calculate_semantic_similarity(self, atom: Atom, pattern: Dict[str, Any],
                                     bindings: Dict[str, Atom],
                                     variables: Dict[str, PatternVariable]) -> float:
        """Calculate semantic similarity using name analysis."""
        
        pattern_name = pattern.get("name")
        if not pattern_name or pattern_name.startswith('$'):
            return 0.5  # Default similarity for variables
        
        # Simple semantic similarity based on string matching
        # In a full implementation, this would use word embeddings
        return self._string_similarity(atom.name, pattern_name)
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using various metrics."""
        
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Case-insensitive match  
        if str1.lower() == str2.lower():
            return 0.95
        
        # Substring match
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Edit distance similarity
        edit_distance = self._levenshtein_distance(str1.lower(), str2.lower())
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (edit_distance / max_len)
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def query_by_example(self, example_atom: Atom, similarity_threshold: float = 0.8) -> List[MatchResult]:
        """Find atoms similar to an example atom."""
        results = []
        
        for atom in self.atomspace.atoms.values():
            if atom == example_atom:
                continue
            
            similarity = self._calculate_atom_similarity(example_atom, atom)
            if similarity >= similarity_threshold:
                result = MatchResult(
                    bindings={"$match": atom},
                    confidence=similarity,
                    match_type=MatchType.FUZZY
                )
                results.append(result)
        
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results[:self.max_results]
    
    def _calculate_atom_similarity(self, atom1: Atom, atom2: Atom) -> float:
        """Calculate overall similarity between two atoms."""
        
        # Type similarity
        type_similarity = 1.0 if atom1.atom_type == atom2.atom_type else 0.3
        
        # Name similarity
        name_similarity = self._string_similarity(atom1.name, atom2.name)
        
        # Structure similarity for links
        struct_similarity = 1.0
        if atom1.outgoing and atom2.outgoing:
            if len(atom1.outgoing) == len(atom2.outgoing):
                struct_similarity = 1.0
            else:
                struct_similarity = min(len(atom1.outgoing), len(atom2.outgoing)) / max(len(atom1.outgoing), len(atom2.outgoing))
        elif atom1.outgoing or atom2.outgoing:
            struct_similarity = 0.5
        
        # Weighted combination
        overall_similarity = (0.4 * type_similarity + 
                            0.4 * name_similarity + 
                            0.2 * struct_similarity)
        
        return overall_similarity
    
    def find_paths(self, start_atom: Atom, end_atom: Atom, 
                   max_depth: int = 3) -> List[List[Atom]]:
        """Find paths between two atoms in the knowledge graph."""
        paths = []
        
        def dfs_path(current_atom: Atom, target_atom: Atom, 
                    current_path: List[Atom], depth: int):
            if depth > max_depth:
                return
            
            if current_atom == target_atom:
                paths.append(current_path + [current_atom])
                return
            
            if current_atom in current_path:
                return  # Avoid cycles
            
            # Explore outgoing links
            for out_atom in current_atom.outgoing:
                dfs_path(out_atom, target_atom, current_path + [current_atom], depth + 1)
            
            # Explore incoming links
            incoming = self.atomspace.get_incoming_set(current_atom)
            for in_atom in incoming:
                dfs_path(in_atom, target_atom, current_path + [current_atom], depth + 1)
        
        dfs_path(start_atom, end_atom, [], 0)
        return paths
    
    def set_match_threshold(self, threshold: float):
        """Set the minimum similarity threshold for fuzzy matching."""
        self.match_threshold = max(0.0, min(1.0, threshold))
    
    def set_max_results(self, max_results: int):
        """Set the maximum number of results to return."""
        self.max_results = max(1, max_results)