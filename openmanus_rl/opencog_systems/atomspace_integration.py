"""
AtomSpace integration for OpenCog systems in OpenManus-RL.

This module provides a Python interface to OpenCog's AtomSpace for knowledge
representation and symbolic reasoning within RL agents.
"""

import uuid
from typing import Dict, List, Any, Optional, Union, Set
from enum import Enum
import json
import logging
from dataclasses import dataclass


class AtomType(Enum):
    """Basic OpenCog atom types."""
    CONCEPT_NODE = "ConceptNode"
    PREDICATE_NODE = "PredicateNode"
    VARIABLE_NODE = "VariableNode"
    LIST_LINK = "ListLink"
    EVALUATION_LINK = "EvaluationLink"
    INHERITANCE_LINK = "InheritanceLink"
    SIMILARITY_LINK = "SimilarityLink"
    IMPLICATION_LINK = "ImplicationLink"
    AND_LINK = "AndLink"
    OR_LINK = "OrLink"
    NOT_LINK = "NotLink"


@dataclass
class Atom:
    """Represents an OpenCog Atom with type, name, and truth value."""
    atom_type: AtomType
    name: str
    outgoing: Optional[List['Atom']] = None
    truth_value: Optional[float] = None
    confidence: Optional[float] = None
    atom_id: str = None
    
    def __post_init__(self):
        if self.atom_id is None:
            self.atom_id = str(uuid.uuid4())
        if self.outgoing is None:
            self.outgoing = []


class AtomSpaceManager:
    """
    Manages an OpenCog AtomSpace for symbolic knowledge representation.
    
    This is a lightweight Python implementation that can interface with
    the full OpenCog AtomSpace when available, or operate independently.
    """
    
    def __init__(self):
        self.atoms: Dict[str, Atom] = {}
        self.name_to_atoms: Dict[str, Set[str]] = {}
        self.type_to_atoms: Dict[AtomType, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize type mappings
        for atom_type in AtomType:
            self.type_to_atoms[atom_type] = set()
    
    def add_atom(self, atom_type: AtomType, name: str, 
                 outgoing: Optional[List[Atom]] = None,
                 truth_value: float = 1.0,
                 confidence: float = 1.0) -> Atom:
        """
        Add an atom to the AtomSpace.
        
        Args:
            atom_type: Type of the atom to create
            name: Name/identifier for the atom
            outgoing: List of atoms this atom points to (for links)
            truth_value: Truth value (0.0 to 1.0)
            confidence: Confidence in the truth value (0.0 to 1.0)
            
        Returns:
            The created Atom object
        """
        atom = Atom(
            atom_type=atom_type,
            name=name,
            outgoing=outgoing or [],
            truth_value=truth_value,
            confidence=confidence
        )
        
        self.atoms[atom.atom_id] = atom
        
        # Update indexes
        if name not in self.name_to_atoms:
            self.name_to_atoms[name] = set()
        self.name_to_atoms[name].add(atom.atom_id)
        self.type_to_atoms[atom_type].add(atom.atom_id)
        
        self.logger.debug(f"Added atom: {atom_type.value} '{name}' with ID {atom.atom_id}")
        return atom
    
    def find_atoms(self, atom_type: Optional[AtomType] = None,
                   name: Optional[str] = None) -> List[Atom]:
        """
        Find atoms matching the given criteria.
        
        Args:
            atom_type: Filter by atom type
            name: Filter by atom name
            
        Returns:
            List of matching atoms
        """
        candidate_ids = set(self.atoms.keys())
        
        if atom_type is not None:
            candidate_ids &= self.type_to_atoms[atom_type]
        
        if name is not None:
            candidate_ids &= self.name_to_atoms.get(name, set())
        
        return [self.atoms[atom_id] for atom_id in candidate_ids]
    
    def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get an atom by its ID."""
        return self.atoms.get(atom_id)
    
    def remove_atom(self, atom_id: str) -> bool:
        """
        Remove an atom from the AtomSpace.
        
        Args:
            atom_id: ID of the atom to remove
            
        Returns:
            True if atom was removed, False if not found
        """
        if atom_id not in self.atoms:
            return False
        
        atom = self.atoms[atom_id]
        
        # Remove from indexes
        self.name_to_atoms[atom.name].discard(atom_id)
        if not self.name_to_atoms[atom.name]:
            del self.name_to_atoms[atom.name]
        
        self.type_to_atoms[atom.atom_type].discard(atom_id)
        
        # Remove the atom
        del self.atoms[atom_id]
        
        self.logger.debug(f"Removed atom {atom_id}")
        return True
    
    def create_concept_node(self, name: str, truth_value: float = 1.0) -> Atom:
        """Create a ConceptNode."""
        return self.add_atom(AtomType.CONCEPT_NODE, name, truth_value=truth_value)
    
    def create_predicate_node(self, name: str, truth_value: float = 1.0) -> Atom:
        """Create a PredicateNode."""
        return self.add_atom(AtomType.PREDICATE_NODE, name, truth_value=truth_value)
    
    def create_variable_node(self, name: str) -> Atom:
        """Create a VariableNode."""
        return self.add_atom(AtomType.VARIABLE_NODE, name)
    
    def create_evaluation_link(self, predicate: Atom, arguments: List[Atom],
                              truth_value: float = 1.0) -> Atom:
        """
        Create an EvaluationLink.
        
        Args:
            predicate: PredicateNode to evaluate
            arguments: List of atoms as arguments
            truth_value: Truth value of the evaluation
            
        Returns:
            Created EvaluationLink atom
        """
        # Create a ListLink for the arguments
        list_link = self.add_atom(
            AtomType.LIST_LINK,
            f"list_{uuid.uuid4().hex[:8]}",
            outgoing=arguments
        )
        
        # Create the EvaluationLink
        return self.add_atom(
            AtomType.EVALUATION_LINK,
            f"eval_{uuid.uuid4().hex[:8]}",
            outgoing=[predicate, list_link],
            truth_value=truth_value
        )
    
    def create_inheritance_link(self, child: Atom, parent: Atom,
                               truth_value: float = 1.0) -> Atom:
        """Create an InheritanceLink."""
        return self.add_atom(
            AtomType.INHERITANCE_LINK,
            f"inherit_{child.name}_{parent.name}",
            outgoing=[child, parent],
            truth_value=truth_value
        )
    
    def get_incoming_set(self, atom: Atom) -> List[Atom]:
        """
        Get all atoms that have the given atom in their outgoing set.
        
        Args:
            atom: The atom to find incoming links for
            
        Returns:
            List of atoms that point to the given atom
        """
        incoming = []
        for candidate in self.atoms.values():
            if atom in candidate.outgoing:
                incoming.append(candidate)
        return incoming
    
    def get_outgoing_set(self, atom: Atom) -> List[Atom]:
        """Get the outgoing set of an atom."""
        return atom.outgoing.copy()
    
    def query_pattern(self, pattern: Dict[str, Any]) -> List[Dict[str, Atom]]:
        """
        Simple pattern matching query.
        
        Args:
            pattern: Dictionary describing the pattern to match
            
        Returns:
            List of variable bindings that satisfy the pattern
        """
        # This is a simplified pattern matcher
        # A full implementation would use OpenCog's pattern matcher
        results = []
        
        if pattern.get("type") == "evaluation":
            predicate_name = pattern.get("predicate")
            arg_patterns = pattern.get("arguments", [])
            
            # Find all EvaluationLinks
            eval_links = self.find_atoms(AtomType.EVALUATION_LINK)
            
            for eval_link in eval_links:
                if len(eval_link.outgoing) >= 2:
                    pred_atom = eval_link.outgoing[0]
                    if pred_atom.name == predicate_name:
                        # Simple match - can be expanded
                        bindings = {"evaluation": eval_link}
                        results.append(bindings)
        
        return results
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the AtomSpace to a dictionary for serialization."""
        export_data = {
            "atoms": {},
            "metadata": {
                "total_atoms": len(self.atoms),
                "atom_types": {atom_type.value: len(ids) 
                              for atom_type, ids in self.type_to_atoms.items()}
            }
        }
        
        for atom_id, atom in self.atoms.items():
            export_data["atoms"][atom_id] = {
                "type": atom.atom_type.value,
                "name": atom.name,
                "outgoing": [out_atom.atom_id for out_atom in atom.outgoing],
                "truth_value": atom.truth_value,
                "confidence": atom.confidence
            }
        
        return export_data
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import AtomSpace data from a dictionary."""
        self.clear()
        
        atoms_data = data.get("atoms", {})
        
        # First pass: create all atoms without outgoing sets
        atom_id_map = {}
        for atom_id, atom_data in atoms_data.items():
            atom_type = AtomType(atom_data["type"])
            atom = Atom(
                atom_type=atom_type,
                name=atom_data["name"],
                truth_value=atom_data.get("truth_value"),
                confidence=atom_data.get("confidence"),
                atom_id=atom_id
            )
            self.atoms[atom_id] = atom
            atom_id_map[atom_id] = atom
            
            # Update indexes
            if atom.name not in self.name_to_atoms:
                self.name_to_atoms[atom.name] = set()
            self.name_to_atoms[atom.name].add(atom_id)
            self.type_to_atoms[atom_type].add(atom_id)
        
        # Second pass: set up outgoing relationships
        for atom_id, atom_data in atoms_data.items():
            atom = self.atoms[atom_id]
            outgoing_ids = atom_data.get("outgoing", [])
            atom.outgoing = [atom_id_map[out_id] for out_id in outgoing_ids 
                           if out_id in atom_id_map]
    
    def clear(self) -> None:
        """Clear all atoms from the AtomSpace."""
        self.atoms.clear()
        self.name_to_atoms.clear()
        for atom_type in AtomType:
            self.type_to_atoms[atom_type].clear()
    
    def size(self) -> int:
        """Get the number of atoms in the AtomSpace."""
        return len(self.atoms)
    
    def __len__(self) -> int:
        return self.size()
    
    def __contains__(self, atom_id: str) -> bool:
        return atom_id in self.atoms