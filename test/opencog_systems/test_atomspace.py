"""
Tests for OpenCog AtomSpace integration.
"""

import unittest
from openmanus_rl.opencog_systems.atomspace_integration import (
    AtomSpaceManager, AtomType, Atom
)


class TestAtomSpaceManager(unittest.TestCase):
    """Test cases for AtomSpace manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpaceManager()
    
    def test_create_concept_node(self):
        """Test creation of concept nodes."""
        atom = self.atomspace.create_concept_node("test_concept")
        
        self.assertIsInstance(atom, Atom)
        self.assertEqual(atom.atom_type, AtomType.CONCEPT_NODE)
        self.assertEqual(atom.name, "test_concept")
        self.assertEqual(atom.truth_value, 1.0)
    
    def test_create_predicate_node(self):
        """Test creation of predicate nodes."""
        atom = self.atomspace.create_predicate_node("test_predicate")
        
        self.assertIsInstance(atom, Atom)
        self.assertEqual(atom.atom_type, AtomType.PREDICATE_NODE)
        self.assertEqual(atom.name, "test_predicate")
    
    def test_create_evaluation_link(self):
        """Test creation of evaluation links."""
        predicate = self.atomspace.create_predicate_node("likes")
        arg1 = self.atomspace.create_concept_node("Alice")
        arg2 = self.atomspace.create_concept_node("Bob")
        
        eval_link = self.atomspace.create_evaluation_link(
            predicate, [arg1, arg2], truth_value=0.8
        )
        
        self.assertEqual(eval_link.atom_type, AtomType.EVALUATION_LINK)
        self.assertEqual(eval_link.truth_value, 0.8)
        self.assertEqual(len(eval_link.outgoing), 2)  # predicate and list_link
    
    def test_create_inheritance_link(self):
        """Test creation of inheritance links."""
        child = self.atomspace.create_concept_node("dog")
        parent = self.atomspace.create_concept_node("animal")
        
        inherit_link = self.atomspace.create_inheritance_link(
            child, parent, truth_value=0.9
        )
        
        self.assertEqual(inherit_link.atom_type, AtomType.INHERITANCE_LINK)
        self.assertEqual(inherit_link.truth_value, 0.9)
        self.assertEqual(len(inherit_link.outgoing), 2)
        self.assertEqual(inherit_link.outgoing[0], child)
        self.assertEqual(inherit_link.outgoing[1], parent)
    
    def test_find_atoms(self):
        """Test finding atoms by type and name."""
        # Create some test atoms
        concept1 = self.atomspace.create_concept_node("concept1")
        concept2 = self.atomspace.create_concept_node("concept2")
        predicate1 = self.atomspace.create_predicate_node("predicate1")
        
        # Test finding by type
        concepts = self.atomspace.find_atoms(AtomType.CONCEPT_NODE)
        self.assertIn(concept1, concepts)
        self.assertIn(concept2, concepts)
        self.assertNotIn(predicate1, concepts)
        
        # Test finding by name
        found = self.atomspace.find_atoms(name="concept1")
        self.assertIn(concept1, found)
        self.assertNotIn(concept2, found)
        
        # Test finding by type and name
        found = self.atomspace.find_atoms(AtomType.CONCEPT_NODE, "concept2")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0], concept2)
    
    def test_get_incoming_set(self):
        """Test getting incoming links for an atom."""
        child = self.atomspace.create_concept_node("child")
        parent = self.atomspace.create_concept_node("parent")
        inherit_link = self.atomspace.create_inheritance_link(child, parent)
        
        # Parent should have incoming inheritance link
        incoming_parent = self.atomspace.get_incoming_set(parent)
        self.assertIn(inherit_link, incoming_parent)
        
        # Child should also have incoming link (from the inheritance link)
        incoming_child = self.atomspace.get_incoming_set(child)
        self.assertIn(inherit_link, incoming_child)
    
    def test_export_import(self):
        """Test exporting and importing atomspace data."""
        # Create some test data
        concept = self.atomspace.create_concept_node("test_concept", truth_value=0.8)
        predicate = self.atomspace.create_predicate_node("test_predicate")
        
        # Export
        export_data = self.atomspace.export_to_dict()
        
        # Clear atomspace and import
        self.atomspace.clear()
        self.assertEqual(self.atomspace.size(), 0)
        
        self.atomspace.import_from_dict(export_data)
        
        # Verify data is restored
        self.assertEqual(self.atomspace.size(), 2)
        concepts = self.atomspace.find_atoms(AtomType.CONCEPT_NODE, "test_concept")
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].truth_value, 0.8)
    
    def test_atomspace_size(self):
        """Test atomspace size tracking."""
        initial_size = self.atomspace.size()
        
        self.atomspace.create_concept_node("test1")
        self.assertEqual(self.atomspace.size(), initial_size + 1)
        
        self.atomspace.create_concept_node("test2")
        self.assertEqual(self.atomspace.size(), initial_size + 2)
        
        self.atomspace.clear()
        self.assertEqual(self.atomspace.size(), 0)
    
    def test_remove_atom(self):
        """Test removing atoms from atomspace."""
        atom = self.atomspace.create_concept_node("to_remove")
        atom_id = atom.atom_id
        initial_size = self.atomspace.size()
        
        # Verify atom exists
        self.assertIn(atom_id, self.atomspace)
        
        # Remove atom
        removed = self.atomspace.remove_atom(atom_id)
        self.assertTrue(removed)
        self.assertEqual(self.atomspace.size(), initial_size - 1)
        self.assertNotIn(atom_id, self.atomspace)
        
        # Try to remove non-existent atom
        removed = self.atomspace.remove_atom("non_existent")
        self.assertFalse(removed)


if __name__ == '__main__':
    unittest.main()