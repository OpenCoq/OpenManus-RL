"""
Knowledge Representation module using OpenCog for OpenManus-RL.

This module provides high-level knowledge representation capabilities
including knowledge graphs, ontologies, and semantic networks.
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .atomspace_integration import AtomSpaceManager, Atom, AtomType


class RelationType(Enum):
    """Types of relationships in the knowledge graph."""
    INHERITANCE = "inheritance"
    SIMILARITY = "similarity"
    CAUSATION = "causation"
    ASSOCIATION = "association"
    COMPOSITION = "composition"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    atom: Optional[Atom] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    atom: Optional[Atom] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


class KnowledgeGraph:
    """
    Knowledge Graph implementation using OpenCog AtomSpace.
    
    Provides high-level operations for building and querying
    semantic knowledge representations.
    """
    
    def __init__(self, atomspace: AtomSpaceManager):
        self.atomspace = atomspace
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.logger = logging.getLogger(__name__)
        
        # Indexes for efficient querying
        self.entity_by_name: Dict[str, str] = {}
        self.entity_by_type: Dict[str, Set[str]] = {}
        self.relationships_by_source: Dict[str, Set[str]] = {}
        self.relationships_by_target: Dict[str, Set[str]] = {}
        
        # Initialize ontology structure
        self._initialize_ontology()
    
    def _initialize_ontology(self):
        """Initialize basic ontological concepts."""
        
        # Basic top-level concepts
        self.add_entity("Entity", {"type": "ontology_class", "level": 0})
        self.add_entity("Relationship", {"type": "ontology_class", "level": 0}) 
        self.add_entity("Property", {"type": "ontology_class", "level": 0})
        
        # Agent-related concepts
        self.add_entity("Agent", {"type": "ontology_class", "parent": "Entity"})
        self.add_entity("Action", {"type": "ontology_class", "parent": "Entity"})
        self.add_entity("Goal", {"type": "ontology_class", "parent": "Entity"})
        self.add_entity("State", {"type": "ontology_class", "parent": "Entity"})
        
        # Environment concepts
        self.add_entity("Environment", {"type": "ontology_class", "parent": "Entity"})
        self.add_entity("Object", {"type": "ontology_class", "parent": "Entity"})
        self.add_entity("Location", {"type": "ontology_class", "parent": "Entity"})
        
        # Create inheritance relationships
        self._create_inheritance_relationships()
        
        self.logger.info("Initialized knowledge graph ontology")
    
    def _create_inheritance_relationships(self):
        """Create basic inheritance relationships in the ontology."""
        inheritance_pairs = [
            ("Agent", "Entity"),
            ("Action", "Entity"), 
            ("Goal", "Entity"),
            ("State", "Entity"),
            ("Environment", "Entity"),
            ("Object", "Entity"),
            ("Location", "Entity")
        ]
        
        for child, parent in inheritance_pairs:
            if child in self.entity_by_name and parent in self.entity_by_name:
                child_id = self.entity_by_name[child]
                parent_id = self.entity_by_name[parent]
                self.add_relationship(child_id, parent_id, RelationType.INHERITANCE)
    
    def add_entity(self, name: str, properties: Dict[str, Any] = None,
                   entity_type: str = "concept") -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            name: Name of the entity
            properties: Dictionary of entity properties
            entity_type: Type classification of the entity
            
        Returns:
            Entity ID
        """
        properties = properties or {}
        
        # Create entity
        entity = Entity(
            id=str(uuid.uuid4()),
            name=name,
            entity_type=entity_type,
            properties=properties
        )
        
        # Create corresponding atom in atomspace
        atom = self.atomspace.create_concept_node(name)
        entity.atom = atom
        
        # Store entity
        self.entities[entity.id] = entity
        
        # Update indexes
        self.entity_by_name[name] = entity.id
        if entity_type not in self.entity_by_type:
            self.entity_by_type[entity_type] = set()
        self.entity_by_type[entity_type].add(entity.id)
        
        # Add properties as evaluation links
        for prop_name, prop_value in properties.items():
            self._add_property_atom(entity, prop_name, prop_value)
        
        self.logger.debug(f"Added entity: {name} ({entity_type})")
        return entity.id
    
    def add_relationship(self, source_id: str, target_id: str, 
                        relation_type: RelationType,
                        strength: float = 1.0,
                        properties: Dict[str, Any] = None) -> str:
        """
        Add a relationship between two entities.
        
        Args:
            source_id: ID of source entity
            target_id: ID of target entity
            relation_type: Type of relationship
            strength: Strength of the relationship (0.0 to 1.0)
            properties: Additional properties of the relationship
            
        Returns:
            Relationship ID
        """
        if source_id not in self.entities or target_id not in self.entities:
            raise ValueError("Source or target entity not found")
        
        properties = properties or {}
        
        # Create relationship
        relationship = Relationship(
            id=str(uuid.uuid4()),
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            properties=properties
        )
        
        # Create corresponding atom in atomspace
        source_atom = self.entities[source_id].atom
        target_atom = self.entities[target_id].atom
        
        if relation_type == RelationType.INHERITANCE:
            atom = self.atomspace.create_inheritance_link(
                source_atom, target_atom, strength
            )
        elif relation_type == RelationType.SIMILARITY:
            atom = self.atomspace.add_atom(
                AtomType.SIMILARITY_LINK,
                f"similarity_{source_id}_{target_id}",
                outgoing=[source_atom, target_atom],
                truth_value=strength
            )
        else:
            # Generic relationship as evaluation link
            predicate = self.atomspace.create_predicate_node(relation_type.value)
            atom = self.atomspace.create_evaluation_link(
                predicate, [source_atom, target_atom], strength
            )
        
        relationship.atom = atom
        
        # Store relationship
        self.relationships[relationship.id] = relationship
        
        # Update indexes
        if source_id not in self.relationships_by_source:
            self.relationships_by_source[source_id] = set()
        if target_id not in self.relationships_by_target:
            self.relationships_by_target[target_id] = set()
        
        self.relationships_by_source[source_id].add(relationship.id)
        self.relationships_by_target[target_id].add(relationship.id)
        
        self.logger.debug(f"Added relationship: {relation_type.value} "
                         f"({self.entities[source_id].name} -> {self.entities[target_id].name})")
        return relationship.id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name."""
        entity_id = self.entity_by_name.get(name)
        return self.entities.get(entity_id) if entity_id else None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.entity_by_type.get(entity_type, set())
        return [self.entities[entity_id] for entity_id in entity_ids]
    
    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Get a relationship by ID."""
        return self.relationships.get(relationship_id)
    
    def get_outgoing_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships where the entity is the source."""
        rel_ids = self.relationships_by_source.get(entity_id, set())
        return [self.relationships[rel_id] for rel_id in rel_ids]
    
    def get_incoming_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships where the entity is the target."""
        rel_ids = self.relationships_by_target.get(entity_id, set())
        return [self.relationships[rel_id] for rel_id in rel_ids]
    
    def find_path(self, source_id: str, target_id: str,
                  max_depth: int = 3) -> List[List[str]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            max_depth: Maximum path length
            
        Returns:
            List of paths, where each path is a list of entity IDs
        """
        paths = []
        visited = set()
        
        def dfs(current_id: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current_id == target_id:
                paths.append(path + [current_id])
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            # Explore outgoing relationships
            for rel in self.get_outgoing_relationships(current_id):
                dfs(rel.target_id, path + [current_id], depth + 1)
            
            visited.remove(current_id)
        
        dfs(source_id, [], 0)
        return paths
    
    def find_related_entities(self, entity_id: str, 
                             relation_types: List[RelationType] = None,
                             max_distance: int = 2) -> List[Tuple[Entity, float]]:
        """
        Find entities related to the given entity.
        
        Args:
            entity_id: Entity to find relations for
            relation_types: Filter by specific relation types
            max_distance: Maximum relationship distance
            
        Returns:
            List of (entity, distance) tuples
        """
        if entity_id not in self.entities:
            return []
        
        related = {}
        queue = [(entity_id, 0)]
        visited = set()
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if current_id in visited or distance > max_distance:
                continue
            
            visited.add(current_id)
            
            # Get all relationships
            outgoing = self.get_outgoing_relationships(current_id)
            incoming = self.get_incoming_relationships(current_id)
            
            for rel in outgoing + incoming:
                # Filter by relation type if specified
                if relation_types and rel.relation_type not in relation_types:
                    continue
                
                # Get the other entity
                other_id = rel.target_id if rel.source_id == current_id else rel.source_id
                
                if other_id != entity_id and other_id not in visited:
                    # Calculate weighted distance
                    weighted_distance = distance + (1.0 - rel.strength)
                    
                    if other_id not in related or weighted_distance < related[other_id]:
                        related[other_id] = weighted_distance
                        queue.append((other_id, distance + 1))
        
        # Convert to list of (entity, distance) tuples
        result = [(self.entities[eid], dist) for eid, dist in related.items()]
        result.sort(key=lambda x: x[1])  # Sort by distance
        
        return result
    
    def query_entities(self, **criteria) -> List[Entity]:
        """
        Query entities based on various criteria.
        
        Args:
            **criteria: Query criteria (name, type, properties, etc.)
            
        Returns:
            List of matching entities
        """
        candidates = list(self.entities.values())
        
        # Filter by name
        if 'name' in criteria:
            name_filter = criteria['name']
            if isinstance(name_filter, str):
                candidates = [e for e in candidates if name_filter.lower() in e.name.lower()]
        
        # Filter by type
        if 'type' in criteria:
            type_filter = criteria['type']
            candidates = [e for e in candidates if e.entity_type == type_filter]
        
        # Filter by properties
        for key, value in criteria.items():
            if key not in ['name', 'type']:
                candidates = [e for e in candidates 
                            if key in e.properties and e.properties[key] == value]
        
        return candidates
    
    def get_entity_neighborhood(self, entity_id: str, 
                              radius: int = 1) -> Dict[str, Any]:
        """
        Get the neighborhood of an entity (nearby entities and relationships).
        
        Args:
            entity_id: Entity to get neighborhood for
            radius: Neighborhood radius
            
        Returns:
            Dictionary with neighborhood information
        """
        if entity_id not in self.entities:
            return {}
        
        neighborhood = {
            "center_entity": self.entities[entity_id],
            "entities": {},
            "relationships": []
        }
        
        # Get related entities within radius
        related = self.find_related_entities(entity_id, max_distance=radius)
        
        for entity, distance in related:
            neighborhood["entities"][entity.id] = {
                "entity": entity,
                "distance": distance
            }
        
        # Get all relationships in the neighborhood
        all_entity_ids = {entity_id} | set(neighborhood["entities"].keys())
        
        for eid in all_entity_ids:
            for rel in self.get_outgoing_relationships(eid):
                if rel.target_id in all_entity_ids:
                    neighborhood["relationships"].append(rel)
        
        return neighborhood
    
    def add_temporal_relationship(self, event1_id: str, event2_id: str,
                                 temporal_type: str = "before",
                                 time_distance: float = 1.0) -> str:
        """
        Add a temporal relationship between two events/entities.
        
        Args:
            event1_id: First event entity ID
            event2_id: Second event entity ID
            temporal_type: Type of temporal relation (before, after, during, etc.)
            time_distance: Temporal distance measure
            
        Returns:
            Relationship ID
        """
        properties = {
            "temporal_type": temporal_type,
            "time_distance": time_distance
        }
        
        return self.add_relationship(
            event1_id, event2_id,
            RelationType.TEMPORAL,
            strength=1.0 - min(0.9, time_distance / 10.0),  # Closer in time = stronger
            properties=properties
        )
    
    def add_causal_relationship(self, cause_id: str, effect_id: str,
                               causation_strength: float = 0.8,
                               evidence: List[str] = None) -> str:
        """
        Add a causal relationship between entities.
        
        Args:
            cause_id: Cause entity ID
            effect_id: Effect entity ID
            causation_strength: Strength of causal relationship
            evidence: List of evidence supporting the causation
            
        Returns:
            Relationship ID
        """
        properties = {
            "causation_type": "direct",
            "evidence": evidence or []
        }
        
        return self.add_relationship(
            cause_id, effect_id,
            RelationType.CAUSATION,
            strength=causation_strength,
            properties=properties
        )
    
    def infer_relationships(self) -> List[str]:
        """
        Infer new relationships based on existing knowledge.
        
        Returns:
            List of new relationship IDs
        """
        new_relationships = []
        
        # Transitivity inference for inheritance
        for entity_id in self.entities:
            parents = []
            for rel in self.get_outgoing_relationships(entity_id):
                if rel.relation_type == RelationType.INHERITANCE:
                    parents.append(rel.target_id)
            
            # Find grandparents
            for parent_id in parents:
                for rel in self.get_outgoing_relationships(parent_id):
                    if rel.relation_type == RelationType.INHERITANCE:
                        grandparent_id = rel.target_id
                        
                        # Check if direct relationship already exists
                        existing = any(
                            r.target_id == grandparent_id and r.relation_type == RelationType.INHERITANCE
                            for r in self.get_outgoing_relationships(entity_id)
                        )
                        
                        if not existing:
                            # Infer transitive inheritance
                            new_rel_id = self.add_relationship(
                                entity_id, grandparent_id,
                                RelationType.INHERITANCE,
                                strength=0.6  # Lower strength for inferred relationships
                            )
                            new_relationships.append(new_rel_id)
        
        # Similarity symmetry inference
        for rel in self.relationships.values():
            if rel.relation_type == RelationType.SIMILARITY:
                # Check if reverse relationship exists
                reverse_exists = any(
                    r.source_id == rel.target_id and r.target_id == rel.source_id
                    and r.relation_type == RelationType.SIMILARITY
                    for r in self.relationships.values()
                )
                
                if not reverse_exists:
                    new_rel_id = self.add_relationship(
                        rel.target_id, rel.source_id,
                        RelationType.SIMILARITY,
                        strength=rel.strength
                    )
                    new_relationships.append(new_rel_id)
        
        self.logger.info(f"Inferred {len(new_relationships)} new relationships")
        return new_relationships
    
    def _add_property_atom(self, entity: Entity, prop_name: str, prop_value: Any):
        """Add a property as an evaluation link in the atomspace."""
        prop_predicate = self.atomspace.create_predicate_node(f"has_{prop_name}")
        value_atom = self.atomspace.create_concept_node(str(prop_value))
        
        self.atomspace.create_evaluation_link(
            prop_predicate,
            [entity.atom, value_atom],
            truth_value=0.9
        )
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the knowledge graph to a dictionary."""
        export_data = {
            "entities": {},
            "relationships": {},
            "metadata": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "entity_types": list(self.entity_by_type.keys())
            }
        }
        
        # Export entities
        for entity_id, entity in self.entities.items():
            export_data["entities"][entity_id] = {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type,
                "properties": entity.properties
            }
        
        # Export relationships
        for rel_id, rel in self.relationships.items():
            export_data["relationships"][rel_id] = {
                "id": rel.id,
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "relation_type": rel.relation_type.value,
                "strength": rel.strength,
                "properties": rel.properties
            }
        
        return export_data
    
    def import_from_dict(self, data: Dict[str, Any]):
        """Import knowledge graph from a dictionary."""
        # Clear existing data
        self.entities.clear()
        self.relationships.clear()
        self.entity_by_name.clear()
        self.entity_by_type.clear()
        self.relationships_by_source.clear()
        self.relationships_by_target.clear()
        
        # Import entities
        entities_data = data.get("entities", {})
        for entity_data in entities_data.values():
            entity_id = self.add_entity(
                entity_data["name"],
                entity_data.get("properties", {}),
                entity_data.get("type", "concept")
            )
            # Update the ID to match imported data
            old_id = self.entity_by_name[entity_data["name"]]
            if old_id != entity_data["id"]:
                # Remap the entity
                entity = self.entities.pop(old_id)
                entity.id = entity_data["id"]
                self.entities[entity_data["id"]] = entity
                self.entity_by_name[entity_data["name"]] = entity_data["id"]
        
        # Import relationships
        relationships_data = data.get("relationships", {})
        for rel_data in relationships_data.values():
            self.add_relationship(
                rel_data["source_id"],
                rel_data["target_id"],
                RelationType(rel_data["relation_type"]),
                rel_data.get("strength", 1.0),
                rel_data.get("properties", {})
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        entity_type_counts = {etype: len(eids) 
                             for etype, eids in self.entity_by_type.items()}
        
        relation_type_counts = {}
        for rel in self.relationships.values():
            rtype = rel.relation_type.value
            relation_type_counts[rtype] = relation_type_counts.get(rtype, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_type_counts,
            "relation_types": relation_type_counts,
            "atomspace_size": self.atomspace.size()
        }