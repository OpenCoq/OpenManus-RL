"""
Configuration settings for OpenCog systems in OpenManus-RL.

This module provides configuration management for OpenCog components
including AtomSpace, reasoning engines, and cognitive architectures.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import os
import json
import yaml


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    FORWARD_CHAINING = "forward_chaining"
    BACKWARD_CHAINING = "backward_chaining"
    HYBRID = "hybrid"
    PROBABILISTIC = "probabilistic"


class AttentionStrategy(Enum):
    """Available attention allocation strategies."""
    FOCUSED = "focused"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"
    GOAL_DIRECTED = "goal_directed"


@dataclass
class AtomSpaceConfig:
    """Configuration for AtomSpace."""
    initial_size_limit: int = 100000
    garbage_collection_threshold: float = 0.8
    truth_value_precision: float = 0.001
    confidence_threshold: float = 0.1
    enable_persistence: bool = False
    persistence_file: Optional[str] = None
    
    def __post_init__(self):
        if self.enable_persistence and not self.persistence_file:
            self.persistence_file = "/tmp/atomspace_dump.json"


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine."""
    default_strategy: ReasoningStrategy = ReasoningStrategy.HYBRID
    max_iterations: int = 10
    max_depth: int = 5
    confidence_threshold: float = 0.5
    rule_strength_decay: float = 0.9
    enable_explanation: bool = True
    cache_results: bool = True
    cache_size_limit: int = 1000
    
    # Forward chaining parameters
    forward_chaining_enabled: bool = True
    forward_max_iterations: int = 5
    
    # Backward chaining parameters
    backward_chaining_enabled: bool = True
    backward_max_depth: int = 3
    
    # Probabilistic reasoning parameters
    probabilistic_enabled: bool = False
    prior_probability: float = 0.1
    evidence_weight: float = 0.8


@dataclass
class PatternMatchingConfig:
    """Configuration for pattern matching."""
    match_threshold: float = 0.7
    max_results: int = 100
    enable_fuzzy_matching: bool = True
    enable_structural_matching: bool = True
    enable_semantic_matching: bool = False
    
    # Fuzzy matching parameters
    fuzzy_string_similarity_threshold: float = 0.6
    fuzzy_type_weight: float = 0.3
    fuzzy_name_weight: float = 0.4
    fuzzy_structure_weight: float = 0.3
    
    # Structural matching parameters
    structural_arity_weight: float = 0.4
    structural_type_weight: float = 0.6
    
    # Semantic matching parameters (requires embeddings)
    semantic_embedding_model: Optional[str] = None
    semantic_similarity_threshold: float = 0.8


@dataclass
class CognitiveConfig:
    """Configuration for cognitive architecture."""
    attention_strategy: AttentionStrategy = AttentionStrategy.ADAPTIVE
    working_memory_size: int = 20
    experience_history_size: int = 100
    learning_rate: float = 0.1
    exploration_factor: float = 0.2
    
    # Cognitive cycle parameters
    max_cycle_time: float = 1.0  # seconds
    enable_parallel_processing: bool = False
    
    # Memory management
    memory_consolidation_threshold: int = 50
    memory_forgetting_rate: float = 0.01
    
    # Goal management
    max_active_goals: int = 5
    goal_priority_decay: float = 0.95
    
    # Action selection
    action_selection_strategy: str = "confidence_based"
    action_confidence_threshold: float = 0.3


@dataclass
class KnowledgeGraphConfig:
    """Configuration for knowledge representation."""
    max_entities: int = 10000
    max_relationships: int = 50000
    relationship_strength_threshold: float = 0.1
    enable_inference: bool = True
    inference_iterations: int = 3
    
    # Entity management
    entity_similarity_threshold: float = 0.9
    enable_entity_merging: bool = False
    
    # Relationship management
    relationship_decay_rate: float = 0.001
    enable_relationship_pruning: bool = True
    
    # Ontology parameters
    max_ontology_depth: int = 10
    enable_ontology_validation: bool = True


@dataclass
class OpenCogConfig:
    """Main configuration container for all OpenCog systems."""
    atomspace: AtomSpaceConfig = field(default_factory=AtomSpaceConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    pattern_matching: PatternMatchingConfig = field(default_factory=PatternMatchingConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    
    # Global settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = False
    metrics_output_file: Optional[str] = None
    
    # Integration settings
    integration_mode: str = "standalone"  # "standalone" or "openmanus_native"
    enable_real_opencog: bool = False  # Use real OpenCog if available
    opencog_config_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {field: convert_dataclass(getattr(obj, field)) 
                       for field in obj.__dataclass_fields__}
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj
        
        return convert_dataclass(self)
    
    def save_to_file(self, filepath: str, format: str = "yaml"):
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            if format.lower() == "yaml":
                yaml.dump(config_dict, f, indent=2, default_flow_style=False)
            elif format.lower() == "json":
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'OpenCogConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            elif filepath.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OpenCogConfig':
        """Create configuration from dictionary."""
        # This is a simplified version - a full implementation would
        # recursively construct all nested dataclass objects
        config = cls()
        
        # Update top-level fields
        for key, value in data.items():
            if hasattr(config, key) and not key.startswith('_'):
                setattr(config, key, value)
        
        return config
    
    def merge_with(self, other: 'OpenCogConfig') -> 'OpenCogConfig':
        """Merge this configuration with another, with other taking precedence."""
        # Simplified merge - a full implementation would handle nested objects
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged_dict, other_dict)
        return self.from_dict(merged_dict)


class ConfigManager:
    """Manager for OpenCog configuration."""
    
    def __init__(self):
        self.config = OpenCogConfig()
        self._config_file_path = None
    
    def load_config(self, config_path: Optional[str] = None) -> OpenCogConfig:
        """
        Load configuration from file or environment.
        
        Args:
            config_path: Path to configuration file, or None to use environment
            
        Returns:
            Loaded configuration
        """
        if config_path:
            self.config = OpenCogConfig.load_from_file(config_path)
            self._config_file_path = config_path
        else:
            # Try to load from environment variables or default locations
            env_config_path = os.getenv('OPENMANUS_OPENCOG_CONFIG')
            
            default_paths = [
                './opencog_config.yaml',
                './config/opencog.yaml',
                '~/.openmanus/opencog_config.yaml',
                '/etc/openmanus/opencog_config.yaml'
            ]
            
            config_path = env_config_path
            if not config_path:
                for path in default_paths:
                    expanded_path = os.path.expanduser(path)
                    if os.path.exists(expanded_path):
                        config_path = expanded_path
                        break
            
            if config_path and os.path.exists(config_path):
                self.config = OpenCogConfig.load_from_file(config_path)
                self._config_file_path = config_path
            else:
                # Use default configuration
                self.config = OpenCogConfig()
        
        # Override with environment variables
        self._override_from_environment()
        
        return self.config
    
    def _override_from_environment(self):
        """Override configuration with environment variables."""
        env_mappings = {
            'OPENMANUS_OPENCOG_LOG_LEVEL': ('log_level', str),
            'OPENMANUS_OPENCOG_ENABLE_REAL': ('enable_real_opencog', lambda x: x.lower() == 'true'),
            'OPENMANUS_REASONING_MAX_DEPTH': ('reasoning.max_depth', int),
            'OPENMANUS_COGNITIVE_LEARNING_RATE': ('cognitive.learning_rate', float),
            'OPENMANUS_ATOMSPACE_SIZE_LIMIT': ('atomspace.initial_size_limit', int),
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_attr(self.config, config_path, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value} ({e})")
    
    def _set_nested_attr(self, obj, attr_path: str, value):
        """Set nested attribute using dot notation."""
        parts = attr_path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file."""
        path = config_path or self._config_file_path or './opencog_config.yaml'
        self.config.save_to_file(path)
    
    def get_config(self) -> OpenCogConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = OpenCogConfig()


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> OpenCogConfig:
    """Get the global OpenCog configuration."""
    return config_manager.get_config()


def load_config(config_path: Optional[str] = None) -> OpenCogConfig:
    """Load OpenCog configuration from file or environment."""
    return config_manager.load_config(config_path)


def update_config(**kwargs):
    """Update global configuration parameters."""
    config_manager.update_config(**kwargs)


# Default configuration instance for easy access
default_config = OpenCogConfig()