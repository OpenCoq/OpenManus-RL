"""
OpenCog integration systems for OpenManus-RL.

This module provides OpenCog AtomSpace integration for symbolic reasoning,
pattern matching, and cognitive architectures within the OpenManus-RL framework.
"""

from .atomspace_integration import AtomSpaceManager, Atom, AtomType
from .reasoning_engine import OpenCogReasoningEngine, ReasoningResult, ReasoningMode
from .pattern_matcher import OpenCogPatternMatcher, PatternQuery, MatchType
from .cognitive_architecture import CognitiveAgent, CognitiveState, AttentionMode, CognitiveAction
from .knowledge_representation import KnowledgeGraph, RelationType
from .config import (
    OpenCogConfig, ConfigManager, get_config, load_config, update_config,
    AtomSpaceConfig, ReasoningConfig, CognitiveConfig, PatternMatchingConfig
)

__all__ = [
    'AtomSpaceManager', 'Atom', 'AtomType',
    'OpenCogReasoningEngine', 'ReasoningResult', 'ReasoningMode',
    'OpenCogPatternMatcher', 'PatternQuery', 'MatchType',
    'CognitiveAgent', 'CognitiveState', 'AttentionMode', 'CognitiveAction',
    'KnowledgeGraph', 'RelationType',
    'OpenCogConfig', 'ConfigManager', 'get_config', 'load_config', 'update_config',
    'AtomSpaceConfig', 'ReasoningConfig', 'CognitiveConfig', 'PatternMatchingConfig'
]