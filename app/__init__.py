"""
Pizza counting application.
This module contains the main application components.
"""

from .config import config, get_config
from .core import (
    BaseTrackedObject,
    BoundingBox,
    Pizza,
    Person,
    PizzaTracker,
    PersonTracker
)

__all__ = [
    'config',
    'get_config',
    'BaseTrackedObject',
    'BoundingBox',
    'Pizza',
    'Person',
    'PizzaTracker',
    'PersonTracker'
] 