"""
Core module for the pizza counting application.
This module contains the base classes, objects, and trackers.
"""

from .base import (
    BaseTrackedObject,
    BoundingBox,
    Positionable,
    Trackable,
    TrackerInterface,
    DetectionStrategy,
    MatchingStrategy
)

from .objects import (
    Pizza,
    Person
)

from .trackers import (
    PizzaTracker,
    PersonTracker
)

__all__ = [
    'BaseTrackedObject',
    'BoundingBox',
    'Positionable',
    'Trackable',
    'TrackerInterface',
    'DetectionStrategy',
    'MatchingStrategy',
    'Pizza',
    'Person',
    'PizzaTracker',
    'PersonTracker'
] 