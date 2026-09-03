"""search: base search space + constraints + priors.

Answers "what is currently searchable?"
    Base Search Space
        ↓
    Constraint / Prior
        ↓
    Current Search Space (= ConstraintResult per state)
"""
from search.constraint import ConstraintResult
from search.space import SearchSpace

__all__ = ["SearchSpace", "ConstraintResult"]
