"""Network-level machinery shared by the simulation and the assessment.

``options``      hydraulic option pinning -- the single source of truth for
                 units, headloss, time and demand model.
``partition``    district validation, inter-district link inventory, storage
                 ownership, and the drift seed-node picker.
``profile``      network-scoped parameters: the keys parameters.yml leaves
                 null because their value is a fact about the network.
``conditioning`` arbitrary ``.inp`` -> simulation-ready model + a recipe.
``capacity``     how much demand a network can carry (lambda*, max_demand_lps).
"""
from . import capacity, conditioning, options, partition, profile

__all__ = ["capacity", "conditioning", "options", "partition", "profile"]
