#!/usr/bin/env python

from .faster import grid_sample as default
from .naive import grid_sample as naive

__all__ = [
    "default",
    "naive",
]
