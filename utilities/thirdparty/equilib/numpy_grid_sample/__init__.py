#!/usr/bin/env python

from .naive import grid_sample as naive
from .faster import grid_sample as default

__all__ = [
    "default",
    "naive",
]
