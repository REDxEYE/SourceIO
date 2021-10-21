#!/usr/bin/env python3


def linear_interp(v0, v1, d, L):
    """Basic Linear Interpolation"""
    return v0 * (1 - d) / L + v1 * d / L
