from __future__ import annotations

from typing import Any

"""
SourceIO mesh module
"""
class SMDBoneDef:
    bone: Any
    pos: Any
    rot: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDModel:
    """
    Parse an SMD/VTA text file. Raises ValueError on malformed input.
    """
    frame_count: Any
    kind: Any
    node_count: Any
    nodes: Any
    skeleton: Any
    triangle_count: Any
    triangles: Any
    version: Any
    vertex_animation: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDNode:
    id: Any
    name: Any
    parent: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDSkeleton:
    frame_count: Any
    frames: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDTriangle:
    material: Any
    vertices: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDVertex:
    normal: Any
    pos: Any
    uv: Any
    weights: Any

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

def decode_index_buffer(input_data: Any, index_size: Any, index_count: Any) -> Any:
    """
    Decode a meshoptimizer-compressed index buffer.
    
    input_data: bytes-like encoded data; index_size: 2 or 4; index_count: number of
    indices (>= 0, multiple of 3).
    Returns bytes of length index_size * index_count.
    Raises TypeError, ValueError (bad sizes or malformed data) or OverflowError.
    """
    ...

def decode_vertex_buffer(input_data: Any, vertex_size: Any, vertex_count: Any) -> Any:
    """
    Decode a meshoptimizer-compressed vertex buffer.
    
    input_data: bytes-like encoded data; vertex_size: bytes per vertex (multiple of 4,
    4..256); vertex_count: number of vertices (>= 0).
    Returns bytes of length vertex_size * vertex_count.
    Raises TypeError, ValueError (bad sizes or malformed data) or OverflowError.
    """
    ...

