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
    frame_count: Any
    node_count: Any
    nodes: Any
    skeleton: Any
    triangle_count: Any
    triangles: Any
    version: Any

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
    Decode compressed index buffer.
    """
    ...

def decode_vertex_buffer(input_data: Any, vertex_size: Any, vertex_count: Any) -> Any:
    """
    Decode compressed vertex buffer.
    """
    ...

