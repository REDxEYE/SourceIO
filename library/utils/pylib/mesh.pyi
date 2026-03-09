from __future__ import annotations

from typing import Any

"""
SourceIO mesh module
"""
class SMDBoneDef:
    bone: int
    pos: tuple[float, float, float]
    rot: tuple[float, float, float]

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDModel:
    version: int
    node_count: int
    frame_count: int
    triangle_count: int
    skeleton: SMDSkeleton
    nodes: list[SMDNode]
    triangles: list[SMDTriangle]

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDNode:
    id: int
    name: str
    parent: int

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDSkeleton:
    frame_count: int
    frames: dict[int, list[SMDBoneDef]]

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDTriangle:
    material: str
    vertices: tuple[SMDVertex, SMDVertex, SMDVertex]

    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

class SMDVertex:
    pos: tuple[float, float, float]
    normal: tuple[float, float, float]
    uv: tuple[float, float]
    weights: list[tuple[int, float]]

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

