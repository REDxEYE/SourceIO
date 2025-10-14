from typing import Any

import bpy
import numpy as np
from bpy.types import Mesh

class FastMesh(Mesh):
    __slots__ = ()

    @classmethod
    def new(cls, name: str) -> 'FastMesh':
        mesh = bpy.data.meshes.new(name)
        mesh.__class__ = cls
        return mesh

    def from_pydata(self,
                    vertices: np.ndarray,
                    edges: Any | None,
                    faces: Any | None,
                    shade_flat=True):

        has_faces = faces is not None and len(faces) > 0
        has_edges = edges is not None and len(edges) > 0
        vertices_len = len(vertices)
        self.vertices.add(vertices_len)

        if has_faces:
            if not isinstance(faces, np.ndarray):
                raise NotImplementedError("FastMesh only works with numpy arrays")
            face_lengths = faces.shape[1]
            faces_len = faces.shape[0]
            self.loops.add(faces_len * face_lengths)
            self.polygons.add(faces_len)
            loop_starts = np.arange(0, faces_len * face_lengths, face_lengths, dtype=np.uint32)
            self.polygons.foreach_set("loop_start", loop_starts)
            self.polygons.foreach_set("vertices", faces.ravel())

        self.vertices.foreach_set("co", vertices.ravel())

        if has_edges:
            if not isinstance(edges, np.ndarray):
                raise NotImplementedError("FastMesh only works with numpy arrays")
            self.edges.add(len(edges))
            self.edges.foreach_set("vertices", edges.ravel())

        if shade_flat:
            self.shade_flat()

        if has_edges or has_faces:
            self.update(
                # Needed to either:
                # - Calculate edges that don't exist for polygons.
                # - Assign edges to polygon loops.
                calc_edges=has_edges,
                # Flag loose edges.
                calc_edges_loose=has_faces,
            )

    def update(self, calc_edges: bool = False, calc_edges_loose: bool = False) -> None:
        return super().update(calc_edges=calc_edges, calc_edges_loose=calc_edges_loose)
