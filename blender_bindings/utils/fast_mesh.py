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
                    edges: np.ndarray,
                    faces: np.ndarray,
                    shade_flat=True):
        """
        Make a mesh from a list of vertices/edges/faces
        Until we have a nicer way to make geometry, use this.

        :arg vertices:

           float triplets each representing (X, Y, Z)
           eg: [(0.0, 1.0, 0.5), ...].

        :type vertices: iterable object
        :arg edges:

           int pairs, each pair contains two indices to the
           *vertices* argument. eg: [(1, 2), ...]

           When an empty iterable is passed in, the edges are inferred from the polygons.

        :type edges: iterable object
        :arg faces:

           iterator of faces, each faces contains three or more indices to
           the *vertices* argument. eg: [(5, 6, 8, 9), (1, 2, 3), ...]

        :type faces: iterable object

        .. warning::

           Invalid mesh data
           *(out of range indices, edges with matching indices,
           2 sided faces... etc)* are **not** prevented.
           If the data used for mesh creation isn't known to be valid,
           run :class:`Mesh.validate` after this function.
        """

        has_faces = len(faces) > 0
        has_edges = len(edges) > 0
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
