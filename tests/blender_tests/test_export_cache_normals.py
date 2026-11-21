"""Export cache fingerprint checks for mesh attributes, run inside Blender's Python."""

import unittest
from types import SimpleNamespace

import bpy
from mathutils import Vector

from SourceIO.blender_bindings.ui.export_nodes.export_cache import geometry_fingerprint


class ExportCacheNormalsTests(unittest.TestCase):
    def setUp(self):
        mesh = bpy.data.meshes.new('cache normals')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], [], [(0, 1, 2, 3)])
        self.addCleanup(bpy.data.meshes.remove, mesh)
        self.obj = bpy.data.objects.new('cache normals', mesh)
        self.addCleanup(bpy.data.objects.remove, self.obj)
        self.geometry = SimpleNamespace(
            key='cache normals', source=self.obj, vertex_animations=(), auto_merge=False, stereo_balance_groups={})

    def fingerprint(self):
        return geometry_fingerprint(self.geometry, None, ())

    def set_normal(self, normal):
        self.obj.data.normals_split_custom_set([Vector(normal)] * len(self.obj.data.loops))

    def test_custom_normals_change_the_fingerprint(self):
        self.set_normal((0, 0, 1))
        self.assertEqual(self.obj.data.attributes['custom_normal'].data_type, 'INT16_2D')
        first = self.fingerprint()
        self.assertIsNotNone(first)
        self.assertEqual(self.fingerprint(), first)

        self.set_normal((1, 0, 0))
        changed = self.fingerprint()
        self.assertNotEqual(changed, first)
        self.assertEqual(self.fingerprint(), changed)

    def test_unknown_attribute_type_is_not_cached(self):
        self.assertIsNotNone(self.fingerprint())
        self.obj.data.attributes.new('label', 'STRING', 'POINT')
        self.assertIsNone(self.fingerprint())


if __name__ == '__main__':
    unittest.main()
