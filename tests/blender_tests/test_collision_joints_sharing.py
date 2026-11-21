"""A $collisionjoints geometry reused as a $model geometry is refused at planning, run inside Blender's Python."""

import unittest

import bpy

from SourceIO.blender_bindings.ui.export_nodes import register_nodes, unregister_nodes
from SourceIO.blender_bindings.ui.export_nodes.build_graph import GraphBuildError, plan_model


class CollisionJointsSharingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_registration = bpy.types.NodeTree.bl_rna_get_subclass_py('sourceio.model_definition') is None
        if cls.owns_registration:
            register_nodes()

    @classmethod
    def tearDownClass(cls):
        if cls.owns_registration:
            unregister_nodes()

    def test_plan_model_rejects_joint_geometry_that_is_also_a_model(self):
        tree = bpy.data.node_groups.new('Joint sharing test', 'sourceio.model_definition')
        mesh = bpy.data.meshes.new('shared_mesh')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        obj = bpy.data.objects.new('shared_object', mesh)
        bpy.context.scene.collection.objects.link(obj)
        try:
            output = tree.nodes.new('SourceIOPropModelNode')
            output.model_path = 'models/props/test.mdl'
            output.content_root = output.game_root = '/tmp'
            geometry = tree.nodes.new('SourceIOPropGeometryNode')
            geometry.basename = 'shared'
            geometry.obj = obj
            entry = tree.nodes.new('SourceIOModelEntryNode')
            entry.entry_name = 'studio'
            tree.links.new(geometry.outputs['Geometry'], entry.inputs['Geometry'])
            tree.links.new(entry.outputs['Model'], output.inputs['Models'])
            collision = tree.nodes.new('SourceIOPropCollisionNode')
            collision.mode = 'JOINTS'
            tree.links.new(geometry.outputs['Geometry'], collision.inputs['Geometry'])
            tree.links.new(collision.outputs['Collision'], output.inputs['Collision'])
            with self.assertRaisesRegex(GraphBuildError, r"\$collisionjoints geometry 'shared' is also used"):
                plan_model(output)
        finally:
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(mesh)
            bpy.data.node_groups.remove(tree)


if __name__ == '__main__':
    unittest.main()
