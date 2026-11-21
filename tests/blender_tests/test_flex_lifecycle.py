"""Flex rules node lifecycle checks (delete, copy, shared slider host, group creation), run inside Blender's Python."""

import unittest

import bpy

from SourceIO.blender_bindings.ui.export_nodes import register_nodes, unregister_nodes
from SourceIO.blender_bindings.ui.export_nodes.flex_graph import repair_flex_groups


class FlexLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_registration = bpy.types.NodeTree.bl_rna_get_subclass_py('sourceio.model_definition') is None
        if cls.owns_registration:
            register_nodes()
        if not hasattr(bpy.types.Object, 'flex_controllers'):
            from SourceIO.blender_bindings.operators.flex_operators import SourceIO_PG_FlexController
            bpy.utils.register_class(SourceIO_PG_FlexController)
            bpy.types.Object.flex_controllers = bpy.props.CollectionProperty(type=SourceIO_PG_FlexController)
            cls.flex_property = SourceIO_PG_FlexController
        else:
            cls.flex_property = None

    @classmethod
    def tearDownClass(cls):
        if cls.flex_property is not None:
            del bpy.types.Object.flex_controllers
            bpy.utils.unregister_class(cls.flex_property)
        if cls.owns_registration:
            unregister_nodes()

    def setUp(self):
        self.tree = bpy.data.node_groups.new('Flex lifecycle test', 'sourceio.model_definition')
        self.addCleanup(bpy.data.node_groups.remove, self.tree)
        mesh = bpy.data.meshes.new('face')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        self.addCleanup(bpy.data.meshes.remove, mesh)
        self.face = bpy.data.objects.new('face', mesh)
        self.addCleanup(bpy.data.objects.remove, self.face)
        bpy.context.scene.collection.objects.link(self.face)
        self.face.shape_key_add(name='Basis')
        self.face.shape_key_add(name='smile')

    def collector(self):
        nodes, links = self.tree.nodes, self.tree.links
        collector = nodes.new('SourceIOFlexRulesNode')
        control = nodes.new('SourceIOFlexControllerNode')
        control.control_name = 'smile'
        rule = nodes.new('SourceIOFlexShapeRuleNode')
        rule.obj, rule.shape_name = self.face, 'smile'
        links.new(control.outputs['Controller'], collector.inputs['Controllers'])
        links.new(control.outputs['Value'], rule.inputs['Value'])
        links.new(rule.outputs['Rule'], collector.inputs['Rules'])
        return collector

    def driver_count(self):
        animation = self.face.data.shape_keys.animation_data
        return len(animation.drivers) if animation else 0

    def test_deleting_the_node_removes_its_drivers_and_sliders(self):
        collector = self.collector()
        collector.use_drivers = True
        self.assertEqual(collector.driver_error, '')
        self.assertEqual(self.driver_count(), 1)
        self.tree.nodes.remove(collector)
        self.assertEqual(self.driver_count(), 0)
        self.assertEqual(len(self.face.flex_controllers), 0)
        self.assertNotIn('sourceio_flex_preview', self.face)

    def test_copied_tree_node_starts_with_drivers_off_and_leaves_the_original_alone(self):
        collector = self.collector()
        collector.use_drivers = True
        original_state = collector.driver_state
        duplicate = self.tree.copy()
        self.addCleanup(bpy.data.node_groups.remove, duplicate)
        copied = next(node for node in duplicate.nodes if node.bl_idname == 'SourceIOFlexRulesNode')
        self.assertFalse(copied.use_drivers)
        self.assertEqual(copied.driver_state, '')
        duplicate.nodes.remove(copied)
        self.assertEqual(self.driver_count(), 1)
        self.assertEqual(collector.driver_state, original_state)
        self.assertIn('sourceio_flex_preview', self.face)

    def test_second_node_on_the_same_slider_object_is_refused_even_without_sliders(self):
        first = self.collector()
        first.use_drivers = True
        first.driver_object = self.face
        first.driver_state = ''  # first node lost its record: marker stays, sliders list emptied below
        self.face.flex_controllers.clear()
        self.assertIn('sourceio_flex_preview', self.face)
        second = self.collector()
        second.driver_object = self.face
        second.use_drivers = True
        self.assertFalse(second.use_drivers)
        self.assertIn('already has flex controller sliders', second.driver_error)
        self.assertIn('sourceio_flex_preview', self.face)  # the refused node did not strip the first node's marker

    def test_refreshing_a_node_still_rebuilds_on_its_own_host(self):
        collector = self.collector()
        collector.use_drivers = True
        from SourceIO.blender_bindings.ui.export_nodes.flex_drivers import apply_node_drivers
        apply_node_drivers(collector)
        self.assertEqual(collector.driver_error, '')
        self.assertTrue(collector.use_drivers)
        self.assertEqual(self.driver_count(), 1)

    def test_new_flex_group_has_one_output_and_one_inner_tree(self):
        trees = {tree.name for tree in bpy.data.node_groups}
        group = self.tree.nodes.new('SourceIOFlexGroupNode')
        self.addCleanup(lambda: bpy.data.node_groups.remove(group.node_tree) if group.node_tree else None)
        new_trees = [tree for tree in bpy.data.node_groups if tree.name not in trees]
        self.assertEqual(len(new_trees), 1, [tree.name for tree in new_trees])
        self.assertIs(group.node_tree, new_trees[0])
        self.assertEqual([socket.name for socket in group.outputs], ['Flex Rules'])

    def test_repair_recreates_a_deleted_inner_tree_only_for_nodes_with_an_output(self):
        group = self.tree.nodes.new('SourceIOFlexGroupNode')
        inner = group.node_tree
        bpy.data.node_groups.remove(inner)
        self.assertIsNone(group.node_tree)
        trees = len(bpy.data.node_groups)
        repair_flex_groups(self.tree)
        self.assertEqual(len(bpy.data.node_groups), trees + 1)  # no extra empty tree
        self.assertIsNotNone(group.node_tree)
        self.assertEqual([socket.name for socket in group.outputs], ['Flex Rules'])
        # A node whose output is not there yet (init() still running) is left alone
        group.outputs.remove(group.outputs['Flex Rules'])
        orphan = group.node_tree
        group.node_tree = None
        repair_flex_groups(self.tree)
        self.assertIsNone(group.node_tree)
        self.assertEqual(len(group.outputs), 0)
        self.addCleanup(bpy.data.node_groups.remove, orphan)


if __name__ == '__main__':
    unittest.main()
