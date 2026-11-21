"""Flex expression and driver-preview evaluation checks, run inside Blender's Python."""

import unittest

import bpy

from SourceIO.blender_bindings.source1.model_export.prop_qc import FlexControllerRecipe, FlexRuleRecipe
from SourceIO.blender_bindings.ui.export_nodes import register_nodes, unregister_nodes
from SourceIO.blender_bindings.ui.export_nodes.build_graph import GraphResolver
from SourceIO.blender_bindings.ui.export_nodes.flex_drivers import _shape_targets
from SourceIO.blender_bindings.ui.export_nodes.flex_graph import (
    apply_range_scaling, apply_rule_ranges, derive_controller_shapes, expand_stereo_rules)


def evaluate(expression, **values):
    return eval(expression, {"min": min, "max": max}, values)


def preview(controls, rules, **values):
    """Driver preview value of the one source shape ``smile`` after the flex rules node's rule pipeline."""
    controls, rules = expand_stereo_rules(controls, rules)
    controls = apply_rule_ranges(controls, rules)
    targets = _shape_targets(apply_range_scaling(rules))
    assert list(targets) == [("face", "smile")], targets
    return evaluate(targets[("face", "smile")], **values)


class FlexEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owns_registration = bpy.types.NodeTree.bl_rna_get_subclass_py('sourceio.model_definition') is None
        if cls.owns_registration:
            register_nodes()

    @classmethod
    def tearDownClass(cls):
        if cls.owns_registration:
            unregister_nodes()

    def setUp(self):
        self.tree = bpy.data.node_groups.new('Flex eval test', 'sourceio.model_definition')
        self.addCleanup(bpy.data.node_groups.remove, self.tree)
        mesh = bpy.data.meshes.new('face')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        self.addCleanup(bpy.data.meshes.remove, mesh)
        self.face = bpy.data.objects.new('face', mesh)
        self.addCleanup(bpy.data.objects.remove, self.face)
        self.face.shape_key_add(name='Basis')
        self.smile = self.face.shape_key_add(name='smile')
        self.smile.slider_min, self.smile.slider_max = -1.0, 1.0

    def raw(self, text):
        node = self.tree.nodes.new('SourceIOFlexRawExpressionNode')
        node.expression = text
        return node

    def test_raw_expression_keeps_precedence_downstream(self):
        nodes, links = self.tree.nodes, self.tree.links
        product = nodes.new('SourceIOFlexMathNode')
        product.operation = 'MULTIPLY'
        two = nodes.new('SourceIOFlexConstantNode')
        two.value = 2.0
        links.new(self.raw('a + b').outputs['Value'], product.inputs['A'])
        links.new(two.outputs['Value'], product.inputs['B'])
        self.assertEqual(evaluate(GraphResolver()(product.outputs['Value']), a=1.0, b=2.0), 6.0)

    def test_parenthesised_raw_controller_still_drives_shape_range(self):
        rule = self.tree.nodes.new('SourceIOFlexShapeRuleNode')
        rule.obj, rule.shape_name, rule.use_shape_range = self.face, 'smile', True
        self.tree.links.new(self.raw(' smile_ctl ').outputs['Value'], rule.inputs['Value'])
        recipe = GraphResolver()(rule.outputs['Rule'])
        self.assertEqual(recipe.weight_range, (-1.0, 1.0))
        controls = (FlexControllerRecipe('smile_ctl', (), wrinkle_scales=()),)
        ranged = apply_rule_ranges(controls, (recipe,))
        self.assertEqual((ranged[0].minimum, ranged[0].maximum), (-1.0, 1.0))
        self.assertEqual(derive_controller_shapes(controls, (recipe,))[0].raw_names, ('smile',))

    def test_range_part_that_is_not_a_bare_controller_is_rejected(self):
        rule = FlexRuleRecipe('face', 'smile', '(c) * 2', weight_range=(-1.0, 1.0))
        with self.assertRaises(ValueError):
            apply_rule_ranges((FlexControllerRecipe('c', (), wrinkle_scales=()),), (rule,))

    def test_negative_range_preview_reproduces_the_slider_in_both_directions(self):
        controls = (FlexControllerRecipe('c', (), wrinkle_scales=()),)
        rules = (FlexRuleRecipe('face', 'smile', 'c', weight_range=(-1.0, 1.0)),)
        for value in (-1.0, -0.5, 0.0, 0.25, 1.0):
            self.assertAlmostEqual(preview(controls, rules, c=value), value)

    def test_asymmetric_range_weights_each_part_by_its_own_scale(self):
        controls = (FlexControllerRecipe('c', (), wrinkle_scales=()),)
        rules = (FlexRuleRecipe('face', 'smile', 'c', weight_range=(-0.5, 2.0)),)
        for value in (-0.5, -0.25, 0.0, 1.0, 2.0):
            self.assertAlmostEqual(preview(controls, rules, c=value), value)

    def test_stereo_sides_are_averaged_not_every_part(self):
        controls = (FlexControllerRecipe('c', (), stereo=True, wrinkle_scales=()),)
        rules = (FlexRuleRecipe('face', 'smile', 'c', weight_range=(-1.0, 1.0)),)
        values = dict(left_c=-0.5, right_c=0.5)
        self.assertAlmostEqual(preview(controls, rules, **values), 0.0)
        self.assertAlmostEqual(preview(controls, rules, left_c=1.0, right_c=1.0), 1.0)
        self.assertAlmostEqual(preview(controls, rules, left_c=-1.0, right_c=-1.0), -1.0)
        self.assertAlmostEqual(preview(controls, rules, left_c=1.0, right_c=0.0), 0.5)
        self.assertAlmostEqual(preview(controls, rules, left_c=0.0, right_c=-1.0), -0.5)

    def test_stereo_without_range_is_the_side_average(self):
        controls = (FlexControllerRecipe('c', (), stereo=True, wrinkle_scales=()),)
        rules = (FlexRuleRecipe('face', 'smile', 'c'),)
        self.assertAlmostEqual(preview(controls, rules, left_c=1.0, right_c=0.0), 0.5)

    def test_ordinary_rules_are_left_unchanged(self):
        rules = (FlexRuleRecipe('face', 'smile', 'c * 2'),)
        self.assertEqual(_shape_targets(rules), {('face', 'smile'): 'c * 2'})


if __name__ == '__main__':
    unittest.main()
