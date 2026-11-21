"""Source 1 flex controls, target binding, and sparse vertex data."""

import tempfile
from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_builder import write_dmx
from SourceIO.blender_bindings.source1.model_export.dmx_flex import _advanced, _stereo_balance
from SourceIO.blender_bindings.source1.model_export.dmx_stereo import _left_weights
from SourceIO.blender_bindings.source1.model_export.dmx_types import Capture, Mesh, Shape, VertexAnimation, face_array
from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    FlexControllerRecipe, FlexRuleRecipe, FlexRuleSetRecipe, GeometryRecipe, ModelEntryRecipe, VirtualControllerRecipe)
from SourceIO.library.utils import datamodel_v2 as dm


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
NORMALS = (((0., 0., 1.),) * 3,)
BASE = ((-1., 0., 0.), (1., 0., 0.), (0., 1., 0.))
VERSIONS = ((2, 1), (4, 15), (5, 18))


def mesh(name='Panel', shapes=()):
    return Mesh(name, IDENTITY, None, None, None, BASE,
                *face_array([((0, 1, 2), NORMALS[0], ((0., 0.), (1., 0.), (0., 1.)), 'panel')]), (), shapes)


def shape(name, z):
    return Shape(name, ((-1., 0., z), *BASE[1:]), NORMALS)


def recipe(**kwargs):
    return ModelEntryRecipe('face', GeometryRecipe(object(), 'face'), **kwargs)


class TypedFlexTests(unittest.TestCase):
    def write(self, capture, entry=None, advanced_text=None, version=(4, 15)):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'face.dmx'
            write_dmx(capture, path, encoding_version=version[0], model_version=version[1],
                      entry=entry, advanced_text=advanced_text)
            return dm.load(filepath=path)

    def test_underscore_names_are_plain_shapes_not_inferred_correctives(self):
        a = shape('a', .2)
        b = shape('b', .3)
        combined = shape('a_b', .7)
        capture = Capture('face', (mesh(shapes=(a, b, combined)),), None)
        for version in VERSIONS:
            with self.subTest(version=version):
                document = self.write(capture, version=version)
                operator = document.root['combinationOperator']
                self.assertEqual([control.name for control in operator['controls']], ['a', 'b', 'a_b'])
                target = document.root['model']['children'][0]['shape']
                self.assertIs(operator['targets'][0], target)
                deltas = {delta.name: delta for delta in target['deltaStates']}
                np.testing.assert_allclose(deltas['a_b']['positions'], ((0, 0, .7),), atol=1e-6)
                self.assertEqual(list(deltas['a_b']['positionsIndices']), [0])
                self.assertEqual(list(target['bindState']['balance']), [1, 0, 0.5])

    def test_auto_balance_and_stereo_halves_split_across_model_x(self):
        # Turned half around Z and moved to the character's left (+X), so
        # local X runs along model -Y.
        turned = ((0., -1., 0., 10.), (1., 0., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        left_eye = replace(mesh(), matrix_world=turned)
        np.testing.assert_array_equal(_stereo_balance(left_eye), [0, 0, 0])
        np.testing.assert_array_equal(_left_weights(left_eye), [1, 1, 1])
        # Model X of BASE ((-1,0,0), (1,0,0), (0,1,0)) is -0, 0 and -1.
        straddling = replace(left_eye, matrix_world=((0., -1., 0., 0.), *turned[1:]))
        np.testing.assert_allclose(_stereo_balance(straddling), [.5, .5, 1])
        np.testing.assert_allclose(_left_weights(straddling), [.5, .5, 0])

    def test_direct_stereo_and_wrinkle_use_uv_domain(self):
        capture = Capture('face', (mesh(shapes=(shape('smile', .4),)),), None)
        entry = recipe(flex_controllers=(FlexControllerRecipe('faceControl', ('smile',), True, True, -.5, 2., (2.,)),))
        document = self.write(capture, entry)
        control = document.root['combinationOperator']['controls'][0]
        self.assertEqual(list(control['rawControlNames']), ['smile'])
        self.assertEqual((control['stereo'], control['eyelid']), (True, True))
        self.assertAlmostEqual(control['flexMin'], -.5)
        self.assertAlmostEqual(control['flexMax'], 2.)
        target = document.root['model']['children'][0]['shape']
        delta = target['deltaStates'][0]
        self.assertEqual(list(delta['wrinkleIndices']), [0])
        np.testing.assert_allclose(delta['wrinkle'], (2.,))
        self.assertEqual(list(target['bindState']['balanceIndices']), [0, 1, 2])

    def test_graph_rules_bind_each_target_and_validate_expressions(self):
        left = mesh('Left', (shape('left', .2),))
        right = mesh('Right', (shape('right', .3),))
        capture = Capture('face', (left, right), None)
        rules = FlexRuleSetRecipe(GeometryRecipe(object(), 'face'),
                                  (FlexControllerRecipe('leftControl', ('left',)), VirtualControllerRecipe('value')),
                                  (FlexRuleRecipe('Left', 'left', 'leftControl * value'),
                                   FlexRuleRecipe('Right', 'right', 'value')))
        document = self.write(capture, recipe(flex_rule_set=rules))
        operator = document.root['combinationOperator']
        targets = {target.name: target for target in operator['targets']}
        self.assertEqual(set(targets), {'Left', 'Right'})
        meshes = {dag.name: dag['shape'] for dag in document.root['model']['children']}
        self.assertIs(targets['Left']['target'], meshes['Left'])
        self.assertIs(targets['Right']['target'], meshes['Right'])
        self.assertEqual(targets['Left']['deltaStates'][0]['expr'], 'leftControl * value')
        invalid = FlexRuleSetRecipe(rules.geometry, rules.controls,
                                    (FlexRuleRecipe('Left', 'left', 'missing'), *rules.rules[1:]))
        with self.assertRaisesRegex(ValueError, 'unknown controller'):
            self.write(capture, recipe(flex_rule_set=invalid))

    def test_flex_rules_and_vertex_animation_clips_are_rejected_together(self):
        base = mesh('Panel', (shape('smile', .2),))
        clips = (VertexAnimation('pulse', 'Panel', (Shape('pulse-0', BASE, NORMALS),)),)
        rules = FlexRuleSetRecipe(GeometryRecipe(object(), 'face'), (FlexControllerRecipe('smile', ('smile',)),),
                                  (FlexRuleRecipe('Panel', 'smile', 'smile'),))
        with self.assertRaisesRegex(ValueError, 'Flex rules cannot be combined'):
            self.write(Capture('face', (base,), None, clips), recipe(flex_rule_set=rules))
        # Clips alone, and rules alone, still export.
        self.write(Capture('face', (base,), None, clips))
        self.write(Capture('face', (base,), None), recipe(flex_rule_set=rules))

    def test_shared_shape_name_still_targets_each_mesh_rule(self):
        capture = Capture('face', (mesh('Left', (shape('smile', .2),)), mesh('Right', (shape('smile', .3),))), None)
        rules = FlexRuleSetRecipe(GeometryRecipe(object(), 'face'),
                                  (FlexControllerRecipe('shared', ('smile',)),),
                                  (FlexRuleRecipe('Left', 'smile', 'shared'),
                                   FlexRuleRecipe('Right', 'smile', 'shared * 0.5')))
        document = self.write(capture, recipe(flex_rule_set=rules))
        target = {element.name: element for element in document.root['combinationOperator']['targets']}
        dag = {element.name: element['shape'] for element in document.root['model']['children']}
        self.assertIs(target['Left']['target'], dag['Left'])
        self.assertIs(target['Right']['target'], dag['Right'])
        self.assertEqual([control.name for control in document.root['combinationOperator']['controls']], ['shared'])


    def test_advanced_text_is_transferred_and_rebound_not_dropped(self):
        source = dm.DataModel('keyvalues2', 1, 'model', 15)
        root = source.add_element('root')
        operator = source.add_element('combinationOperator', 'DmeCombinationOperator')
        root['combinationOperator'] = operator
        control = source.add_element('smileControl', 'DmeCombinationInputControl')
        control['rawControlNames'] = dm.StringArray(['smile'])
        control['stereo'] = False
        control['eyelid'] = False
        control['flexMin'] = 0.
        control['flexMax'] = .6
        control['wrinkleScales'] = dm.FloatArray([0.])
        operator['controls'] = dm.ElementArray([control])
        operator['controlValues'] = dm.Vector3Array([(0., 0., .5)])
        operator['controlValuesLagged'] = dm.Vector3Array([(0., 0., .5)])
        operator['usesLaggedValues'] = False
        operator['dominators'] = dm.ElementArray()
        source_mesh = source.add_element('Panel', 'DmeMesh')
        operator['targets'] = dm.ElementArray([source_mesh])
        shared = source.add_element('shared', 'DmElement')
        shared['owner'] = operator
        shared['axis'] = dm.Angle((1., 2., 3.))
        shared['angles'] = dm.AngleArray([(4., 5., 6.)])
        shared['flags'] = dm.BoolArray([True, False])
        operator['metadata'] = shared
        control['metadata'] = shared
        text = dm.dump(source)
        capture = Capture('face', (mesh(shapes=(shape('smile', .4),)),), None)
        document = self.write(capture, recipe(flex_controller_source='face-controller'), text)
        new_operator = document.root['combinationOperator']
        self.assertAlmostEqual(new_operator['controls'][0]['flexMax'], .6)
        self.assertIs(new_operator['targets'][0], document.root['model']['children'][0]['shape'])
        self.assertIs(new_operator['metadata'], new_operator['controls'][0]['metadata'])
        self.assertIs(new_operator['metadata']['owner'], new_operator)
        self.assertIsInstance(new_operator['metadata']['axis'], dm.Angle)
        self.assertIsInstance(new_operator['metadata']['angles'], dm.AngleArray)
        self.assertEqual(list(new_operator['metadata']['flags']), [True, False])
        second = self.write(capture, recipe(flex_controller_source='face-controller'), text)
        self.assertEqual(new_operator.id, second.root['combinationOperator'].id)
        self.assertEqual(new_operator['metadata'].id, second.root['combinationOperator']['metadata'].id)

        target = dm.DataModel('binary', 4, 'model', 15)
        target.add_element('face')
        target_mesh = target.add_element('Panel', 'DmeMesh')
        with patch.object(dm, 'parse', return_value=source):
            copied, _ = _advanced(target, text, {'Panel': target_mesh}, {'smile': {'Panel'}})
        self.assertIs(copied['targets'][0], target_mesh)
        self.assertIsNot(copied['metadata'], shared)
        self.assertIs(copied['metadata']['owner'], copied)
        self.assertIsNot(copied['controls'][0]['rawControlNames'], control['rawControlNames'])
        self.assertFalse(np.shares_memory(copied['controlValues'], operator['controlValues']))
        self.assertFalse(np.shares_memory(copied['metadata']['axis'], shared['axis']))
        self.assertFalse(np.shares_memory(copied['metadata']['angles'], shared['angles']))
        self.assertFalse(np.shares_memory(copied['metadata']['flags'], shared['flags']))

        source_mesh.name = 'Missing'
        with self.assertRaisesRegex(ValueError, "unknown mesh 'Missing'"):
            self.write(capture, recipe(flex_controller_source='face-controller'), dm.dump(source))
        source_mesh.name = 'Panel'
        unresolved = text.replace(f'"element" "{source_mesh.id}"', '"element" "00000000-0000-0000-0000-000000000001"')
        self.assertNotEqual(unresolved, text)
        with self.assertRaisesRegex(ValueError, 'cannot be parsed: Unresolved element'):
            self.write(capture, recipe(flex_controller_source='face-controller'), unresolved)
        rules = source.add_element('Panel', 'DmeFlexRules')
        rule = source.add_element('smile', 'DmeFlexRuleExpression')
        rule['expr'] = 'missingController'
        rules['deltaStates'] = dm.ElementArray([rule])
        rules['deltaStateWeights'] = dm.Vector2Array([(0., 0.)])
        operator['targets'] = dm.ElementArray([source_mesh, rules])
        with self.assertRaisesRegex(ValueError, 'unknown controller'):
            self.write(capture, recipe(flex_controller_source='face-controller'), dm.dump(source))
        rule['expr'] = 'smileControl'
        bound = self.write(capture, recipe(flex_controller_source='face-controller'), dm.dump(source))
        mesh_target, rule_target = bound.root['combinationOperator']['targets']
        self.assertIs(rule_target['target'], mesh_target)
        with self.assertRaisesRegex(ValueError, 'was not supplied'):
            self.write(capture, recipe(flex_controller_source='face-controller'))


if __name__ == '__main__':
    unittest.main()
