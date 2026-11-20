"""Consumer-visible contracts captured from headless reference DMX exports."""

import unittest
from pathlib import Path

import numpy as np

from SourceIO.library.utils import datamodel_v2


FIXTURES = Path(__file__).parent / 'blender_tests/fixtures/source1_dmx/data'
VERSIONS = ((2, 1), (4, 15), (5, 18))


class Source1DMXReferenceTests(unittest.TestCase):
    def load(self, case, encoding, model_format):
        model = datamodel_v2.load(filepath=str(FIXTURES / f'{case}_{encoding}_{model_format}.dmx'))
        self.assertEqual((model.encoding_ver, model.format_ver), (encoding, model_format))
        self.assertEqual(model.root['skeleton'], model.root['model'])
        return model

    def test_single_material_faces_and_corner_uv_seam(self):
        for encoding, model_format in VERSIONS:
            with self.subTest(encoding=encoding, format=model_format):
                model = self.load('single', encoding, model_format)
                mesh = model.root['model']['children'][0]['shape']
                vertex = mesh['bindState']
                np.testing.assert_allclose(vertex['positions'], ((0, 0, 0.125), (-1, 0, 0.125), (-1, -1.5, 0.125),
                                                                 (0, -1.5, 0.125)))
                self.assertEqual(list(vertex['positionsIndices']), [0, 1, 2, 0, 2, 3])
                self.assertEqual(list(vertex['textureCoordinatesIndices']), [0, 1, 2, 3, 2, 4])
                self.assertEqual([list(face['faces']) for face in mesh['faceSets']], [[0, 1, 2, -1], [3, 4, 5, -1]])

    def test_multi_object_hierarchy(self):
        for encoding, model_format in VERSIONS:
            with self.subTest(encoding=encoding, format=model_format):
                model = self.load('multi', encoding, model_format).root['model']
                self.assertEqual([(dag.name, dag['shape'].name, dag['shape']['faceSets'][0]['material']['mtlName'])
                                  for dag in model['children']],
                                 [('Panel Left', 'Panel Left', 'reference_left'),
                                  ('Panel Right', 'Panel Right', 'reference_right')])
                self.assertNotEqual(model['children'][0]['transform']['orientation'].tolist(),
                                    model['children'][1]['transform']['orientation'].tolist())

    def test_rig_hierarchy_weights_and_versioned_joint_list(self):
        for encoding, model_format in VERSIONS:
            with self.subTest(encoding=encoding, format=model_format):
                document = self.load('rigged', encoding, model_format)
                model = document.root['model']
                self.assertEqual([item.name for item in model['children']],
                                 ['blender_implicit', 'Root', 'Weighted Panel'])
                self.assertEqual([bone.name for bone in model['children'][1]['children']], ['Child'])
                vertex = model['children'][2]['shape']['bindState']
                self.assertEqual(vertex['jointCount'], 2)
                np.testing.assert_allclose(vertex['jointWeights'], [1, 0, 1, 0, 0.5, 0.5])
                self.assertEqual(list(vertex['jointIndices']), [1, 0, 2, 0, 1, 2])
                if model_format == 1:
                    self.assertNotIn('jointList', model)
                else:
                    self.assertEqual([item.name for item in model['jointList']],
                                     ['blender_implicit', 'Root', 'Child', 'Weighted Panel'])
                    self.assertEqual([item.dm_type for item in document.elements if item.name is None],
                                     ['DmeTransform'])

    def test_flex_deltas_and_typed_rules(self):
        for encoding, model_format in VERSIONS:
            with self.subTest(encoding=encoding, format=model_format):
                root = self.load('flex', encoding, model_format).root
                mesh = root['model']['children'][0]['shape']
                smile, blink = mesh['deltaStates']
                self.assertEqual((smile.name, blink.name), ('smile', 'blink'))
                self.assertEqual(list(smile['positionsIndices']), [0])
                np.testing.assert_allclose(smile['positions'], [(-0.2, 0, -0.2)])
                np.testing.assert_allclose(smile['wrinkle'], [0.4])
                self.assertEqual(list(blink['positionsIndices']), [0])
                op = root['combinationOperator']
                face_ctrl, virtual = op['controls']
                self.assertEqual(list(face_ctrl['rawControlNames']), ['smile'])
                self.assertTrue(face_ctrl['stereo'])
                self.assertTrue(face_ctrl['eyelid'])
                self.assertAlmostEqual(face_ctrl['flexMin'], -0.25)
                self.assertAlmostEqual(face_ctrl['flexMax'], 0.8)
                self.assertEqual(list(virtual['rawControlNames']), [])
                target, = op['targets']
                self.assertIs(target['target'], mesh)
                self.assertEqual([(rule.name, rule['expr']) for rule in target['deltaStates']],
                                 [('smile', '(faceCtrl * 0.5)'), ('blink', 'expression_driver')])

    def test_collection_vertex_animation_samples(self):
        for encoding, model_format in VERSIONS:
            with self.subTest(encoding=encoding, format=model_format):
                root = self.load('vertex_animation', encoding, model_format).root
                model = root['model']
                self.assertEqual(model['children'][0].name, 'vcabone_pulse')
                mesh = model['children'][1]['shape']
                first, second = mesh['deltaStates']
                self.assertEqual((first.name, second.name), ('pulse-0', 'pulse-1'))
                self.assertEqual(list(first['positionsIndices']), [])
                self.assertEqual(list(second['positionsIndices']), [0, 1, 2])
                np.testing.assert_allclose(second['positions'], [(0, 0, 0.3)] * 3)
                self.assertEqual(list(root['combinationOperator']['controls'][0]['rawControlNames']),
                                 ['pulse-0', 'pulse-1'])
                self.assertIs(root['combinationOperator']['targets'][0], mesh)


if __name__ == '__main__':
    unittest.main()
