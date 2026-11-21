"""Consumer-visible geometry and serialization behavior of the Source 1 builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_builder import (
    build_document, write_dmx, write_vca_sequences)
from SourceIO.blender_bindings.source1.model_export.dmx_types import (
    Bone, Capture, Mesh, Rig, Shape, VertexAnimation, face_array)
from SourceIO.library.utils import datamodel_v2 as dm


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
VERSIONS = ((2, 1), (4, 15), (5, 18))


def panel(*, name='Panel', matrix=IDENTITY, parent=None, parent_type=None, parent_bone=None, weights=(),
          shapes=(), faces=None):
    if faces is None:
        faces = face_array([((0, 1, 2), ((0., 0., 1.),) * 3, ((0., 0.), (1., 0.), (0., 1.)), 'panel')])
    return Mesh(name, matrix, parent, parent_type, parent_bone, ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), *faces,
                weights, shapes)


def rotation(quaternion):
    x, y, z, w = np.asarray(quaternion).tolist()
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                     (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                     (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))))


class DMXBuilderTests(unittest.TestCase):
    def test_versioned_roundtrip_hierarchy_and_stable_ids(self):
        capture = Capture('Scene', (panel(),), None)
        for encoding, model_version in VERSIONS:
            with self.subTest(encoding=encoding, model_version=model_version):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'geometry.dmx'
                    write_dmx(capture, path, encoding_version=encoding, model_version=model_version)
                    first = path.read_bytes()
                    write_dmx(capture, path, encoding_version=encoding, model_version=model_version)
                    self.assertEqual(path.read_bytes(), first)
                    document = dm.load(filepath=path)
                    self.assertEqual((document.encoding_ver, document.format_ver), (encoding, model_version))
                    model = document.root['model']
                    self.assertIs(document.root['skeleton'], model)
                    self.assertEqual([(dag.name, dag['shape'].name) for dag in model['children']], [('Panel', 'Panel')])
                    self.assertEqual(len(model['baseStates'][0]['transforms']), 1)
                    self.assertEqual(len(model['jointTransforms']), 1)
                    self.assertEqual(model['transform'].name, '' if encoding < 4 else None)
                    self.assertEqual('jointList' in model, model_version >= 11)
                    if model_version >= 11:
                        self.assertEqual([joint.name for joint in model['jointList']], ['Panel'])

    def test_material_override_uv_seam_and_reflected_world_geometry(self):
        angle = 0.5
        cosine, sine = np.cos(angle), np.sin(angle)
        world = ((-cosine, -1.5*sine, 0., 2.), (-sine, 1.5*cosine, 0., 3.), (0., 0., 1., 1.), (0., 0., 0., 1.))
        faces = face_array([((0, 1, 2), ((0., 0., 1.),) * 3, ((0., 0.), (1., 0.), (1., 1.)), 'base'),
                            ((0, 1, 2), ((0., 0., 1.),) * 3, ((0.5, 0.), (1., 1.), (0., 1.)), 'override')])
        capture = Capture('Scene', (panel(matrix=world, faces=faces),), None)
        document = build_document(capture, encoding_version=4, model_version=15)
        dag = document.root['model']['children'][0]
        mesh = dag['shape']
        vertex = mesh['bindState']
        actual = np.asarray(vertex['positions']) @ rotation(dag['transform']['orientation']).T
        actual += np.asarray(dag['transform']['position'])
        expected = np.asarray(panel().positions) @ np.asarray(world)[:3, :3].T
        expected += np.asarray(world)[:3, 3]
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        self.assertEqual([item['material']['mtlName'] for item in mesh['faceSets']], ['base', 'override'])
        self.assertEqual([list(item['faces']) for item in mesh['faceSets']], [[0, 1, 2, -1], [3, 4, 5, -1]])
        self.assertNotEqual(vertex['textureCoordinatesIndices'][0], vertex['textureCoordinatesIndices'][3])
        for face_set in mesh['faceSets']:
            a, b, c = (actual[vertex['positionsIndices'][index]] for index in face_set['faces'][:3])
            n = rotation(dag['transform']['orientation']) @ vertex['normals'][face_set['faces'][0]]
            self.assertGreater(np.dot(np.cross(b - a, c - a), n), 0)

    def test_mesh_parent_world_transform_and_order(self):
        parent_world = ((1., 0., 0., 3.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        child_world = ((1., 0., 0., 3.), (0., 1., 0., 2.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        child = panel(name='A Child', parent='Z Parent', parent_type='OBJECT', matrix=child_world)
        parent = panel(name='Z Parent', matrix=parent_world)
        model = build_document(Capture('Scene', (child, parent), None),
                               encoding_version=4, model_version=15).root['model']
        self.assertEqual([dag.name for dag in model['children']], ['Z Parent'])
        self.assertEqual([dag.name for dag in model['children'][0]['children']], ['A Child'])
        self.assertEqual([joint.name for joint in model['jointList']], ['Z Parent', 'A Child'])
        np.testing.assert_allclose(model['children'][0]['transform']['position'], (3, 0, 0))
        np.testing.assert_allclose(model['children'][0]['children'][0]['transform']['position'], (0, 2, 0))

    def test_joint_parenting_weights_and_bone_attachment(self):
        translated = ((1., 0., 0., 0.), (0., 1., 0., 1.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        rig = Rig('Rig', IDENTITY, (Bone('Root', None, IDENTITY), Bone('Child', 'Root', translated),
                                   Bone('Other', None, IDENTITY), Bone('Third', None, IDENTITY)))
        weights = ((('Root', .2), ('Child', .4), ('Other', .3), ('Third', .1)),
                   (('Root', 1.),), (('Child', .5), ('Root', .5)))
        attached = panel(parent='Rig', parent_type='BONE', parent_bone='Child', matrix=translated, weights=weights)
        document = build_document(Capture('Scene', (attached,), rig), encoding_version=5, model_version=18)
        model = document.root['model']
        self.assertEqual([joint.name for joint in model['jointList']],
                         ['blender_implicit', 'Root', 'Child', 'Other', 'Third', 'Panel'])
        self.assertEqual([child.name for child in model['children'][1]['children']], ['Child'])
        child = model['children'][1]['children'][0]
        self.assertEqual([dag.name for dag in child['children']], ['Panel'])
        np.testing.assert_allclose(child['transform']['position'], (0, 1, 0))
        np.testing.assert_allclose(child['children'][0]['transform']['position'], (0, 0, 0))
        vertex = child['children'][0]['shape']['bindState']
        self.assertEqual(vertex['jointCount'], 3)
        self.assertEqual(list(vertex['jointIndices'][:3]), [2, 3, 1])
        np.testing.assert_allclose(vertex['jointWeights'][:3], [.4/.9, .3/.9, .2/.9])
        np.testing.assert_allclose(np.asarray(vertex['jointWeights']).reshape(3, 3).sum(axis=1), (1, 1, 1))
        untrimmed = build_document(Capture('Scene', (attached,), rig), encoding_version=5, model_version=18,
                                   trim_weights=False)
        vertex = untrimmed.root['model']['children'][1]['children'][0]['children'][0]['shape']['bindState']
        self.assertEqual(vertex['jointCount'], 4)
        self.assertEqual(list(vertex['jointIndices']), [2, 3, 1, 4, 1, 0, 0, 0, 1, 2, 0, 0])
        np.testing.assert_allclose(vertex['jointWeights'], [.4, .3, .2, .1, 1, 0, 0, 0, .5, .5, 0, 0])

    def test_sparse_position_and_corner_normal_shape_deltas(self):
        base = panel(shapes=(Shape('smile', ((0.2, 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                                   (((0., 1., 0.), (0., 0., 1.), (0., 0., 1.)),)),
                             Shape('same', panel().positions, (((0., 0., 1.),) * 3,))))
        mesh = build_document(Capture('Scene', (base,), None), encoding_version=5,
                              model_version=18).root['model']['children'][0]['shape']
        smile, unchanged = mesh['deltaStates']
        self.assertEqual([shape.name for shape in mesh['deltaStates']], ['smile', 'same'])
        self.assertEqual(list(smile['positionsIndices']), [0])
        np.testing.assert_allclose(smile['positions'], ((.2, 0, 0),))
        self.assertEqual(list(smile['normalsIndices']), [0])
        self.assertEqual(list(unchanged['positionsIndices']), [])
        self.assertEqual(list(unchanged['normalsIndices']), [])
        self.assertEqual(len(mesh['deltaStateWeights']), 2)

    def test_reflection_preserves_shape_delta_and_shear_is_rejected(self):
        reflection = ((-1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        unchanged_normals = (((0., 0., 1.),) * 3,)
        shapes = (Shape('move', ((.2, 0., 0.), (1., 0., 0.), (0., 1., 0.)), unchanged_normals),
                  Shape('normal_only', panel().positions, (((0., 1., 0.), (0., 0., 1.), (0., 0., 1.)),)))
        capture = Capture('Scene', (panel(matrix=reflection, shapes=shapes),), None)
        dag = build_document(capture, encoding_version=5, model_version=18).root['model']['children'][0]
        move, normal_only = dag['shape']['deltaStates']
        self.assertEqual(list(move['positionsIndices']), [0])
        np.testing.assert_allclose(rotation(dag['transform']['orientation']) @ move['positions'][0],
                                   (-.2, 0, 0), atol=1e-5)
        self.assertEqual(list(normal_only['positionsIndices']), [])
        self.assertEqual(list(normal_only['normalsIndices']), [0])

        shear = ((1., .25, 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.dmx'
            path.write_bytes(b'old geometry')
            with self.assertRaisesRegex(ValueError, 'shear'):
                write_dmx(Capture('Scene', (panel(matrix=shear),), None), path, encoding_version=5, model_version=18)
            self.assertEqual(path.read_bytes(), b'old geometry')

    def test_invalid_version_and_partial_write_leave_previous_file_intact(self):
        capture = Capture('Scene', (panel(),), None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.dmx'
            path.write_bytes(b'prior geometry')
            with self.assertRaisesRegex(ValueError, 'requires binary'):
                write_dmx(capture, path, encoding_version=2, model_version=15)
            def fail_write(_, stream):
                stream.write(b'partial')
                raise OSError('disk full')
            with patch.object(dm, 'dump', fail_write):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    write_dmx(capture, path, encoding_version=4, model_version=15)
            self.assertEqual(path.read_bytes(), b'prior geometry')
            self.assertEqual([file.name for file in Path(directory).iterdir()], ['geometry.dmx'])

    def test_vertex_animation_samples_and_sequence_drive_matching_joint(self):
        base = panel()
        moved = Shape('pulse-1', ((0., 0., .3), *base.positions[1:]), base.faces['normals'][:1])
        frames = (Shape('pulse-0', base.positions, base.faces['normals'][:1]), moved)
        animation = VertexAnimation('pulse', 'Panel', frames, export_sequence=True, frame_rate=24.0)
        capture = Capture('Scene', (base,), None, (animation,))
        for encoding, model_version in VERSIONS:
            with self.subTest(encoding=encoding, model_version=model_version):
                with tempfile.TemporaryDirectory() as directory:
                    geometry_path = Path(directory) / 'geometry.dmx'
                    write_dmx(capture, geometry_path, encoding_version=encoding, model_version=model_version)
                    geometry = dm.load(filepath=geometry_path)
                    mesh = geometry.root['model']['children'][-1]['shape']
                    self.assertEqual([delta.name for delta in mesh['deltaStates']], ['pulse-0', 'pulse-1'])
                    self.assertEqual(list(mesh['deltaStates'][0]['positionsIndices']), [])
                    self.assertEqual(list(mesh['deltaStates'][1]['positionsIndices']), [0])
                    np.testing.assert_allclose(mesh['deltaStates'][1]['positions'][0], (0, 0, .3))
                    self.assertIn('vcabone_pulse', [joint.name for joint in geometry.root['model']['children']])
                    paths = write_vca_sequences(capture, directory, encoding_version=encoding,
                                                model_version=model_version)
                    self.assertEqual([path.name for path in paths], ['vcaanim_pulse.dmx'])
                    sequence = dm.load(filepath=paths[0])
                    clip = sequence.root['animationList']['animations'][0]
                    self.assertNotIn('model', sequence.root)
                    self.assertAlmostEqual(float(clip['timeFrame']['duration'])
                                           if model_version > 11 else
                                           clip['timeFrame']['durationTime'] / 10_000,
                                           1 / 24, delta=.0001)
                    self.assertEqual(clip['frameRate'], 24)
                    position = next(channel for channel in clip['channels'] if channel.name == 'vcabone_pulse_p')
                    layer = position['log']['layers'][0]
                    np.testing.assert_allclose(layer['values'], ((1, 0, 0), (1, 1, 0)))
                    expected_time = 1 / 24 if model_version > 11 else round(10_000 / 24)
                    self.assertAlmostEqual(layer['times'][1], expected_time, delta=.0001)
                    self.assertIs(position['toElement'],
                                  next(joint['transform'] for joint in sequence.root['skeleton']['children']
                                       if joint.name == 'vcabone_pulse'))

    def test_one_vca_controller_targets_all_collection_meshes(self):
        left = panel(name='Left')
        right = panel(name='Right')
        def frames():
            return (Shape('pulse-0', left.positions, left.faces['normals'][:1]),
                    Shape('pulse-1', ((0., 0., .2), *left.positions[1:]), left.faces['normals'][:1]))
        capture = Capture('Scene', (left, right), None,
                          (VertexAnimation('pulse', 'Left', frames()),
                           VertexAnimation('pulse', 'Right', frames())))
        document = build_document(capture, encoding_version=4, model_version=15)
        operator = document.root['combinationOperator']
        self.assertEqual([control.name for control in operator['controls']], ['pulse'])
        self.assertEqual([target.name for target in operator['targets']], ['Left', 'Right'])
        self.assertEqual([joint.name for joint in document.root['model']['children']
                          if joint.dm_type == 'DmeJoint'], ['vcabone_pulse'])


if __name__ == '__main__':
    unittest.main()
