"""World-space axis conversion for the Blender-independent DMX capture."""

from __future__ import annotations

import unittest

import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_axis import orient_capture
from SourceIO.blender_bindings.source1.model_export.dmx_builder import build_document, build_vca_sequence
from SourceIO.blender_bindings.source1.model_export.dmx_types import (
    Bone, Capture, Face, Mesh, Rig, Shape, VertexAnimation)


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


def translated(x=0., y=0., z=0.):
    return ((1., 0., 0., x), (0., 1., 0., y), (0., 0., 1., z), (0., 0., 0., 1.))


def mesh(name, world=IDENTITY, *, parent=None, parent_type=None, parent_bone=None, weights=()):
    return Mesh(name, world, parent, parent_type, parent_bone, ((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)),
                (Face((0, 1, 2), ((0., -1., 0.),) * 3, ((0., 0.), (1., 0.), (0., 1.)), 'material'),),
                weights, (Shape('tall', ((0., 0., 1.), (1., 0., 0.), (0., 0., 1.)), (((0., -1., 0.),) * 3,)),))


class AxisCaptureTests(unittest.TestCase):
    def test_identity_invalid_axis_and_unsupported_matrices(self):
        capture = Capture('Model', (mesh('Panel'),), None)
        self.assertIs(orient_capture(capture, 'Z'), capture)
        with self.assertRaisesRegex(ValueError, 'up axis'):
            orient_capture(capture, 'X')
        for invalid in (((1.,),),
                        ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 1., 1.)),
                        ((float('nan'), 0., 0., 0.), *IDENTITY[1:])):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'affine'):
                orient_capture(Capture('Model', (mesh('Panel', invalid),), None), 'Y')

    def test_world_rotation_preserves_parent_relative_and_local_samples(self):
        parent = mesh('Parent', translated(0., 0., 2.))
        child = mesh('Child', translated(0., 1., 2.), parent='Parent', parent_type='OBJECT')
        original = Capture('Model', (child, parent), None)
        oriented = orient_capture(original, 'Y')
        self.assertIsNot(oriented, original)
        self.assertEqual(oriented.meshes[0].matrix_world[1][3], 2.)
        self.assertEqual(oriented.meshes[0].matrix_world[2][3], -1.)
        self.assertIs(oriented.meshes[0].positions, child.positions)
        self.assertIs(oriented.meshes[0].faces, child.faces)
        self.assertIs(oriented.meshes[0].shapes, child.shapes)
        self.assertEqual(original.meshes[0].matrix_world, translated(0., 1., 2.))
        model = build_document(oriented, encoding_version=4, model_version=15).root['model']
        parent_dag = model['children'][0]
        child_dag = parent_dag['children'][0]
        np.testing.assert_allclose(parent_dag['transform']['position'], (0., 2., 0.))
        np.testing.assert_allclose(child_dag['transform']['position'], (0., 1., 0.))
        # A +Z local point becomes +Y in model space, while its normal is
        # still stored in object-local coordinates.
        np.testing.assert_allclose(parent_dag['shape']['bindState']['positions'][2], (0., 0., 1.))
        quaternion = np.asarray(parent_dag['transform']['orientation'])
        self.assertAlmostEqual(abs(quaternion[0]), 2 ** -.5)
        self.assertAlmostEqual(abs(quaternion[3]), 2 ** -.5)

    def test_rig_rest_and_vca_skeleton_share_orientation(self):
        rig = Rig('Rig', translated(0., 0., 2.),
                  (Bone('Root', None, IDENTITY), Bone('Child', 'Root', translated(0., 0., 1.))))
        attached = mesh('Attached', translated(0., 0., 3.), parent='Rig', parent_type='BONE', parent_bone='Child',
                        weights=((('Root', 1.),),) * 3)
        frames = (Shape('pulse-0', attached.positions, (attached.faces[0].normals,)),
                  Shape('pulse-1', attached.shapes[0].positions, (attached.faces[0].normals,)))
        capture = Capture('Model', (attached,), rig, (VertexAnimation('pulse', 'Attached', frames, True),))
        oriented = orient_capture(capture, 'Y')
        self.assertIs(oriented.rig.bones, rig.bones)
        self.assertIs(oriented.vertex_animations, capture.vertex_animations)
        self.assertEqual(rig.matrix_world, translated(0., 0., 2.))
        geometry = build_document(oriented, encoding_version=4, model_version=15)
        sequence = build_vca_sequence(oriented, 'pulse', encoding_version=4, model_version=15)
        geometry_joints = {joint.name: joint for joint in geometry.root['model']['jointList']}
        sequence_joints = {joint.name: joint for joint in sequence.root['skeleton']['jointList']}
        for name in ('Root', 'Child'):
            for attribute in ('position', 'orientation'):
                np.testing.assert_allclose(geometry_joints[name]['transform'][attribute],
                                           sequence_joints[name]['transform'][attribute])
        np.testing.assert_allclose(geometry_joints['Root']['transform']['position'], (0., 2., 0.))
        np.testing.assert_allclose(geometry_joints['Child']['transform']['position'], (0., 0., 1.))
        self.assertEqual([dag.name for dag in geometry_joints['Child']['children']], ['Attached'])
        np.testing.assert_allclose(geometry_joints['Attached']['transform']['position'], (0., 0., 0.))
        self.assertIn('vcabone_pulse', geometry_joints)
        self.assertIn('vcabone_pulse', sequence_joints)


if __name__ == '__main__':
    unittest.main()
