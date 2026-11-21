"""Pure Source 1 SMD geometry builder and atomic installation contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_axis import orient_capture
from SourceIO.blender_bindings.source1.model_export.dmx_types import Bone, Capture, Mesh, Rig, face_array
from SourceIO.blender_bindings.source1.model_export.smd_builder import build_document, write_smd
from SourceIO.library.models import smd


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))


def translated(x=0., y=0., z=0.):
    return ((1., 0., 0., x), (0., 1., 0., y), (0., 0., 1., z), (0., 0., 0., 1.))


def panel(name='Panel', *, world=IDENTITY, parent=None, parent_type=None, parent_bone=None, weights=(), faces=None):
    if faces is None:
        faces = face_array([((0, 1, 2), ((0., 0., 1.),) * 3, ((0., 0.), (1., 0.), (0., 1.)), 'materials/panel')])
    return Mesh(name, world, parent, parent_type, parent_bone, ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), *faces,
                weights, ())


def corners(document, face=None):
    rows = document.triangles['corners']
    return rows if face is None else rows[face]


def materials(document):
    return [document.materials[index] for index in document.triangles['material']]


def links(corner):
    return corner['links'][:corner['link_count']].tolist()


class SMDBuilderTests(unittest.TestCase):
    def test_static_multimesh_names_and_file_roundtrip(self):
        capture = Capture('Collection', (panel('Left'), panel('Right', world=translated(3))), None)
        document = build_document(capture)
        self.assertEqual([(node.id, node.name, node.parent) for node in document.nodes],
                         [(0, 'Left', -1), (1, 'Right', -1)])
        self.assertEqual([pose.bone for pose in document.skeleton[0]], [0, 1])
        self.assertEqual(corners(document)['bone'][:, 0].tolist(), [0, 1])
        self.assertEqual(corners(document, 1)['position'][0].tolist(), [3., 0., 0.])
        self.assertEqual(materials(document), ['materials/panel'] * 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.smd'
            write_smd(capture, path)
            self.assertEqual(smd.load(filepath=path), document)
            initial = path.read_bytes()
            write_smd(capture, path)
            self.assertEqual(path.read_bytes(), initial)
            self.assertEqual([item.name for item in path.parent.iterdir()], ['mesh.smd'])

    def test_original_rig_hierarchy_local_pose_and_normalized_links(self):
        # Child rest matrix is armature-local, not parent-relative.
        rig = Rig('Armature', translated(2., 0., 0.),
                  (Bone('Root', None, translated(0., 1., 0.)), Bone('Child', 'Root', translated(0., 3., 0.))))
        weights = ((('Root', 1.), ('Child', 3.)), (('Child', 1.),), (('Root', 2.), ('Child', 2.)))
        mesh = panel(world=translated(2., 3., 0.), parent='Armature', parent_type='BONE', parent_bone='Child',
                     weights=weights)
        document = build_document(Capture('Model', (mesh,), rig))
        self.assertEqual([(node.name, node.parent) for node in document.nodes], [('Root', -1), ('Child', 0)])
        self.assertEqual([pose.position for pose in document.skeleton[0]], [(2., 1., 0.), (0., 2., 0.)])
        first, second, third = corners(document, 0)
        self.assertEqual(first['bone'], 0)  # Explicit links carry the skin association.
        self.assertEqual(links(first), [(1, .75), (0, .25)])
        self.assertEqual(links(second), [(1, 1.)])
        self.assertEqual(links(third), [(0, .5), (1, .5)])
        self.assertEqual(first['position'].tolist(), [2., 3., 0.])
        self.assertEqual(smd.parse(smd.dump(document)), document)

    def test_local_bone_euler_rotation_from_armature_rest_matrices(self):
        quarter_turn = ((0., -1., 0., 0.), (1., 0., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        child = ((0., -1., 0., 0.), (1., 0., 0., 2.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        rig = Rig('Rig', IDENTITY, (Bone('Root', None, quarter_turn), Bone('Child', 'Root', child)))
        document = build_document(Capture('Model', (panel(parent='Rig', parent_type='BONE', parent_bone='Child'),),
                                           rig))
        root, child_pose = document.skeleton[0]
        np.testing.assert_allclose(root.rotation, (0., 0., np.pi / 2), atol=1e-10)
        np.testing.assert_allclose(child_pose.position, (2., 0., 0.), atol=1e-10)
        np.testing.assert_allclose(child_pose.rotation, (0., 0., 0.), atol=1e-10)

    def test_inverse_transpose_uv_seams_and_reflected_winding(self):
        world = ((-2., 0., 0., 5.), (0., 1., 0., 0.), (0., 0., 3., 0.), (0., 0., 0., 1.))
        normal = tuple(np.array((1., 0., 1.)) / np.sqrt(2))
        faces = face_array([((0, 1, 2), (normal,) * 3, ((0., 0.), (1., 0.), (0., 1.)), 'paint'),
                            ((0, 1, 2), (normal,) * 3, ((.25, .75), (1., 1.), (0., 1.)), 'paint')])
        document = build_document(Capture('Model', (panel(world=world, faces=faces),), None))
        a, b, c = corners(document, 0)
        self.assertEqual([v['position'].tolist() for v in (a, b, c)], [[5., 0., 0.], [5., 1., 0.], [3., 0., 0.]])
        self.assertEqual([v['uv'].tolist() for v in (a, b, c)], [[0., 0.], [0., 1.], [1., 0.]])
        self.assertEqual(corners(document, 1)[0]['uv'].tolist(), [.25, .75])
        expected = np.array((-.5, 0., 1. / 3.))
        np.testing.assert_allclose(a['normal'], expected / np.linalg.norm(expected))
        # Winding remains forward-facing after reflection even with corner normals.
        flat = build_document(Capture('Model', (panel(world=world),), None))
        points = corners(flat, 0)['position']
        face_normal = corners(flat, 0)[0]['normal']
        self.assertGreater(np.dot(np.cross(points[1] - points[0], points[2] - points[0]), face_normal), 0)

    def test_axis_orientation_and_bone_parent_are_world_space(self):
        rig = Rig('Rig', translated(0., 0., 2.), (Bone('Root', None, IDENTITY),
                                                Bone('Tip', 'Root', translated(0., 0., 1.))))
        mesh = panel(world=translated(0., 0., 3.), parent='Rig', parent_type='BONE', parent_bone='Tip')
        document = build_document(orient_capture(Capture('Model', (mesh,), rig), 'Y'))
        self.assertEqual([pose.position for pose in document.skeleton[0]], [(0., 2., 0.), (0., 0., 1.)])
        self.assertEqual(corners(document, 0)[0]['position'].tolist(), [0., 3., 0.])
        self.assertEqual(corners(document, 0)[0]['bone'], 1)
        np.testing.assert_allclose(corners(document, 0)[0]['normal'], (0., 1., 0.), atol=1e-10)

    def test_collision_hulls_group_by_object_and_reuse_render_bones(self):
        rig = Rig('Rig', IDENTITY, (Bone('Root', None, IDENTITY), Bone('Tip', 'Root', translated(0., 1., 0.))))
        weighted = panel('Loose', weights=((('Root', .2), ('Tip', .8)),) * 3)
        attached = panel('Attached', world=translated(0., 1., 0.), parent='Rig', parent_type='BONE', parent_bone='Root')
        capture = Capture('Model', (weighted, attached), rig)
        document = build_document(capture, joints=True)
        self.assertEqual([node.name for node in document.nodes], ['Root', 'Tip'])
        self.assertEqual(materials(document), ['Loose', 'Attached'])
        self.assertEqual(corners(document)['bone'].tolist(), [[1] * 3, [0] * 3])
        self.assertTrue((corners(document)['link_count'] == -1).all())
        self.assertEqual([node.name for node in build_document(
            Capture('Static', (panel('One'), panel('Two')), None), joints=True).nodes], ['root'])
        mixed = panel('Mixed', weights=((('Root', 1.),), (('Tip', 1.),), (('Root', 1.),)))
        with self.assertRaisesRegex(ValueError, 'spans multiple bones'):
            build_document(Capture('Model', (mixed,), rig), joints=True)

    def test_shear_bakes_and_singular_bad_links_and_failed_install_rollback(self):
        shear = ((1., .25, 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        singular = ((0., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        sheared = build_document(Capture('Model', (panel(world=shear),), None))
        self.assertEqual(corners(sheared, 0)[2]['position'].tolist(), [.25, 1., 0.])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mesh.smd'
            path.write_bytes(b'previous geometry\x00')
            with self.assertRaisesRegex(ValueError, 'singular'):
                write_smd(Capture('Model', (panel(world=singular),), None), path)
            self.assertEqual(path.read_bytes(), b'previous geometry\x00')
            rig = Rig('Rig', IDENTITY, (Bone('Root', None, IDENTITY),))
            invalid = panel(weights=((('missing', 1.),),) * 3)
            with self.assertRaisesRegex(ValueError, 'invalid weight'):
                build_document(Capture('Model', (invalid,), rig))
            def partial_dump(_, stream):
                stream.write('version 1\npartial')
                raise OSError('disk full')
            with patch.object(smd, 'dump', partial_dump):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    write_smd(Capture('Model', (panel(),), None), path)
            self.assertEqual(path.read_bytes(), b'previous geometry\x00')
            self.assertEqual([item.name for item in path.parent.iterdir()], ['mesh.smd'])


if __name__ == '__main__':
    unittest.main()
