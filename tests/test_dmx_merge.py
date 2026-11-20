"""Pure capture-level tests for opt-in collection coalescing."""

import unittest

import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_builder import build_document, build_vca_sequence
from SourceIO.blender_bindings.source1.model_export.dmx_merge import merge_collection_capture
from SourceIO.blender_bindings.source1.model_export.dmx_types import (
    Bone, Capture, Face, Mesh, Rig, Shape, VertexAnimation)


IDENTITY = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
NORMALS = ((0., 0., 1.),) * 3


def mesh(name, *, matrix=IDENTITY, material='material', uv=0., shapes=(), balance=None, weights=(), parent=None,
         parent_type=None, parent_bone=None):
    face = Face((0, 1, 2), NORMALS, ((uv, 0.), (1., 0.), (0., 1.)), material)
    return Mesh(name, matrix, parent, parent_type, parent_bone, ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
                (face,), weights, shapes, balance)


def moved(name, z):
    return Shape(name, ((0., 0., z), (1., 0., 0.), (0., 1., 0.)), (NORMALS,))


class DMXMergeTests(unittest.TestCase):
    def test_world_bake_material_uv_reflection_shapes_and_original_unchanged(self):
        reflected = ((-1., 0., 0., 3.), (0., 2., 0., 1.), (0., 0., 1., 0.), (0., 0., 0., 1.))
        first = mesh('Left', shapes=(moved('raise', 1.),), balance=(0., .5, 1.))
        second = mesh('Right', matrix=reflected, material='other', uv=.5, shapes=(moved('raise', 2.),),
                      balance=(1., .5, 0.))
        source = Capture('Collection', (first, second), None)
        merged = merge_collection_capture(source)
        self.assertEqual(source.meshes, (first, second))
        self.assertEqual(merged.meshes[0].name, 'Collection')
        self.assertEqual(merged.meshes[0].matrix_world, IDENTITY)
        self.assertEqual(merged.meshes[0].balance, (0., .5, 1., 1., .5, 0.))
        self.assertEqual([face.material for face in merged.meshes[0].faces], ['material', 'other'])
        self.assertEqual(merged.meshes[0].faces[1].vertices, (3, 5, 4))
        self.assertEqual(merged.meshes[0].faces[1].uvs[0], (.5, 0.))
        np.testing.assert_allclose(merged.meshes[0].positions[3], (3., 1., 0.))
        np.testing.assert_allclose(merged.meshes[0].shapes[0].positions[3], (3., 1., 2.))
        document = build_document(merged, encoding_version=4, model_version=15)
        shape = document.root['model']['children'][0]['shape']
        self.assertEqual(len(shape['faceSets']), 2)
        self.assertEqual(list(shape['deltaStates'][0]['positionsIndices']), [0, 3])

    def test_missing_shape_and_auto_balance_use_model_x(self):
        first = mesh('A', shapes=(moved('raise', 1.),))
        # Moved 100 units to the character's left (+X).
        second = mesh('B', matrix=((1., 0., 0., 100.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)))
        merged = merge_collection_capture(Capture('Collection', (first, second), None))
        self.assertEqual(merged.meshes[0].shapes[0].positions[3:], merged.meshes[0].positions[3:])
        # A straddles the middle; all of B is on the left.
        self.assertEqual(merged.meshes[0].balance, (0.5, 0.0, 0.5, 0.0, 0.0, 0.0))
        document = build_document(merged, encoding_version=4, model_version=15)
        delta = document.root['model']['children'][0]['shape']['deltaStates'][0]
        self.assertEqual(list(delta['positionsIndices']), [0])

    def test_vca_frames_are_merged_and_sequence_is_still_exportable(self):
        first = mesh('A')
        second = mesh('B', matrix=((1., 0., 0., 5.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)))
        a = VertexAnimation('pulse', 'A', (moved('pulse-0', 1.), moved('pulse-1', 2.)), True, 24.)
        b = VertexAnimation('pulse', 'B', (moved('pulse-0', 3.), moved('pulse-1', 4.)), True, 24.)
        merged = merge_collection_capture(Capture('Collection', (first, second), None, (a, b)))
        self.assertEqual(len(merged.vertex_animations), 1)
        clip = merged.vertex_animations[0]
        self.assertEqual(clip.mesh_name, 'Collection')
        self.assertEqual(clip.frames[1].positions[3], (5., 0., 4.))
        document = build_document(merged, encoding_version=4, model_version=15)
        shape = next(child['shape'] for child in document.root['model']['children'] if child.name == 'Collection')
        self.assertEqual(len(shape['deltaStates']), 2)
        sequence = build_vca_sequence(merged, 'pulse', encoding_version=4, model_version=15)
        self.assertIn('animationList', sequence.root)

    def test_rig_weights_survive_and_missing_clip_is_rejected(self):
        rig = Rig('Rig', IDENTITY, (Bone('Root', None, IDENTITY),))
        weights = ((('Root', 1.),),) * 3
        first = mesh('A', weights=weights)
        second = mesh('B', weights=weights, parent='Rig', parent_type='BONE', parent_bone='Root')
        merged = merge_collection_capture(Capture('Collection', (first, second), rig))
        self.assertEqual(merged.meshes[0].weights, weights + weights)
        document = build_document(merged, encoding_version=4, model_version=15)
        self.assertEqual(document.root['model']['children'][-1]['shape']['bindState']['jointCount'], 1)
        clip = VertexAnimation('pulse', 'A', (moved('pulse-0', 1.),))
        with self.assertRaisesRegex(ValueError, 'does not cover every merged mesh'):
            merge_collection_capture(Capture('Collection', (first, second), rig, (clip,)))

    def test_single_mesh_only_renames_and_retargets_clip(self):
        source = Capture('Collection', (mesh('A'),), None, (VertexAnimation('pulse', 'A', (moved('pulse-0', 1.),)),))
        merged = merge_collection_capture(source)
        self.assertIs(merged.meshes[0].positions, source.meshes[0].positions)
        self.assertEqual(merged.vertex_animations[0].mesh_name, 'Collection')
        self.assertIs(merge_collection_capture(Capture('A', source.meshes, None)).meshes[0], source.meshes[0])

    def test_shape_missing_on_a_member_is_a_zero_delta_there(self):
        first = mesh('A', shapes=(moved('A', 1.), moved('B', 2.), moved('A+B', 4.)))
        second = mesh('B', shapes=(moved('A', 1.), moved('B', 2.)))
        merged = merge_collection_capture(Capture('Collection', (first, second), None))
        self.assertEqual(merged.meshes[0].shapes[2].positions[3], merged.meshes[0].positions[3])
        document = build_document(merged, encoding_version=4, model_version=15)
        delta = document.root['model']['children'][0]['shape']['deltaStates'][2]
        self.assertEqual(list(delta['positionsIndices']), [0])

    def test_shape_conflicts_fail_before_writing(self):
        first = mesh('A', shapes=(moved('pulse-0', 1.),))
        second = mesh('B', shapes=(moved('pulse-0', 1.),))
        animation = VertexAnimation('pulse', 'A', (moved('pulse-0', 2.),))
        other = VertexAnimation('pulse', 'B', (moved('pulse-0', 3.),))
        with self.assertRaisesRegex(ValueError, 'collides with a captured shape'):
            merge_collection_capture(Capture('Collection', (first, second), None, (animation, other)))


if __name__ == '__main__':
    unittest.main()
