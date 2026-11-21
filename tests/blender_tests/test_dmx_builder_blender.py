"""Run with Blender in background mode; verifies capture-to-document geometry."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_capture import (capture_geometry, capture_vertex_animations)
from SourceIO.blender_bindings.source1.model_export.dmx_builder import (_split_transform, build_document, write_dmx,
                                                                        write_vca_sequences)
from SourceIO.blender_bindings.source1.model_export.dmx_types import VertexAnimationRange
from SourceIO.blender_bindings.source1.model_export.prop_qc import (
    FlexControllerRecipe, GeometryRecipe, ModelEntryRecipe)
from SourceIO.library.utils import datamodel_v2


class NativeDMXIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.active = bpy.context.view_layer.objects.active
        self.selected = tuple(bpy.context.selected_objects)
        self.collection = bpy.data.collections.new('Native DMX test')
        bpy.context.scene.collection.children.link(self.collection)

    def tearDown(self):
        for obj in tuple(self.collection.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(self.collection, do_unlink=True)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in self.selected:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.active

    def mesh(self, name):
        data = bpy.data.meshes.new(name + ' data')
        data.from_pydata(((0, 0, 0), (1, 0, 0), (0, 1, 0)), (), ((0, 1, 2),))
        obj = bpy.data.objects.new(name, data)
        self.collection.objects.link(obj)
        self.addCleanup(lambda: bpy.data.meshes.remove(data))
        return obj

    def write(self, capture, encoding, model_version):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'geometry.dmx'
            write_dmx(capture, path, encoding_version=encoding, model_version=model_version)
            return datamodel_v2.load(filepath=path)

    def test_rotated_mirror_splits_and_real_shear_is_rejected(self):
        cosine, sine = np.cos(.5), np.sin(.5)
        rotation = np.array(((cosine, -sine, 0.), (sine, cosine, 0.), (0., 0., 1.)))
        for scale in ((-1., 1., 1.), (-1., 2., .5), (1., -1., 1.)):
            with self.subTest(scale=scale):
                matrix = np.eye(4)
                matrix[:3, :3] = rotation @ np.diag(scale)
                rigid, residual = _split_transform(matrix, 'panel')
                self.assertGreater(np.linalg.det(rigid[:3, :3]), 0)
                np.testing.assert_allclose(rigid[:3, :3] @ residual, matrix[:3, :3], atol=1e-6)
                np.testing.assert_allclose(residual, np.diag(np.diag(residual)), atol=1e-6)
        shear = np.eye(4)
        shear[:3, :3] = rotation @ np.diag((-1., 1., 1.)) @ ((1., .25, 0.), (0., 1., 0.), (0., 0., 1.))
        with self.assertRaisesRegex(ValueError, 'shear'):
            _split_transform(shear, 'panel')

    def test_rotated_mirror_object_exports(self):
        obj = self.mesh('Mirror')
        obj.scale = (-1, 1, 1)
        obj.rotation_euler.z = .5
        document = self.write(capture_geometry(obj), 5, 18)
        self.assertEqual(document.root['model']['children'][0].name, 'Mirror')

    def test_named_collection_materials_transform_and_shapes(self):
        left = self.mesh('Panel Left')
        right = self.mesh('Panel Right')
        right.location = (2, 3, 1)
        right.scale = (-1, 1.5, 1)
        right.rotation_euler.z = 0.5
        material = bpy.data.materials.new('Bound')
        override = bpy.data.materials.new('Override')
        self.addCleanup(lambda: bpy.data.materials.remove(material))
        self.addCleanup(lambda: bpy.data.materials.remove(override))
        right.data.materials.append(material)
        right.material_slots[0].link = 'OBJECT'
        right.material_slots[0].material = override
        right.shape_key_add(name='Basis')
        key = right.shape_key_add(name='smile')
        key.data[0].co.z += .2
        before = (tuple(right.location), tuple(right.scale), key.value)
        capture = capture_geometry(self.collection, material_names={'Override': 'mapped_override'})
        self.assertEqual([item.name for item in capture.meshes], ['Panel Left', 'Panel Right'])
        self.assertEqual(before, (tuple(right.location), tuple(right.scale), key.value))
        for encoding, model_version in ((2, 1), (4, 15), (5, 18)):
            with self.subTest(encoding=encoding):
                document = self.write(capture, encoding, model_version)
                children = document.root['model']['children']
                self.assertEqual([item.name for item in children], ['Panel Left', 'Panel Right'])
                left_faces = children[0]['shape']['faceSets']
                self.assertEqual(left_faces[0]['material']['mtlName'], 'no_material')
                right_mesh = children[1]['shape']
                self.assertEqual(right_mesh['faceSets'][0]['material']['mtlName'], 'mapped_override')
                self.assertEqual(right_mesh['deltaStates'][0].name, 'smile')
                np.testing.assert_allclose(right_mesh['deltaStates'][0]['positions'][0], (0, 0, .2), atol=1e-5)
                np.testing.assert_allclose(children[1]['transform']['position'], (2, 3, 1))

    def test_surface_layout_matches_per_corner_reference(self):
        obj = self.mesh('Layout')
        data = obj.data
        data.clear_geometry()
        data.from_pydata(((0, 0, 0), (1, 0, 0), (1.5, 1, 0), (.5, 1.7, .2), (-.5, 1, 0), (2, 0, 1), (2, 1, 0)), (),
                         ((0, 1, 2, 3, 4), (1, 5, 6, 2), (5, 6, 2), (0, 4, 3, 2, 1)))
        materials = [bpy.data.materials.new(name) for name in ('layout_a', 'layout_b')]
        for material in materials:
            data.materials.append(material)
            self.addCleanup(lambda material=material: bpy.data.materials.remove(material))
        for polygon, slot in zip(data.polygons, (1, 0, 1, 2)):  # slot 2 is past the end: no_material
            polygon.material_index = slot
        layer = data.uv_layers.new(name='UVMap')
        for index, loop in enumerate(data.loops):  # seams plus duplicate UVs across faces
            layer.data[index].uv = ((loop.vertex_index % 3) * .25, -0.0 if index % 2 else (index % 4) * .5)
        data.normals_split_custom_set_from_vertices([(0, 0, 1)] * 3 + [(.3, .4, .866)] * 4)
        obj.scale.x = -1
        capture = capture_geometry(obj)
        document = build_document(capture, encoding_version=5, model_version=18)
        state = document.root['model']['children'][0]['shape']['bindState']
        mesh = capture.meshes[0]
        corners = (0, 2, 1)  # the mirrored object flips winding
        positions, uvs, uv_indices, by_material = [], [], [], {}
        for face in mesh.faces:
            rows = by_material.setdefault(mesh.materials[face['material']] or 'no_material', [])
            for corner in corners:
                positions.append(int(face['vertices'][corner]))
                uv = tuple(face['uvs'][corner].tolist())
                if uv not in uvs:
                    uvs.append(uv)
                uv_indices.append(uvs.index(uv))
                rows.append(len(positions) - 1)
            rows.append(-1)
        self.assertEqual(list(state['positionsIndices']), positions)
        self.assertEqual(list(state['textureCoordinatesIndices']), uv_indices)
        np.testing.assert_allclose(np.asarray(state['textureCoordinates']), uvs, atol=1e-7)
        face_sets = document.root['model']['children'][0]['shape']['faceSets']
        self.assertEqual({item['material']['mtlName']: list(item['faces']) for item in face_sets}, by_material)
        self.assertEqual([item['material']['mtlName'] for item in face_sets], list(by_material))

    def test_curve_and_generated_uvs_survive_assembly(self):
        data = bpy.data.curves.new('Pipe data', 'CURVE')
        self.addCleanup(lambda: bpy.data.curves.remove(data))
        data.dimensions = '3D'
        data.bevel_depth = .1
        data.bevel_resolution = 1
        spline = data.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (0, 0, 0, 1)
        spline.points[1].co = (0, 0, 1, 1)
        pipe = bpy.data.objects.new('Pipe', data)
        self.collection.objects.link(pipe)
        capture = capture_geometry(pipe)
        document = self.write(capture, 4, 15)
        mesh = document.root['model']['children'][0]['shape']
        vertex = mesh['bindState']
        self.assertGreater(len(vertex['positions']), 8)
        self.assertTrue(any(abs(float(uv)) > 1e-6 for pair in vertex['textureCoordinates'] for uv in pair))
        self.assertEqual(mesh['faceSets'][0]['material']['mtlName'], 'no_material')

    def test_posed_armature_does_not_double_skin_bind_geometry(self):
        rig_data = bpy.data.armatures.new('Rig data')
        rig = bpy.data.objects.new('Rig', rig_data)
        self.collection.objects.link(rig)
        self.addCleanup(lambda: bpy.data.armatures.remove(rig_data))
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        root = rig_data.edit_bones.new('Root')
        root.head, root.tail = (0, 0, 0), (0, 1, 0)
        child = rig_data.edit_bones.new('Child')
        child.head, child.tail, child.parent = (0, 1, 0), (0, 2, 0), root
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = self.mesh('Weighted')
        obj.vertex_groups.new(name='Root').add((0, 1, 2), 1, 'REPLACE')
        armature = obj.modifiers.new('Armature', 'ARMATURE')
        armature.object = rig
        rig.pose.bones['Root'].location = (0, .5, 0)
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        try:
            posed = tuple(tuple(vertex.co) for vertex in evaluated.to_mesh().vertices)
        finally:
            evaluated.to_mesh_clear()
        self.assertNotEqual(posed, tuple(tuple(vertex.co) for vertex in obj.data.vertices))
        capture = capture_geometry(self.collection)
        np.testing.assert_allclose(capture.meshes[0].positions, tuple(tuple(vertex.co) for vertex in obj.data.vertices))
        document = self.write(capture, 5, 18)
        model = document.root['model']
        self.assertEqual([item.name for item in model['jointList']], ['blender_implicit', 'Root', 'Child', 'Weighted'])
        vertex = model['children'][-1]['shape']['bindState']
        np.testing.assert_allclose(vertex['positions'], capture.meshes[0].positions)
        self.assertEqual(list(vertex['jointIndices']), [1, 1, 1])
        attached = self.mesh('Attached')
        attached.vertex_groups.new(name='Root').add((0, 1, 2), 1, 'REPLACE')
        attached.parent = rig
        attached.parent_type = 'BONE'
        attached.parent_bone = 'Child'
        bpy.context.view_layer.update()
        pose_world = rig.matrix_world @ rig.pose.bones['Child'].matrix
        expected_local = pose_world.inverted() @ attached.matrix_world
        attached_capture = capture_geometry(self.collection)
        attached_document = self.write(attached_capture, 5, 18)
        child_joint = attached_document.root['model']['children'][1]['children'][0]
        attached_dag, = child_joint['children']
        np.testing.assert_allclose(attached_dag['transform']['position'], expected_local.translation, atol=1e-5)
        rig.pose.bones['Root'].location = (0, 0, 0)
        bpy.context.view_layer.update()

    def test_collection_capture_writes_direct_flex_and_animated_sequence(self):
        obj = self.mesh('Face')
        obj.shape_key_add(name='Basis')
        smile = obj.shape_key_add(name='smile')
        smile.data[0].co.z += .2
        obj.location.z = 0.
        obj.keyframe_insert(data_path='location', frame=1)
        obj.location.z = .3
        obj.keyframe_insert(data_path='location', frame=2)
        scene = bpy.context.scene
        original_frame, original_subframe = scene.frame_current, scene.frame_subframe
        self.addCleanup(lambda: scene.frame_set(original_frame, subframe=original_subframe))
        scene.frame_set(1)
        before = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.shape_keys), len(bpy.data.collections))
        captured = capture_vertex_animations(self.collection, (VertexAnimationRange('pulse', 1, 3, True),))
        self.assertEqual((len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.shape_keys),
                          len(bpy.data.collections)), before)
        entry = ModelEntryRecipe('studio', GeometryRecipe(obj, 'face'),
                                 flex_controllers=(FlexControllerRecipe('face', ('smile',)),))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'face.dmx'
            write_dmx(captured, path, encoding_version=5, model_version=18, entry=entry)
            sequence_paths = write_vca_sequences(captured, directory, encoding_version=5, model_version=18)
            document = datamodel_v2.load(filepath=path)
            sequence = datamodel_v2.load(filepath=sequence_paths[0])
        mesh = next(child['shape'] for child in document.root['model']['children'] if child.name == 'Face')
        self.assertEqual([delta.name for delta in mesh['deltaStates']], ['smile', 'pulse-0', 'pulse-1'])
        self.assertEqual([control.name for control in document.root['combinationOperator']['controls']],
                         ['face', 'pulse'])
        self.assertIs(document.root['combinationOperator']['targets'][0], mesh)
        self.assertEqual(list(mesh['deltaStates'][0]['positionsIndices']), [0])
        self.assertEqual(list(mesh['deltaStates'][1]['positionsIndices']), [])
        np.testing.assert_allclose(mesh['deltaStates'][2]['positions'], ((0, 0, .3),) * 3, atol=1e-5)
        self.assertNotIn('model', sequence.root)
        self.assertEqual(len(next(channel for channel in sequence.root['animationList']['animations'][0]['channels']
                                  if channel.name == 'vcabone_pulse_p')['log']['layers'][0]['times']), 2)


if __name__ == '__main__':
    unittest.main(argv=['test_dmx_builder_blender'])
