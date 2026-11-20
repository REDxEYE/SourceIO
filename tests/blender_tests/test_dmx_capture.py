"""Source 1 DMX capture checks; run with Blender in background mode."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_capture import (capture_geometry, capture_vertex_animations)
from SourceIO.blender_bindings.source1.model_export.dmx_types import VertexAnimationRange


class DMXCaptureTests(unittest.TestCase):
    def setUp(self):
        self.scene = bpy.context.scene
        self.frame = self.scene.frame_current
        active = bpy.context.view_layer.objects.active
        self.active_name = active.name if active else None
        self.selected_names = tuple(obj.name for obj in bpy.context.selected_objects)
        self.collection = bpy.data.collections.new('Capture tests')
        bpy.context.scene.collection.children.link(self.collection)
        self.objects = []
        self.meshes = []
        self.curves = []
        self.armatures = []
        self.materials = []
        self.collections = [self.collection]

    def tearDown(self):
        for obj in self.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for data in self.meshes:
            bpy.data.meshes.remove(data)
        for data in self.curves:
            bpy.data.curves.remove(data)
        for data in self.armatures:
            bpy.data.armatures.remove(data)
        for material in self.materials:
            bpy.data.materials.remove(material)
        for collection in reversed(self.collections):
            bpy.data.collections.remove(collection, do_unlink=True)
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for name in self.selected_names:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                obj.select_set(True)
        bpy.context.view_layer.objects.active = (bpy.data.objects.get(self.active_name) if self.active_name else None)
        self.scene.frame_set(self.frame)

    def mesh(self, name, verts=((0, 0, 0), (1, 0, 0), (0, 1, 0)), faces=((0, 1, 2),), collection=None):
        data = bpy.data.meshes.new(name + ' data')
        data.from_pydata(verts, (), faces)
        obj = bpy.data.objects.new(name, data)
        (collection or self.collection).objects.link(obj)
        self.objects.append(obj)
        self.meshes.append(data)
        return obj

    def counts(self):
        return (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.collections), len(bpy.data.shape_keys))

    def test_nested_mesh_curve_material_uv_and_original_state(self):
        nested = bpy.data.collections.new('Nested capture')
        self.collection.children.link(nested)
        self.collections.append(nested)
        mesh = self.mesh('Z panel', ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 1, 2), (0, 2, 3)), nested)
        base = bpy.data.materials.new('Original base')
        override = bpy.data.materials.new('Object override')
        self.materials.extend((base, override))
        mesh.data.materials.append(base)
        mesh.data.materials.append(base)
        mesh.material_slots[1].link = 'OBJECT'
        mesh.material_slots[1].material = override
        mesh.data.polygons[1].material_index = 1
        layer = mesh.data.uv_layers.new(name='UVMap')
        for index, uv in enumerate(((0, 0), (1, 0), (1, 1), (0.5, 0), (1, 1), (0, 1))):
            layer.data[index].uv = uv
        modifier = mesh.modifiers.new('Displacement', 'DISPLACE')
        modifier.mid_level = 0
        modifier.strength = 0.125
        mesh.scale.x = -1
        mesh.location = (3, 2, 1)

        curve_data = bpy.data.curves.new('Curved pipe', 'CURVE')
        curve_data.dimensions = '3D'
        curve_data.bevel_depth = 0.1
        curve_data.bevel_resolution = 1
        spline = curve_data.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (0, 0, 0, 1)
        spline.points[1].co = (0, 0, 1, 1)
        curve = bpy.data.objects.new('A pipe', curve_data)
        self.collection.objects.link(curve)
        self.objects.append(curve)
        self.curves.append(curve_data)

        scene = bpy.context.scene
        scene.frame_set(7)
        before = self.counts()
        capture = capture_geometry(self.collection, material_names={base.name: 'bound_base',
                                                                    override.name: 'bound_override'})
        self.assertEqual(self.counts(), before)
        self.assertEqual(scene.frame_current, 7)
        self.assertEqual([item.name for item in capture.meshes], ['A pipe', 'Z panel'])
        pipe, panel = capture.meshes
        self.assertTrue(any(uv != (0.0, 0.0) for face in pipe.faces for uv in face.uvs))
        self.assertEqual([face.material for face in panel.faces], ['bound_base', 'bound_override'])
        self.assertNotEqual(panel.faces[0].uvs[0], panel.faces[1].uvs[0])
        np.testing.assert_allclose(panel.positions[0], (0, 0, 0.125))
        self.assertAlmostEqual(panel.matrix_world[0][0], -1)
        self.assertAlmostEqual(panel.matrix_world[0][3], 3)
        self.assertEqual(tuple(mesh.data.vertices[0].co), (0, 0, 0))
        self.assertIs(mesh.material_slots[1].material, override)
        self.assertEqual(modifier.strength, 0.125)

    def test_missing_uvs_project_continuously_across_coplanar_faces(self):
        obj = self.mesh('No UV quad', ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)), ((0, 1, 2), (0, 2, 3)))
        faces = capture_geometry(obj).meshes[0].faces
        by_vertex = {}
        for face in faces:
            for vertex, uv in zip(face.vertices, face.uvs):
                self.assertEqual(by_vertex.setdefault(vertex, uv), uv)
        self.assertEqual(set(by_vertex.values()), {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)})

    def test_shape_sampling_preserves_values_and_cleans_copies(self):
        obj = self.mesh('Shape face')
        obj.shape_key_add(name='Basis')
        smile = obj.shape_key_add(name='smile')
        smile.data[0].co.z += 0.5
        smile.value = 0.35
        smile.mute = True
        obj.active_shape_key_index = 1
        obj.show_only_shape_key = True
        displace = obj.modifiers.new('Raised', 'DISPLACE')
        displace.mid_level = 0
        displace.strength = 0.125
        scene = bpy.context.scene
        scene.frame_set(11)
        before = self.counts()
        result = capture_geometry(obj)
        self.assertEqual(self.counts(), before)
        self.assertEqual(scene.frame_current, 11)
        self.assertAlmostEqual(smile.value, 0.35)
        self.assertTrue(smile.mute)
        self.assertTrue(obj.show_only_shape_key)
        self.assertEqual(displace.strength, 0.125)
        base = result.meshes[0]
        self.assertEqual([shape.name for shape in base.shapes], ['smile'])
        np.testing.assert_allclose(base.positions[0], (0, 0, 0.125))
        self.assertGreater(base.shapes[0].positions[0][2] - base.positions[0][2], 0.45)
        self.assertNotEqual(base.faces[0].normals, tuple(map(tuple, base.shapes[0].normals[0].tolist())))

    def test_absolute_shape_keys_sample_their_frames(self):
        obj = self.mesh('Absolute face')
        obj.shape_key_add(name='Basis')
        morph = obj.shape_key_add(name='morph')
        morph.data[0].co.z += 0.4
        keys = obj.data.shape_keys
        keys.use_relative = False
        keys.eval_time = 5.0
        before = self.counts()
        result = capture_geometry(obj).meshes[0]
        self.assertEqual(self.counts(), before)
        self.assertEqual(keys.eval_time, 5.0)
        np.testing.assert_allclose(result.positions[0], (0, 0, 0))
        np.testing.assert_allclose(result.shapes[0].positions[0], (0, 0, 0.4))

    def test_rig_weights_parent_bone_and_unweighted_rejection(self):
        rig_data = bpy.data.armatures.new('Capture rig data')
        rig = bpy.data.objects.new('Capture rig', rig_data)
        self.collection.objects.link(rig)
        self.objects.append(rig)
        self.armatures.append(rig_data)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        root = rig_data.edit_bones.new('Root')
        root.head, root.tail = (0, 0, 0), (0, 1, 0)
        child = rig_data.edit_bones.new('Child')
        child.head, child.tail, child.parent = (0, 1, 0), (0, 2, 0), root
        bpy.ops.object.mode_set(mode='OBJECT')
        mesh = self.mesh('Weighted face')
        root_group = mesh.vertex_groups.new(name='Root')
        child_group = mesh.vertex_groups.new(name='Child')
        root_group.add([0, 1], 1.0, 'REPLACE')
        child_group.add([1, 2], 0.5, 'REPLACE')
        mesh.parent = rig
        mesh.parent_type = 'BONE'
        mesh.parent_bone = 'Child'
        modifier = mesh.modifiers.new('Armature', 'ARMATURE')
        modifier.object = rig
        result = capture_geometry(self.collection)
        self.assertEqual([bone.name for bone in result.rig.bones], ['Root', 'Child'])
        self.assertEqual(result.rig.bones[1].parent, 'Root')
        self.assertEqual(result.meshes[0].parent_bone, 'Child')
        self.assertEqual(result.meshes[0].weights[1], (('Child', 0.5), ('Root', 1.0)))
        root_group.remove([0])
        with self.assertRaises(ValueError):
            capture_geometry(self.collection)

    def test_shape_that_retessellates_a_quad_keeps_base_triangles(self):
        obj = self.mesh('Dented quad', ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 1, 2, 3),))
        obj.shape_key_add(name='Basis')
        key = obj.shape_key_add(name='dent')
        key.data[1].co = (0.3, 0.6, 0)  # reflex corner: Blender now splits along 1-3
        key.value = 0
        result = capture_geometry(obj).meshes[0]
        self.assertEqual([face.vertices for face in result.faces], [(0, 1, 2), (0, 2, 3)])
        shape = result.shapes[0]
        self.assertEqual([round(value, 5) for value in shape.positions[1]], [0.3, 0.6, 0.0])
        self.assertEqual(len(shape.normals), len(result.faces))
        self.assertEqual(obj.data.shape_keys.key_blocks['dent'].value, 0)

    def test_failed_shape_evaluation_releases_temporary_datablocks(self):
        obj = self.mesh('Welded face', ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 1, 2), (0, 2, 3)))
        obj.shape_key_add(name='Basis')
        key = obj.shape_key_add(name='collapse')
        key.data[3].co = key.data[0].co
        key.value = 0
        weld = obj.modifiers.new('Weld shape', 'WELD')
        weld.merge_threshold = 0.1
        before = self.counts()
        with self.assertRaises(ValueError):
            capture_geometry(obj)
        self.assertEqual(self.counts(), before)
        self.assertEqual(obj.data.shape_keys.key_blocks['collapse'].value, 0)
        self.assertAlmostEqual(weld.merge_threshold, 0.1)

    def test_collection_vertex_animation_frames_restore_scene_and_bind_space(self):
        obj = self.mesh('Animated face')
        old_fps, old_base = self.scene.render.fps, self.scene.render.fps_base
        self.addCleanup(lambda: setattr(self.scene.render, 'fps', old_fps))
        self.addCleanup(lambda: setattr(self.scene.render, 'fps_base', old_base))
        self.scene.render.fps = 24
        self.scene.render.fps_base = 1.001
        obj.location.z = 0.
        obj.keyframe_insert(data_path='location', frame=1)
        obj.location.z = .3
        obj.keyframe_insert(data_path='location', frame=2)
        self.scene.frame_set(1, subframe=.25)
        bind_z = obj.location.z
        before = self.counts()
        captured = capture_vertex_animations(self.collection, (VertexAnimationRange('pulse', 1, 3, True),))
        self.assertEqual(self.counts(), before)
        self.assertEqual((self.scene.frame_current, self.scene.frame_subframe), (1, .25))
        self.assertEqual(len(captured.vertex_animations), 1)
        animation = captured.vertex_animations[0]
        self.assertEqual((animation.name, animation.mesh_name, animation.export_sequence), ('pulse', obj.name, True))
        self.assertAlmostEqual(animation.frame_rate, self.scene.render.fps / self.scene.render.fps_base)
        self.assertEqual([frame.name for frame in animation.frames], ['pulse-0', 'pulse-1'])
        self.assertAlmostEqual(animation.frames[0].positions[0][2], -bind_z, delta=.001)
        self.assertAlmostEqual(animation.frames[1].positions[0][2], .3 - bind_z, delta=.001)
        np.testing.assert_allclose(animation.frames[0].normals[0], captured.meshes[0].faces[0].normals, atol=1e-6)
        self.assertEqual(tuple(obj.location), (0., 0., bind_z))


if __name__ == '__main__':
    unittest.main(argv=['test_dmx_capture'])
