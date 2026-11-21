"""Source 1 DMX capture checks; run with Blender in background mode."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import bpy
import numpy as np

from SourceIO.blender_bindings.source1.model_export.dmx_capture import (capture_geometry, capture_vertex_animations,
                                                                   rest_world_matrix)
from SourceIO.blender_bindings.source1.model_export.dmx_types import VertexAnimationRange, face_array


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
        self.assertTrue(pipe.faces['uvs'].any())
        self.assertEqual([panel.materials[index] for index in panel.faces['material']], ['bound_base', 'bound_override'])
        self.assertFalse(np.array_equal(panel.faces['uvs'][0, 0], panel.faces['uvs'][1, 0]))
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
            for vertex, uv in zip(face['vertices'].tolist(), face['uvs'].tolist()):
                self.assertEqual(by_vertex.setdefault(vertex, tuple(uv)), tuple(uv))
        self.assertEqual(set(by_vertex.values()), {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)})

    def reference_faces(self, obj, names, with_uv):
        # Straightforward per-triangle transcription of the original sampler.
        data = obj.data
        data.calc_loop_triangles()
        corners = data.corner_normals
        positions = [tuple(map(float, vertex.co)) for vertex in data.vertices]
        origin = tuple(min(p[axis] for p in positions) for axis in range(3))
        span = tuple(max(p[axis] for p in positions) - origin[axis] or 1.0 for axis in range(3))
        faces = []
        for triangle in data.loop_triangles:
            loops = tuple(triangle.loops)
            if with_uv:
                uvs = tuple(tuple(map(float, data.uv_layers.active.data[i].uv)) for i in loops)
            else:
                axis = max(range(3), key=lambda index: abs(triangle.normal[index]))
                u, v = ((1, 2), (0, 2), (0, 1))[axis]
                uvs = tuple(((positions[i][u] - origin[u]) / span[u], (positions[i][v] - origin[v]) / span[v])
                            for i in triangle.vertices)
            slot = data.polygons[triangle.polygon_index].material_index
            material = obj.material_slots[slot].material if slot < len(obj.material_slots) else None
            faces.append((tuple(triangle.vertices),
                          tuple(tuple(map(float, corners[i].vector)) for i in loops), uvs,
                          names.get(material.name, material.name) if material else None))
        return faces

    def test_vectorized_sampling_matches_per_triangle_reference(self):
        verts = ((0, 0, 0), (1, 0, 0), (1.5, 1, 0), (0.5, 1.7, 0.2), (-.5, 1, 0), (2, 0, 1), (2, 1, 0), (3, .5, 1))
        faces = ((0, 1, 2, 3, 4), (1, 5, 6, 2), (5, 7, 6), (0, 4, 3, 2, 1))
        for with_uv in (True, False):
            with self.subTest(with_uv=with_uv):
                obj = self.mesh(f'Reference {with_uv}', verts, faces)
                first, second = bpy.data.materials.new('ref first'), bpy.data.materials.new('ref second')
                self.materials.extend((first, second))
                obj.data.materials.append(first)
                obj.data.materials.append(second)
                obj.data.materials.append(None)
                for polygon, slot in zip(obj.data.polygons, (1, 0, 2, 3)):
                    polygon.material_index = slot  # 2 is an empty slot, 3 is past the end
                if with_uv:
                    layer = obj.data.uv_layers.new(name='UVMap')
                    for index, loop in enumerate(obj.data.loops):
                        # Seams: one UV per (vertex, parity), plus a signed zero.
                        layer.data[index].uv = (loop.vertex_index * .125 + (index % 2) * .5, -0.0 if index % 3 else .75)
                obj.data.normals_split_custom_set_from_vertices([(0, 0, 1)] * 3 + [(.3, .4, .866)] * 5)
                obj.scale.x = -1
                names = {'ref first': 'named_first'}
                result = capture_geometry(obj, material_names=names).meshes[0]
                expected = self.reference_faces(obj, names, with_uv)
                rows, _ = face_array(expected)  # float32 like the captured arrays
                self.assertEqual([result.materials[index] for index in result.faces['material']],
                                 [row[3] for row in expected])
                for field in ('vertices', 'normals', 'uvs'):
                    np.testing.assert_array_equal(result.faces[field], rows[field])

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
        self.assertFalse(np.array_equal(base.faces['normals'][0], base.shapes[0].normals[0]))

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

    def test_driven_and_narrow_slider_shapes_sample_full_range(self):
        obj = self.mesh('Driven face')
        obj.shape_key_add(name='Basis')
        driven = obj.shape_key_add(name='driven')
        narrow = obj.shape_key_add(name='narrow')
        driven.data[0].co.z += 0.4
        narrow.data[1].co.z += 0.4
        keys = obj.data.shape_keys
        curve = keys.driver_add('key_blocks["driven"].value')
        curve.driver.type = 'SCRIPTED'
        curve.driver.expression = '0'
        narrow.slider_min = 0.2
        narrow.slider_max = 0.5
        narrow.value = 0.3
        before = self.counts()
        result = capture_geometry(obj).meshes[0]
        self.assertEqual(self.counts(), before)
        np.testing.assert_allclose(result.shapes[0].positions[0], (0, 0, 0.4), atol=1e-5)
        np.testing.assert_allclose(result.shapes[1].positions[1], (1, 0, 0.4), atol=1e-5)
        self.assertEqual(len(keys.animation_data.drivers), 1)
        np.testing.assert_allclose((narrow.slider_min, narrow.slider_max), (0.2, 0.5), atol=1e-6)
        self.assertAlmostEqual(narrow.value, 0.3)

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
        with self.assertRaises(ValueError):  # partly weighted: still an error
            capture_geometry(self.collection)
        mesh.vertex_groups.clear()
        result = capture_geometry(self.collection)  # rigid on its parent bone
        self.assertEqual(set(result.meshes[0].weights), {(('Child', 1.0),)})

    def test_shape_that_retessellates_a_quad_keeps_base_triangles(self):
        obj = self.mesh('Dented quad', ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 1, 2, 3),))
        obj.shape_key_add(name='Basis')
        key = obj.shape_key_add(name='dent')
        key.data[1].co = (0.3, 0.6, 0)  # reflex corner: Blender now splits along 1-3
        key.value = 0
        result = capture_geometry(obj).meshes[0]
        self.assertEqual(result.faces['vertices'].tolist(), [[0, 1, 2], [0, 2, 3]])
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
        np.testing.assert_allclose(animation.frames[0].normals[0], captured.meshes[0].faces['normals'][0], atol=1e-6)
        self.assertEqual(tuple(obj.location), (0., 0., bind_z))

    def test_children_of_bone_parented_meshes_ignore_the_pose(self):
        rig_data = bpy.data.armatures.new('Chain rig data')
        rig = bpy.data.objects.new('Chain rig', rig_data)
        self.collection.objects.link(rig)
        self.objects.append(rig)
        self.armatures.append(rig_data)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bone = rig_data.edit_bones.new('Root')
        bone.head, bone.tail = (0, 0, 0), (0, 1, 0)
        bpy.ops.object.mode_set(mode='OBJECT')
        holder = self.mesh('Holder')
        holder.parent, holder.parent_type, holder.parent_bone = rig, 'BONE', 'Root'
        child = self.mesh('Child')
        child.parent = holder
        child.location = (0, 0, 2)
        for item in (holder, child):
            item.vertex_groups.new(name='Root').add([0, 1, 2], 1.0, 'REPLACE')
        bpy.context.view_layer.update()
        rest = [tuple(map(tuple, rest_world_matrix(item))) for item in (holder, child)]
        pose = rig.pose.bones['Root']
        pose.rotation_mode = 'XYZ'
        pose.rotation_euler.x = 1.0
        pose.location.x = 3.0
        bpy.context.view_layer.update()
        self.assertFalse(np.allclose(child.matrix_world, rest[1], atol=.1))
        np.testing.assert_allclose([tuple(map(tuple, rest_world_matrix(item))) for item in (holder, child)], rest, atol=1e-5)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        np.testing.assert_allclose(rest_world_matrix(child, depsgraph), rest[1], atol=1e-5)
        captured = {mesh.name: mesh for mesh in capture_geometry(self.collection).meshes}
        np.testing.assert_allclose(captured['Child'].matrix_world, rest[1], atol=1e-5)

    def test_vertex_animation_keeps_animated_shape_keys_and_zeroes_static_ones(self):
        obj = self.mesh('Keyed face')
        obj.shape_key_add(name='Basis')
        keyed = obj.shape_key_add(name='keyed')
        driven = obj.shape_key_add(name='driven')
        static = obj.shape_key_add(name='static')
        keyed.data[0].co.z += 0.4
        driven.data[1].co.z += 0.2
        static.data[2].co.z += 0.6
        static.value = 0.5
        keys = obj.data.shape_keys
        keyed.value = 0.0
        keyed.keyframe_insert('value', frame=1)
        keyed.value = 1.0
        keyed.keyframe_insert('value', frame=2)
        curve = keys.driver_add('key_blocks["driven"].value')
        curve.driver.type = 'SCRIPTED'
        curve.driver.expression = 'frame / 2'
        before = self.counts()
        animation = capture_vertex_animations(obj, (VertexAnimationRange('pulse', 1, 3),)).vertex_animations[0]
        self.assertEqual(self.counts(), before)
        self.assertEqual(len(keys.animation_data.drivers), 1)
        self.assertEqual((static.value, static.mute), (0.5, False))
        first, second = animation.frames
        for frame in animation.frames:
            self.assertEqual(frame.positions.dtype, np.float32)
            self.assertEqual(frame.positions.shape, (3, 3))
            self.assertFalse(frame.positions.flags.writeable)
            self.assertAlmostEqual(frame.positions[2][2], 0.0, places=5)  # static key stays zeroed
        np.testing.assert_allclose((first.positions[0][2], second.positions[0][2]), (0.0, 0.4), atol=1e-5)
        np.testing.assert_allclose((first.positions[1][2], second.positions[1][2]), (0.1, 0.2), atol=1e-5)


if __name__ == '__main__':
    unittest.main(argv=['test_dmx_capture'])
