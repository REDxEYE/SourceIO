"""Behavioral Source 1 export graph checks run inside Blender's Python."""

import tempfile
import unittest
from math import radians
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import bpy
import numpy as np
from mathutils import Matrix, Vector

from SourceIO.blender_bindings.source1.model_export import procedural_bones
from SourceIO.blender_bindings.source1.model_export.sfm_rig import detect_biped
from SourceIO.blender_bindings.ui.export_nodes.nodes import (procedural_bone_nodes, prop_nodes, sfm_rig_nodes)
from SourceIO.blender_bindings.ui.export_nodes import register_nodes, unregister_nodes
from SourceIO.blender_bindings.source1.model_export.bst_bridge import SourceToolsError
from SourceIO.blender_bindings.ui.export_nodes.build_graph import (
    GraphBuildError, GraphResolver, _GraphAction, compile_plan, export_materials, export_plan, model_cache, plan_model,
    store_direct_issues, write_compile_log)
from SourceIO.blender_bindings.source1.model_export.compiler import CompilationError, direct_command
from SourceIO.blender_bindings.source1.model_export import compiler as export_compiler
from SourceIO.blender_bindings.source1.model_export.material_assets import texture_pixels
from SourceIO.library.utils.pylib.vtf import ImageFormat, TextureFlags, VTFFile, load_vtf_texture
from SourceIO.library.utils import datamodel_v2
from SourceIO.library.models import smd


def advanced_flex_text(shape, flex_max):
    """KeyValues2 DMX with one combination control, as authored for a Model entry's advanced flex Text."""
    document = datamodel_v2.DataModel('keyvalues2', 1, 'model', 1)
    root = document.add_element('flex', uuid='test/flex')
    operator = document.add_element('combinationOperator', 'DmeCombinationOperator', uuid='test/operator')
    root['combinationOperator'] = operator
    control = document.add_element(shape, 'DmeCombinationInputControl', uuid=f'test/control/{shape}')
    control['rawControlNames'] = datamodel_v2.StringArray([shape])
    control['stereo'] = False
    control['eyelid'] = False
    control['flexMin'] = 0.0
    control['flexMax'] = float(flex_max)
    control['wrinkleScales'] = datamodel_v2.FloatArray([0.0])
    operator['controls'] = datamodel_v2.ElementArray([control])
    operator['controlValues'] = datamodel_v2.Vector3Array([(0.0, 0.0, 0.5)])
    operator['targets'] = datamodel_v2.ElementArray()
    return datamodel_v2.dump(document, encoding='keyvalues2')


class ExportGraphTests(unittest.TestCase):
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
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.tree = bpy.data.node_groups.new('Source 1 prop test', 'sourceio.model_definition')
        self.addCleanup(lambda: bpy.data.node_groups.remove(bpy.data.node_groups['Source 1 prop test']))
        self.meshes = []
        self.objects = []
        self.materials = []
        self.images = []
        self.texts = []
        self.addCleanup(self._remove_data)
        self.output = self.tree.nodes.new('SourceIOPropModelNode')
        self.output.model_path = 'models/props/test.mdl'
        self.output.content_root = str(self.root / 'content')
        self.output.game_root = str(self.root / 'game')

    def _remove_data(self):
        for name in self.objects:
            obj = bpy.data.objects.get(name)
            if obj is not None:
                bpy.data.objects.remove(obj, do_unlink=True)
        for name in self.meshes:
            mesh = bpy.data.meshes.get(name)
            if mesh is not None:
                bpy.data.meshes.remove(mesh)
        for material in self.materials:
            if material.name in bpy.data.materials:
                bpy.data.materials.remove(material)
        for image in self.images:
            if image.name in bpy.data.images:
                bpy.data.images.remove(image)
        for text in self.texts:
            if text.name in bpy.data.texts:
                bpy.data.texts.remove(text)

    def mesh_node(self, basename, material=None):
        mesh = bpy.data.meshes.new(f'{basename}_mesh')
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        if material is not None:
            mesh.materials.append(material)
        obj = bpy.data.objects.new(f'{basename}_object', mesh)
        bpy.context.scene.collection.objects.link(obj)
        self.meshes.append(mesh.name)
        self.objects.append(obj.name)
        node = self.tree.nodes.new('SourceIOPropGeometryNode')
        node.basename = basename
        node.obj = obj
        return node

    def model_entry(self, geometry, name='studio', directive='MODEL', output=None):
        entry = self.tree.nodes.new('SourceIOModelEntryNode')
        entry.entry_name = name
        entry.directive = directive
        self.tree.links.new(geometry.outputs['Geometry'], entry.inputs['Geometry'])
        self.tree.links.new(entry.outputs['Model'], (output or self.output).inputs['Models'])
        return entry

    def material_node(self, material, filename):
        node = self.tree.nodes.new('SourceIOVertexLitGenericVMTNode')
        node.mat = material
        node.vmt_path = f'models/props/{filename}.vmt'
        return node

    def test_bodygroup_skin_lod_collision_and_raw_qc_render_in_graph_order(self):
        base = bpy.data.materials.new('base')
        self.materials.append(base)
        ref = self.mesh_node('reference', base)
        alternate = self.mesh_node('alternate', base)
        self.model_entry(ref, 'main')
        self.model_entry(alternate, 'secondary', 'BODY')
        bodygroup = self.tree.nodes.new('SourceIOPropBodygroupNode')
        bodygroup.bodygroup_name = 'sleeves'
        blank = self.tree.nodes.new('SourceIOPropBlankNode')
        self.tree.links.new(alternate.outputs['Geometry'], bodygroup.inputs['Choices'])
        self.tree.links.new(blank.outputs['Blank'], bodygroup.inputs['Choices'])
        self.tree.links.new(bodygroup.outputs[0], self.output.inputs['Bodygroups'])
        lod = self.tree.nodes.new('SourceIOPropLodNode')
        lod.distance = 32
        self.tree.links.new(ref.outputs[0], lod.inputs['Original mesh'])
        self.tree.links.new(alternate.outputs[0], lod.inputs['Replacement mesh'])
        self.tree.links.new(lod.outputs[0], self.output.inputs['LODs'])
        collision = self.tree.nodes.new('SourceIOPropCollisionNode')
        collision.mass = 12
        self.tree.links.new(ref.outputs[0], collision.inputs[0])
        self.tree.links.new(collision.outputs[0], self.output.inputs['Collision'])
        raw = self.tree.nodes.new('SourceIOPropRawQcNode')
        raw.snippet = '$surfaceprop "metal"'
        self.tree.links.new(raw.outputs[0], self.output.inputs['Raw QC'])
        base_vmt = self.material_node(base, 'base')
        replacement = self.material_node(base, 'red')
        self.tree.links.new(base_vmt.outputs[0], self.output.inputs['Materials'])
        self.tree.links.new(replacement.outputs[0], self.output.inputs['Materials'])
        for vmt in (base_vmt, replacement):
            skin = self.tree.nodes.new('SourceIOPropSkinFamilyNode')
            self.tree.links.new(vmt.outputs[0], skin.inputs['Materials'])
            self.tree.links.new(skin.outputs[0], self.output.inputs['Skin families'])
        plan = plan_model(self.output)
        self.assertEqual([item.key for item, _ in plan.geometry], ['alternate', 'reference'])
        qc = plan.qc_text
        self.assertIn('$modelname "props/test.mdl"', qc)
        self.assertLess(qc.index('$body "secondary" "alternate.dmx"'), qc.index('$model "main" "reference.dmx"'))
        self.assertIn('$bodygroup "sleeves" {\n    blank\n    studio "alternate.dmx"\n}', qc)
        self.assertIn('$texturegroup "skinfamilies" {\n    { "red" }\n    { "base" }\n}', qc)
        self.assertIn('$lod 32 {\n    replacemodel "reference.dmx" "alternate.dmx"\n}', qc)
        self.assertIn('$collisionmodel "reference.dmx" {\n    $mass 12\n}', qc)
        self.assertTrue(qc.endswith('$surfaceprop "metal"\n'))
        self.assertFalse(plan.qc_path.exists())  # Preview is side-effect-free.

    def test_native_dmx_graph_exports_flex_vca_and_qc_without_source_tools(self):
        import sys
        from unittest.mock import patch

        geometry = self.mesh_node('face')
        obj = geometry.obj
        obj.shape_key_add(name='Basis')
        obj.shape_key_add(name='smile').data[0].co.z += .2
        obj.location.z = 0
        obj.keyframe_insert(data_path='location', frame=1)
        obj.location.z = .3
        obj.keyframe_insert(data_path='location', frame=2)
        scene = bpy.context.scene
        frame, subframe = scene.frame_current, scene.frame_subframe
        self.addCleanup(lambda: scene.frame_set(frame, subframe=subframe))
        scene.frame_set(1)
        geometry.vertex_animation_ranges = 'pulse 1 3 sequence'
        geometry.subdir = 'nested'
        entry = self.model_entry(geometry)
        control = entry.flex_controllers.add()
        control.name = 'face'
        control.raw_name = 'smile'
        self.output.native_settings_confirmed = False
        with self.assertRaisesRegex(GraphBuildError, 'Native settings reviewed'):
            plan_model(self.output)
        self.output.native_settings_confirmed = True

        self.output.dmx_encoding_version = '2'
        self.output.dmx_model_version = '15'
        with self.assertRaisesRegex(GraphBuildError, 'encoding'):
            plan_model(self.output)
        self.assertFalse((self.root / 'content').exists())

        self.output.dmx_encoding_version = '4'
        snapshot = (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.collections), obj.location.z,
                    scene.frame_current)
        with patch.dict(sys.modules, {'io_scene_valvesource': None}):
            plan = plan_model(self.output)
            self.assertIn('$bonemerge "vcabone_pulse"', plan.qc_text)
            self.assertIn('$sequence "pulse" "nested/face/vcaanim_pulse.dmx" fps ', plan.qc_text)
            export_plan(plan)
        self.assertEqual(snapshot, (len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.collections),
                                    obj.location.z, scene.frame_current))
        geometry_path = plan.geometry[0][1]
        sequence_path = geometry_path.with_suffix('') / 'vcaanim_pulse.dmx'
        self.assertTrue(sequence_path.is_file())
        self.assertEqual(plan.qc_path.read_text(), plan.qc_text)
        document = datamodel_v2.load(filepath=geometry_path)
        sequence = datamodel_v2.load(filepath=sequence_path)
        self.assertEqual((document.encoding_ver, document.format_ver), (4, 15))
        self.assertEqual([control.name for control in document.root['combinationOperator']['controls']],
                         ['face', 'pulse'])
        mesh = next(child['shape'] for child in document.root['model']['children'] if child.name == obj.name)
        self.assertEqual([delta.name for delta in mesh['deltaStates']], ['smile', 'pulse-0', 'pulse-1'])
        np.testing.assert_allclose(mesh['deltaStates'][2]['positions'], ((0, 0, .3),) * 3, atol=1e-5)
        self.assertNotIn('model', sequence.root)
        channels = sequence.root['animationList']['animations'][0]['channels']
        self.assertEqual(len(next(channel for channel in channels
                                  if channel.name == 'vcabone_pulse_p')['log']['layers'][0]['times']), 2)

    def test_native_collection_auto_merge_and_y_up_preserve_shapes_and_materials(self):
        collection = bpy.data.collections.new('Merged panels')
        self.addCleanup(lambda: bpy.data.collections.remove(collection, do_unlink=True))
        bpy.context.scene.collection.children.link(collection)
        matte = bpy.data.materials.new('Original matte')
        shiny = bpy.data.materials.new('Original shiny')
        self.materials.extend((matte, shiny))
        left = self.mesh_node('left', matte)
        right = self.mesh_node('right', shiny)
        for material, filename in ((matte, 'matte'), (shiny, 'shiny')):
            vmt = self.material_node(material, filename)
            self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        collection.objects.link(left.obj)
        collection.objects.link(right.obj)
        right.obj.location.y = 2
        left.obj.shape_key_add(name='Basis')
        left.obj.shape_key_add(name='smile').data[0].co.z += .2
        left.source_type = 'COLLECTION'
        left.collection = collection
        left.auto_merge = True
        entry = self.model_entry(left)
        control = entry.flex_controllers.add()
        control.name = 'smile'
        control.raw_name = 'smile'
        self.output.dmx_up_axis = 'Y'
        plan = plan_model(self.output)
        self.assertIn('up axis: Y', plan.manifest())
        export_plan(plan)
        doc = datamodel_v2.load(filepath=plan.geometry[0][1])
        dag, = doc.root['model']['children']
        self.assertEqual(dag.name, collection.name)
        mesh = dag['shape']
        self.assertEqual(len(mesh['bindState']['positions']), 6)
        self.assertEqual([faces['material']['mtlName'] for faces in mesh['faceSets']], ['matte', 'shiny'])
        self.assertEqual([len(faces['faces']) for faces in mesh['faceSets']], [4, 4])
        self.assertEqual(mesh['deltaStates'][0].name, 'smile')
        self.assertEqual(list(mesh['deltaStates'][0]['positionsIndices']), [0])
        self.assertAlmostEqual(mesh['deltaStates'][0]['positions'][0][2], .2)
        self.assertEqual([control.name for control in doc.root['combinationOperator']['controls']], ['smile'])

    def test_native_dmx_failure_preserves_previous_geometry_file(self):
        geometry = self.mesh_node('face')
        geometry.stereo_balance_groups = 'face_object=missing_group'
        self.model_entry(geometry)
        plan = plan_model(self.output)
        target = plan.geometry[0][1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'previous complete DMX')
        with self.assertRaisesRegex(GraphBuildError, 'no stereo balance group'):
            export_plan(plan)
        self.assertEqual(target.read_bytes(), b'previous complete DMX')
        self.assertFalse(plan.qc_path.exists())

    def test_native_advanced_flex_text_exports_headless_without_bst_parser(self):
        import sys
        from unittest.mock import patch
        from SourceIO.library.utils import datamodel as kv2

        geometry = self.mesh_node('face')
        geometry.obj.shape_key_add(name='Basis')
        geometry.obj.shape_key_add(name='smile').data[0].co.z += .2
        entry = self.model_entry(geometry)
        source = kv2.DataModel('model', 15)
        root = source.add_element('root')
        operator = source.add_element('combinationOperator', 'DmeCombinationOperator')
        root['combinationOperator'] = operator
        control = source.add_element('faceControl', 'DmeCombinationInputControl')
        control['rawControlNames'] = kv2.make_array(['smile'], str)
        control['stereo'] = False
        control['eyelid'] = False
        control['flexMin'] = 0.
        control['flexMax'] = .6
        control['wrinkleScales'] = kv2.make_array([0.], float)
        operator['controls'] = kv2.make_array([control], kv2.Element)
        values = kv2.make_array([kv2.Vector3((0., 0., .5))], kv2.Vector3)
        operator['controlValues'] = values
        operator['controlValuesLagged'] = values
        operator['usesLaggedValues'] = False
        operator['dominators'] = kv2.make_array([], kv2.Element)
        old_mesh = source.add_element(geometry.obj.name, 'DmeMesh')
        operator['targets'] = kv2.make_array([old_mesh], kv2.Element)
        text = bpy.data.texts.new('Headless advanced flex')
        self.texts.append(text)
        text.from_string(source.echo('keyvalues2', 1))
        entry.flex_controller_text = text
        with patch.dict(sys.modules, {'io_scene_valvesource': None}):
            plan = plan_model(self.output)
            export_plan(plan)
        exported = datamodel_v2.load(filepath=plan.geometry[0][1])
        flex = exported.root['combinationOperator']
        mesh = exported.root['model']['children'][0]['shape']
        self.assertAlmostEqual(flex['controls'][0]['flexMax'], .6)
        self.assertIs(flex['targets'][0], mesh)

    def test_auto_merged_graph_rules_bind_collection_mesh(self):
        collection = bpy.data.collections.new('Rule collection')
        self.addCleanup(lambda: bpy.data.collections.remove(collection, do_unlink=True))
        bpy.context.scene.collection.children.link(collection)
        left = self.mesh_node('left')
        right = self.mesh_node('right')
        for node, shape in ((left, 'smile'), (right, 'blink')):
            collection.objects.link(node.obj)
            node.obj.shape_key_add(name='Basis')
            node.obj.shape_key_add(name=shape).data[0].co.z += .2
        geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
        geometry.source_type = 'COLLECTION'
        geometry.collection = collection
        geometry.basename = 'rules'
        geometry.auto_merge = True
        entry = self.model_entry(geometry)
        group = self.tree.nodes.new('SourceIOFlexGroupNode')
        inner = group.node_tree
        self.addCleanup(lambda: bpy.data.node_groups.remove(inner, do_unlink=True))
        self.tree.links.new(group.outputs['Flex Rules'], entry.inputs['Flex Rules'])
        plan = plan_model(self.output)
        export_plan(plan)
        exported = datamodel_v2.load(filepath=plan.geometry[0][1])
        dag, = exported.root['model']['children']
        targets = exported.root['combinationOperator']['targets']
        self.assertEqual(dag.name, collection.name)
        self.assertEqual(len(targets), 1)
        self.assertIs(targets[0]['target'], dag['shape'])
        self.assertEqual({rule.name for rule in targets[0]['deltaStates']}, {'smile', 'blink'})

    def test_export_cache_rewrites_only_changed_or_externally_modified_meshes(self):
        first = self.mesh_node('first')
        second = self.mesh_node('second')
        self.model_entry(first, name='first')
        self.model_entry(second, name='second')

        def export():
            cache = model_cache(self.output)
            plan = plan_model(self.output, cache=cache)
            export_plan(plan, cache)
            cache.save()
            return {path.stem: path.stat().st_mtime_ns for _, path in plan.geometry}, plan

        before, plan = export()
        self.assertEqual(export()[0], before)
        first.obj.data.vertices[0].co.z += 0.5
        after, _ = export()
        self.assertNotEqual(after['first'], before['first'])
        self.assertEqual(after['second'], before['second'])
        second_path = next(path for _, path in plan.geometry if path.stem == 'second')
        second_path.write_bytes(second_path.read_bytes() + b' ')
        self.assertNotEqual(export()[0]['second'], after['second'])
        self.output.use_export_cache = False
        self.assertIsNone(model_cache(self.output))

    def test_geometry_basename_defaults_to_selected_source_name(self):
        geometry = self.mesh_node('prop')
        geometry.basename = ''
        self.model_entry(geometry)
        self.assertEqual(plan_model(self.output).geometry[0][1].name, 'prop_object.dmx')

        geometry.obj.name = 'renamed_prop'
        self.objects[-1] = 'renamed_prop'
        self.assertEqual(plan_model(self.output).geometry[0][1].name, 'renamed_prop.dmx')
        geometry.basename = 'custom'
        self.assertEqual(plan_model(self.output).geometry[0][1].name, 'custom.dmx')

        collection = bpy.data.collections.new('prop_collection')
        collection_name = collection.name
        self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections[collection_name], do_unlink=True))
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(geometry.obj)
        geometry.source_type = 'COLLECTION'
        geometry.collection = collection
        geometry.basename = ''
        self.assertEqual(plan_model(self.output).geometry[0][1].name, 'prop_collection.dmx')

    def test_geometry_to_models_link_inserts_model_entry_without_loosening_socket_types(self):
        existing = self.model_entry(self.mesh_node('other'), name='crate')
        geometry = self.mesh_node('crate')
        geometry.location = (0, 0)
        self.output.location = (500, 0)

        self.tree.links.new(geometry.outputs['Geometry'], self.output.inputs['Models'])
        entries = [node for node in self.tree.nodes if node.bl_idname == 'SourceIOModelEntryNode']
        self.assertEqual(len(entries), 2)
        inserted = next(node for node in entries if node != existing)
        self.assertEqual(inserted.entry_name, 'crate.001')
        self.assertEqual(tuple(inserted.location), (250, 0))
        self.assertEqual(inserted.inputs['Geometry'].links[0].from_node, geometry)
        self.assertTrue(any(link.from_node == inserted for link in self.output.inputs['Models'].links))
        qc = plan_model(self.output).qc_text
        self.assertIn('$model "crate" "other.dmx"', qc)
        self.assertIn('$model "crate.001" "crate.dmx"', qc)

        bodygroup = self.tree.nodes.new('SourceIOPropBodygroupNode')
        self.tree.links.new(geometry.outputs['Geometry'], bodygroup.inputs['Choices'])
        self.assertEqual(bodygroup.inputs['Choices'].links[0].from_node, geometry)
        invalid = self.tree.nodes.new('SourceIOImageSourceNode')
        self.tree.links.new(invalid.outputs['Texture'], self.output.inputs['Models'])
        self.assertFalse(any(link.from_node == invalid for link in self.output.inputs['Models'].links))
        self.assertEqual(len([node for node in self.tree.nodes if node.bl_idname == 'SourceIOModelEntryNode']), 2)

    def test_geometry_to_bodygroups_inserts_named_bodygroup_with_choice(self):
        self.model_entry(self.mesh_node('main'))
        existing = self.tree.nodes.new('SourceIOPropBodygroupNode')
        existing.bodygroup_name = 'variants'
        self.tree.links.new(self.mesh_node('other').outputs['Geometry'], existing.inputs['Choices'])
        self.tree.links.new(existing.outputs['Bodygroup'], self.output.inputs['Bodygroups'])
        geometry = self.mesh_node('variants')
        geometry.location = (0, 0)
        self.output.location = (500, 0)

        self.tree.links.new(geometry.outputs['Geometry'], self.output.inputs['Bodygroups'])
        bodygroups = [node for node in self.tree.nodes if node.bl_idname == 'SourceIOPropBodygroupNode']
        self.assertEqual(len(bodygroups), 2)
        inserted = next(node for node in bodygroups if node != existing)
        self.assertEqual(inserted.bodygroup_name, 'variants.001')
        self.assertEqual(tuple(inserted.location), (250, 0))
        self.assertEqual(inserted.inputs['Choices'].links[0].from_node, geometry)
        self.assertTrue(any(link.from_node == inserted for link in self.output.inputs['Bodygroups'].links))
        qc = plan_model(self.output).qc_text
        self.assertIn('$bodygroup "variants.001" {\n    studio "variants.dmx"\n}', qc)
        self.assertIn('$bodygroup "variants" {\n    studio "other.dmx"\n}', qc)

    def test_material_to_skin_families_inserts_skin_family(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        self.model_entry(self.mesh_node('main', material))
        base = self.material_node(material, 'base')
        self.tree.links.new(base.outputs['Material'], self.output.inputs['Materials'])
        replacement = self.material_node(material, 'red')
        replacement.location = (0, 0)
        self.output.location = (500, 0)

        self.tree.links.new(replacement.outputs['Material'], self.output.inputs['Skin families'])
        families = [node for node in self.tree.nodes if node.bl_idname == 'SourceIOPropSkinFamilyNode']
        self.assertEqual(len(families), 1)
        inserted = families[0]
        self.assertEqual(tuple(inserted.location), (250, 0))
        self.assertEqual(inserted.inputs['Materials'].links[0].from_node, replacement)
        self.assertTrue(any(link.from_node == inserted for link in self.output.inputs['Skin families'].links))
        plan = plan_model(self.output)
        self.assertIn('$texturegroup "skinfamilies" {\n    { "red" }\n}', plan.qc_text)
        self.assertIn(self.root / 'game/materials/models/props/red.vmt', dict(plan.assets))

    def test_multiple_models_share_a_lod_level_with_explicit_sources(self):
        first = self.mesh_node('first')
        second = self.mesh_node('second')
        first_low = self.mesh_node('first_low')
        second_low = self.mesh_node('second_low')
        self.model_entry(first, 'first')
        self.model_entry(second, 'second')
        for source, replacement in ((first, first_low), (second, second_low)):
            lod = self.tree.nodes.new('SourceIOPropLodNode')
            lod.distance = 32
            self.tree.links.new(source.outputs[0], lod.inputs['Original mesh'])
            self.tree.links.new(replacement.outputs[0], lod.inputs['Replacement mesh'])
            self.tree.links.new(lod.outputs[0], self.output.inputs['LODs'])
        qc = plan_model(self.output).qc_text
        self.assertEqual(qc.count('$lod 32 {'), 1)
        self.assertIn('$lod 32 {\n    replacemodel "second.dmx" "second_low.dmx"\n'
                      '    replacemodel "first.dmx" "first_low.dmx"\n}', qc)
        self.assertIn('$sequence "idle" "second.dmx" fps 1', qc)

    def test_lods_sort_by_distance_independent_of_link_order(self):
        reference = self.mesh_node('reference')
        self.model_entry(reference)
        for distance, name, shadow in (
                (80, 'far', False), (0, 'shadow', True),
                (10, 'near', False), (30, 'middle', False)):
            replacement = self.mesh_node(name)
            lod = self.tree.nodes.new('SourceIOPropLodNode')
            lod.distance = distance
            lod.shadow = shadow
            self.tree.links.new(reference.outputs['Geometry'], lod.inputs['Original mesh'])
            self.tree.links.new(replacement.outputs['Geometry'], lod.inputs['Replacement mesh'])
            self.tree.links.new(lod.outputs['LOD'], self.output.inputs['LODs'])
        plan = plan_model(self.output)
        self.assertEqual([(lod.shadow, lod.distance) for lod in plan.model.lods],
                         [(False, 10), (False, 30), (False, 80), (True, 0)])
        qc = plan.qc_text
        self.assertLess(qc.index('$lod 10 {'), qc.index('$lod 30 {'))
        self.assertLess(qc.index('$lod 30 {'), qc.index('$lod 80 {'))
        self.assertLess(qc.index('$lod 80 {'), qc.index('$shadowlod\n{'))
        for name, directive in (('near', '$lod 10 {'), ('middle', '$lod 30 {'),
                                ('far', '$lod 80 {'), ('shadow', '$shadowlod\n{')):
            start = qc.index(directive)
            block = qc[start:qc.index('}', start)]
            self.assertIn(f'replacemodel "reference.dmx" "{name}.dmx"', block)

    def test_skinned_lod_and_joint_collision_preserve_bone_associations(self):
        rig_data = bpy.data.armatures.new('joint_test_rig')
        rig = bpy.data.objects.new('joint_test_armature', rig_data)
        bpy.context.scene.collection.objects.link(rig)
        self.addCleanup(lambda: (bpy.data.objects.remove(rig, do_unlink=True), bpy.data.armatures.remove(rig_data)))
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name in ('Pelvis', 'Spine'):
            bone = rig_data.edit_bones.new(name)
            bone.head = (0, 0, 0)
            bone.tail = (0, 0, 1)
        bpy.ops.object.mode_set(mode='OBJECT')

        reference = self.mesh_node('skinned_reference')
        replacement = self.mesh_node('skinned_shadow')
        for geometry in (reference, replacement):
            geometry.rig = rig
            pelvis = geometry.obj.vertex_groups.new(name='Pelvis')
            spine = geometry.obj.vertex_groups.new(name='Spine')
            pelvis.add([0, 1], 1.0, 'REPLACE')
            spine.add([2], 1.0, 'REPLACE')
        self.model_entry(reference)
        shadow = self.tree.nodes.new('SourceIOPropLodNode')
        shadow.shadow = True
        self.tree.links.new(reference.outputs['Geometry'], shadow.inputs['Original mesh'])
        self.tree.links.new(replacement.outputs['Geometry'], shadow.inputs['Replacement mesh'])
        self.tree.links.new(shadow.outputs['LOD'], self.output.inputs['LODs'])

        hull_mesh = bpy.data.meshes.new('pelvis_hull_mesh')
        hull_mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [],
                              [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        hull = bpy.data.objects.new('pelvis_hull', hull_mesh)
        hull.parent = rig
        hull.vertex_groups.new(name='Pelvis').add([0, 1, 2, 3], 1.0, 'REPLACE')
        collection = bpy.data.collections.new('joint_test_hulls')
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(hull)
        self.meshes.append(hull_mesh.name)
        self.objects.append(hull.name)
        self.addCleanup(lambda: bpy.data.collections.remove(collection, do_unlink=True))
        collision_geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
        collision_geometry.source_type = 'COLLECTION'
        collision_geometry.collection = collection
        collision_geometry.rig = rig
        collision_geometry.basename = 'pelvis_collision'
        collision = self.tree.nodes.new('SourceIOPropCollisionNode')
        collision.mode = 'JOINTS'
        collision.mass = 60
        collision.joint_qc = '$rootbone "Pelvis"\n$jointconstrain "Spine" x limit -30 30 0.2'
        self.tree.links.new(collision_geometry.outputs['Geometry'], collision.inputs['Geometry'])
        self.tree.links.new(collision.outputs['Collision'], self.output.inputs['Collision'])

        qc = plan_model(self.output).qc_text
        self.assertIn('$shadowlod\n{\n    replacemodel "skinned_reference.dmx" "skinned_shadow.dmx"\n}', qc)
        self.assertIn('$collisionjoints "pelvis_collision.dmx" {\n    $mass 60\n'
                      '    $rootbone "Pelvis"\n'
                      '    $jointconstrain "Spine" x limit -30 30 0.2\n}', qc)
        hull.vertex_groups.new(name='Spine').add([0], 0.25, 'REPLACE')
        with self.assertRaisesRegex(GraphBuildError, 'joint hull vertex 0'):
            plan_model(self.output)
        hull.vertex_groups['Spine'].remove([0])
        replacement.rig = None
        with self.assertRaisesRegex(GraphBuildError, 'LOD replacement rig must match'):
            plan_model(self.output)

    def test_attachment_euler_is_rendered_in_qc_degrees(self):
        self.model_entry(self.mesh_node('reference'))
        attachment = self.tree.nodes.new('SourceIOPropAttachmentNode')
        attachment.attachment_name = 'eyes'
        attachment.bone = 'Head'
        attachment.position = (1, 2, 3)
        attachment.rotation = (radians(90), 0, radians(-45))
        self.tree.links.new(attachment.outputs['Attachment'], self.output.inputs['Attachments'])
        line = next(line for line in plan_model(self.output).qc_text.splitlines()
                    if line.startswith('$attachment "eyes" "Head" 1 2 3 rotate '))
        angles = tuple(map(float, line.split(' rotate ', 1)[1].split()))
        self.assertEqual(len(angles), 3)
        for actual, expected in zip(angles, (90, 0, -45)):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_qc_names_preserve_backslashes_and_reject_quotes(self):
        self.model_entry(self.mesh_node('reference'))
        attachment = self.tree.nodes.new('SourceIOPropAttachmentNode')
        attachment.attachment_name = 'eyes\\'
        attachment.bone = r'Head\\bone'
        self.tree.links.new(attachment.outputs['Attachment'], self.output.inputs['Attachments'])
        self.assertIn('$attachment "eyes\\" "Head\\\\bone" ', plan_model(self.output).qc_text)
        attachment.attachment_name = 'bad"name'
        with self.assertRaisesRegex(GraphBuildError, 'cannot contain double quotes'):
            plan_model(self.output)
        self.assertFalse((self.root / 'game').exists())

    def test_reroute_cycle_reports_graph_error(self):
        first = self.tree.nodes.new('NodeReroute')
        second = self.tree.nodes.new('NodeReroute')
        self.tree.links.new(first.outputs[0], second.inputs[0])
        self.tree.links.new(second.outputs[0], first.inputs[0])
        with self.assertRaisesRegex(GraphBuildError, 'Cycle detected'):
            GraphResolver()(first.outputs[0])

    def test_compiler_preflight_does_not_export_invalid_configuration(self):
        self.model_entry(self.mesh_node('reference'))
        self.output.compiler_backend = 'STUDIOMDL'
        self.output.bst_engine_root = str(self.root / 'missing-tools')
        plan = plan_model(self.output)
        with self.assertRaises((CompilationError, SourceToolsError)):
            compile_plan(plan)
        self.assertFalse(plan.qc_path.exists())
        self.assertFalse(any(path.exists() for _, path in plan.geometry))

        self.output.game_root = str(self.root / 'not-a-directory')
        (self.root / 'not-a-directory').write_text('file')
        with self.assertRaisesRegex(GraphBuildError, 'not a directory'):
            plan_model(self.output)

    def test_windows_style_model_path_uses_same_qc_cache_and_model_path(self):
        self.model_entry(self.mesh_node('reference'))
        self.output.model_path = r'models\props\test.mdl'
        plan = plan_model(self.output)
        self.assertEqual(plan.model.model_path, 'models/props/test.mdl')
        self.assertEqual(plan.qc_path, self.root / 'content/models/props/test.qc')
        self.assertEqual(model_cache(self.output).path,
                         self.root / 'content/models/props/test.sourceio-cache.json')
        self.assertIn('$modelname "props/test.mdl"', plan.qc_text)

    def test_compiler_preserves_non_utf8_output_in_log(self):
        log = export_compiler.compile_direct(
            self.root / 'test.qc',
            [bpy.app.binary_path, '-b', '--factory-startup', '--python-expr',
             "import os; os.write(1, b'\\xff')"])
        self.assertIn('\ufffd', log)

    def test_compiler_command_expands_paths_as_single_arguments(self):
        qc = self.root / 'content dir' / 'model.qc'
        game = self.root / 'game dir'
        argv = direct_command(qc, game, 'wine "/opt/Source SDK/studiomdl.exe" -game {game} {qc}')
        self.assertEqual(argv, ['wine', '/opt/Source SDK/studiomdl.exe', '-game', str(game), str(qc)])
        with self.assertRaises(CompilationError):
            direct_command(qc, game, 'wine studiomdl -game {game} {missing}')

    def test_compiler_command_preserves_windows_backslashes(self):
        qc = self.root / 'content dir' / 'model.qc'
        game = self.root / 'game dir'
        with patch.object(export_compiler, 'os', SimpleNamespace(name='nt')):
            self.assertEqual(
                direct_command(qc, game, r'D:\SDK\bin\studiomdl.exe -game {game} {qc}'),
                [r'D:\SDK\bin\studiomdl.exe', '-game', str(game), str(qc)])
            self.assertEqual(
                direct_command(qc, game, r'"D:\Source SDK\studiomdl.exe" -game "{game}" "{qc}"'),
                [r'D:\Source SDK\studiomdl.exe', '-game', str(game), str(qc)])

    def _flex_rules(self):

        """Face with smile and blink.L/R keys, teeth with an open key; a flex rules collector over them."""
        if not hasattr(bpy.types.Object, 'flex_controllers'):  # registered by the add-on, not by register_nodes
            from SourceIO.blender_bindings.operators.flex_operators import SourceIO_PG_FlexController
            bpy.utils.register_class(SourceIO_PG_FlexController)
            bpy.types.Object.flex_controllers = bpy.props.CollectionProperty(type=SourceIO_PG_FlexController)

            def unregister():
                del bpy.types.Object.flex_controllers
                bpy.utils.unregister_class(SourceIO_PG_FlexController)
            self.addCleanup(unregister)
        objects = {}
        for name, keys in (('face', ('smile', 'blink.L', 'blink.R')), ('teeth', ('open',))):
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
            obj = bpy.data.objects.new(name, mesh)
            bpy.context.scene.collection.objects.link(obj)
            self.addCleanup(lambda obj=obj, mesh=mesh: (bpy.data.objects.remove(obj), bpy.data.meshes.remove(mesh)))
            obj.shape_key_add(name='Basis')
            for key in keys:
                obj.shape_key_add(name=key)
            objects[name] = obj
        nodes, links = self.tree.nodes, self.tree.links
        collector = nodes.new('SourceIOFlexRulesNode')
        smile = nodes.new('SourceIOFlexControllerNode')
        smile.control_name = 'smile'
        blink = nodes.new('SourceIOFlexControllerNode')
        blink.control_name, blink.stereo = 'blink', True
        for controller in (smile, blink):
            links.new(controller.outputs['Controller'], collector.inputs['Controllers'])
        smile_rule = nodes.new('SourceIOFlexShapeRuleNode')
        smile_rule.obj, smile_rule.shape_name = objects['face'], 'smile'
        links.new(smile.outputs['Value'], smile_rule.inputs['Value'])
        blink_rule = nodes.new('SourceIOFlexStereoRuleNode')
        blink_rule.obj, blink_rule.left_shape, blink_rule.right_shape = objects['face'], 'blink.L', 'blink.R'
        links.new(blink.outputs['Value'], blink_rule.inputs['Value'])
        half = nodes.new('SourceIOFlexConstantNode')
        half.value = 0.5
        product = nodes.new('SourceIOFlexMathNode')
        product.operation = 'MULTIPLY'
        links.new(smile.outputs['Value'], product.inputs['A'])
        links.new(half.outputs['Value'], product.inputs['B'])
        open_rule = nodes.new('SourceIOFlexShapeRuleNode')
        open_rule.obj, open_rule.shape_name = objects['teeth'], 'open'
        links.new(product.outputs['Value'], open_rule.inputs['Value'])
        for rule in (smile_rule, blink_rule, open_rule):
            links.new(rule.outputs['Rule'], collector.inputs['Rules'])
        return collector, objects['face'], objects['teeth']

    def test_flex_rule_drivers_drive_shape_keys_from_controller_sliders_and_turn_off_cleanly(self):
        collector, face, teeth = self._flex_rules()
        collector.use_drivers = True
        self.assertEqual(collector.driver_error, '')
        sliders = {slider.display_name: slider for slider in face.flex_controllers}
        self.assertEqual(set(sliders), {'smile', 'blink'})
        self.assertTrue(sliders['blink'].split)
        self.assertEqual((sliders['blink'].L, sliders['blink'].R), ('left_blink', 'right_blink'))
        self.assertEqual(len(teeth.flex_controllers), 0)

        face.data['smile'] = 0.8
        face.data['left_blink'] = 1.0
        face.data.update_tag()
        bpy.context.view_layer.update()
        keys = face.data.shape_keys.key_blocks
        self.assertAlmostEqual(keys['smile'].value, 0.8, places=5)
        self.assertAlmostEqual(keys['blink.L'].value, 1.0, places=5)
        self.assertAlmostEqual(keys['blink.R'].value, 0.0, places=5)
        # Drivers on other meshes read the slider object's controllers
        self.assertAlmostEqual(teeth.data.shape_keys.key_blocks['open'].value, 0.4, places=5)

        collector.use_drivers = False
        for obj in (face, teeth):
            animation = obj.data.shape_keys.animation_data
            self.assertEqual(len(animation.drivers) if animation else 0, 0)
        self.assertEqual(len(face.flex_controllers), 0)
        self.assertNotIn('smile', face.data)
        self.assertNotIn('sourceio_flex_preview', face)

    def test_flex_rule_drivers_refuse_missing_shape_keys_and_switch_off(self):
        collector, face, _ = self._flex_rules()
        face.shape_key_remove(face.data.shape_keys.key_blocks['smile'])
        collector.use_drivers = True
        self.assertFalse(collector.use_drivers)
        self.assertIn("no shape key 'smile'", collector.driver_error)
        self.assertEqual(len(face.flex_controllers), 0)

    def test_flex_rule_drivers_can_put_the_sliders_on_the_skeleton(self):
        collector, face, teeth = self._flex_rules()
        skeleton_data = bpy.data.armatures.new('skeleton')
        skeleton = bpy.data.objects.new('skeleton', skeleton_data)
        bpy.context.scene.collection.objects.link(skeleton)
        self.addCleanup(lambda: (bpy.data.objects.remove(skeleton), bpy.data.armatures.remove(skeleton_data)))
        collector.driver_object = skeleton
        collector.use_drivers = True
        self.assertEqual(collector.driver_error, '')
        self.assertEqual({slider.display_name for slider in skeleton.flex_controllers}, {'smile', 'blink'})
        self.assertEqual(len(face.flex_controllers), 0)
        skeleton.data['smile'] = 1.0
        skeleton.data.update_tag()
        bpy.context.view_layer.update()
        self.assertAlmostEqual(face.data.shape_keys.key_blocks['smile'].value, 1.0, places=5)
        self.assertAlmostEqual(teeth.data.shape_keys.key_blocks['open'].value, 0.5, places=5)
        collector.use_drivers = False
        self.assertNotIn('smile', skeleton.data)
        self.assertEqual(len(skeleton.flex_controllers), 0)

    def test_builtin_compiler_writes_the_model_or_refuses_before_exporting(self):
        self.model_entry(self.mesh_node('reference'))
        self.output.compiler_backend = 'BUILTIN'
        self.output.fit_weight_poses = 7
        # The hidden DMX fields don't apply: fixed versions, weights left for studiomdl to reduce
        self.output.dmx_encoding_version, self.output.dmx_model_version = '2', '1'
        self.output.dmx_trim_weights = True
        plan = plan_model(self.output)
        self.assertEqual((plan.encoding_version, plan.model_version, plan.trim_weights), (5, 18, False))
        self.assertEqual(plan.fit_weight_poses, 7)
        try:
            from SourceIO.library.utils.pylib import studiomdl  # noqa: F401
        except ImportError:  # a native build without pylib.studiomdl
            with self.assertRaisesRegex(CompilationError, 'built-in studiomdl is not available'):
                compile_plan(plan)
            self.assertFalse(plan.qc_path.exists())
            return
        log = write_compile_log(plan.model.model_path, compile_plan(plan))
        self.addCleanup(lambda: bpy.data.texts.remove(log))
        self.assertEqual(log.name, '.models/props/test.compile.log')
        self.assertIn(plan.model_file.stem, log.as_string())
        self.assertEqual(plan.model_file.read_bytes()[:4], b'IDST')
        for suffix in ('.vvd', '.dx90.vtx'):
            self.assertTrue(plan.model_file.with_name(plan.model_file.stem + suffix).is_file(), suffix)

    def test_direct_plan_uses_the_shared_pose_count(self):
        self.model_entry(self.mesh_node('reference'))
        self.output.compiler_backend = 'DIRECT'
        self.output.fit_weight_poses = 5
        self.assertEqual(plan_model(self.output).fit_weight_poses, 5)

    def test_only_one_pose_count_control_is_shown(self):
        for backend in ('BUILTIN', 'DIRECT'):
            self.output.compiler_backend = backend
            layout = MagicMock()
            self.output.draw_buttons(None, layout)
            properties = [call.args[1] for call in layout.prop.call_args_list]
            self.assertEqual(properties.count('fit_weight_poses'), 1)
            self.assertNotIn('test_weight_poses', properties)

    def test_builtin_compiler_uses_one_pose_count_for_fitting_and_testing(self):
        counts = []
        backend = SimpleNamespace(compile_qc=lambda path, **options: counts.append(options['fit_weights']))
        with patch.object(export_compiler, 'builtin_studiomdl', return_value=backend), \
                patch.object(export_compiler, '_run_studiomdl',
                             side_effect=lambda module, run, root: run(None)):
            export_compiler.compile_builtin(self.root / 'unused.qc', self.root, 5)
            export_compiler.compile_builtin(self.root / 'unused.qc', self.root, 0)
        self.assertEqual(counts, [(5, 5), None])

    def test_conflicting_shared_texture_path_fails_before_any_file_write(self):
        mat_a = bpy.data.materials.new('first')
        mat_b = bpy.data.materials.new('second')
        self.materials.extend([mat_a, mat_b])
        ref = self.mesh_node('reference', mat_a)
        self.model_entry(ref)
        for material, color in ((mat_a, (1, 0, 0, 1)), (mat_b, (0, 1, 0, 1))):
            image = bpy.data.images.new(material.name + '_img', width=4, height=4)
            image.pixels[:] = list(color) * 16
            self.images.append(image)
            source = self.tree.nodes.new('SourceIOImageSourceNode')
            source.image = image
            texture = self.tree.nodes.new('SourceIOVTFOutputNode')
            texture.vtf_name = 'models/props/shared.vtf'
            self.tree.links.new(source.outputs[0], texture.inputs[0])
            vmt = self.material_node(material, material.name)
            self.tree.links.new(texture.outputs[0], vmt.inputs['Base texture'])
            self.tree.links.new(vmt.outputs[0], self.output.inputs['Materials'])
        with self.assertRaisesRegex(GraphBuildError, 'Conflicting material recipes'):
            plan_model(self.output)
        self.assertFalse((self.root / 'game').exists())

    def test_material_export_reports_texture_auto_resize_and_clears_when_fixed(self):
        image = bpy.data.images.new('npot_source', width=47, height=47)
        self.images.append(image)
        material = bpy.data.materials.new('npot_material')
        self.materials.append(material)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        texture = self.tree.nodes.new('SourceIOVTFOutputNode')
        texture.vtf_name = 'models/props/npot.vtf'
        texture.img_format = 'RGBA8888'
        texture.generate_mipmaps = False
        self.tree.links.new(source.outputs['Texture'], texture.inputs['Texture'])
        vmt = self.material_node(material, 'npot_material')
        self.tree.links.new(texture.outputs['VTF'], vmt.inputs['Base texture'])
        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        self.output.use_export_cache = False
        reports = []

        def export(output, cache):
            export_materials(output, cache)
            return {'FINISHED'}

        action = SimpleNamespace(action='MATERIALS', _action=export,
                                 report=lambda kind, message: reports.append((kind, message)))
        self.assertEqual(_GraphAction._run(action, self.output), {'FINISHED'})
        self.assertIn('47x47 to 32x32', self.output.texture_issues)
        self.assertEqual(reports[-1], ({'WARNING'}, self.output.texture_issues))

        image.scale(32, 32)
        self.assertEqual(_GraphAction._run(action, self.output), {'FINISHED'})
        self.assertEqual(self.output.texture_issues, '')
        self.assertEqual(texture.resize_warning, '')
        self.assertEqual(len(reports), 1)

    def test_duplicate_vtf_destinations_survive_cached_material_export(self):
        image = bpy.data.images.new('duplicate_vtf_source', width=4, height=4)
        image.pixels[:] = [0.25, 0.5, 0.75, 1.0] * 16
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        for name in ('duplicate_first', 'duplicate_second'):
            material = bpy.data.materials.new(name)
            self.materials.append(material)
            texture = self.tree.nodes.new('SourceIOVTFOutputNode')
            texture.img_format = 'RGBA8888'
            texture.generate_mipmaps = False
            texture.vtf_name = 'models/props/duplicated.vtf'
            self.tree.links.new(source.outputs['Texture'], texture.inputs['Texture'])
            vmt = self.material_node(material, name)
            self.tree.links.new(texture.outputs['VTF'], vmt.inputs['Base texture'])
            self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        cache = model_cache(self.output)
        export_materials(self.output, cache)
        cache.save()
        path = self.root / 'game/materials/models/props/duplicated.vtf'
        original = path.read_bytes()
        export_materials(self.output, model_cache(self.output))
        self.assertEqual(path.read_bytes(), original)

    def test_cached_sibling_texture_still_checks_shared_path_and_names_errors(self):
        material = bpy.data.materials.new('shared')
        self.materials.append(material)
        reference = self.mesh_node('reference', material)
        other = self.tree.nodes.new('SourceIOPropModelNode')
        other.model_path = 'models/props/other.mdl'
        other.content_root = self.output.content_root
        other.game_root = self.output.game_root
        for output in (self.output, other):
            self.model_entry(reference, output=output)
        images = []
        for output, name in ((self.output, 'current'), (other, 'sibling')):
            image = bpy.data.images.new(name, width=4, height=4)
            image.pixels[:] = [0.25, 0.5, 0.75, 1.0] * 16
            images.append(image)
            self.images.append(image)
            source = self.tree.nodes.new('SourceIOImageSourceNode')
            source.image = image
            texture = self.tree.nodes.new('SourceIOVTFOutputNode')
            texture.img_format = 'RGBA8888'
            texture.generate_mipmaps = False
            texture.vtf_name = 'models/props/shared.vtf'
            self.tree.links.new(source.outputs['Texture'], texture.inputs['Texture'])
            vmt = self.material_node(material, 'shared')
            self.tree.links.new(texture.outputs['VTF'], vmt.inputs['Base texture'])
            self.tree.links.new(vmt.outputs['Material'], output.inputs['Materials'])
        cache = model_cache(other)
        export_plan(plan_model(other, check_shared=False, cache=cache), cache)
        cache.save()
        shared_path = self.root / 'game/materials/models/props/shared.vtf'
        self.assertIn(shared_path, plan_model(other, check_shared=False,
                                               cache=model_cache(other)).cached_assets)
        self.assertEqual(dict(plan_model(self.output).assets)[shared_path],
                         shared_path.read_bytes())
        images[0].pixels[:] = [1.0, 0.0, 0.0, 1.0] * 16
        with self.assertRaisesRegex(GraphBuildError, 'Conflicting material recipes'):
            plan_model(self.output)
        other.native_settings_confirmed = False
        with self.assertRaisesRegex(GraphBuildError, f'{other.name}: Review geometry'):
            plan_model(self.output)

    def test_material_stems_cannot_collide_across_directories(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        self.model_entry(self.mesh_node('reference', material))
        first = self.material_node(material, 'base')
        second = self.material_node(material, 'base')
        self.tree.links.new(first.outputs[0], self.output.inputs['Materials'])
        self.tree.links.new(second.outputs[0], self.output.inputs['Materials'])
        plan_model(self.output)  # Reusing the same destination is allowed.
        second.vmt_path = 'models/other/BASE.vmt'
        for backend in ('COMMAND', 'STUDIOMDL', 'BUILTIN', 'DIRECT'):
            with self.subTest(backend=backend):
                self.output.compiler_backend = backend
                with self.assertRaisesRegex(GraphBuildError, 'Conflicting material filename'):
                    plan_model(self.output)
        self.assertFalse((self.root / 'game').exists())

    def test_shared_material_recipe_must_match_across_model_outputs(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        reference = self.mesh_node('reference', material)
        other = self.tree.nodes.new('SourceIOPropModelNode')
        other.model_path = 'models/props/other.mdl'
        other.content_root = self.output.content_root
        other.game_root = self.output.game_root
        for output in (self.output, other):
            self.model_entry(reference, output=output)
        first = self.material_node(material, 'base')
        second = self.material_node(material, 'base')
        self.tree.links.new(first.outputs[0], self.output.inputs['Materials'])
        self.tree.links.new(second.outputs[0], other.inputs['Materials'])
        self.assertEqual(plan_model(self.output).assets, plan_model(other).assets)
        second.phong = True
        with self.assertRaisesRegex(GraphBuildError, 'Conflicting material recipes'):
            plan_model(self.output)

    def test_texture_transform_does_not_modify_shared_image_source(self):
        image = bpy.data.images.new('shared_color', width=4, height=4)
        image.pixels[:] = [0.25, 0.4, 0.7, 1.0] * 16
        initial_red = image.pixels[0]
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        invert = self.tree.nodes.new('SourceIOTextureInvertChannelsNode')
        invert.invert_r = True
        self.tree.links.new(source.outputs[0], invert.inputs[0])
        direct = self.tree.nodes.new('SourceIOVTFOutputNode')
        changed = self.tree.nodes.new('SourceIOVTFOutputNode')
        for node in (direct, changed):
            node.img_format = 'RGBA8888'
            node.generate_mipmaps = False
        self.tree.links.new(source.outputs[0], direct.inputs[0])
        self.tree.links.new(invert.outputs[0], changed.inputs[0])
        resolver = GraphResolver()
        inverted, _, _, _ = load_vtf_texture(resolver(changed.outputs[0]).artifact.data)
        original, _, _, _ = load_vtf_texture(resolver(direct.outputs[0]).artifact.data)
        self.assertGreater(inverted[0], original[0])
        self.assertEqual(image.pixels[0], initial_red)

    def test_texture_math_broadcasts_channels_and_selects_output_from_both_inputs(self):
        image = bpy.data.images.new('math_source', width=2, height=1)
        image.pixels[:] = [0.2, 0.4, 0.8, 0.6, 0.7, 0.3, 0.1, 0.25]
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        self.tree.links.new(source.outputs['Texture'], split.inputs['texture'])
        pixels = texture_pixels(GraphResolver()(source.outputs['Texture']))
        red = pixels[..., 0]

        channel_first = self.tree.nodes.new('SourceIOTextureMathNode')
        channel_first.operation = 'MUL'
        self.tree.links.new(split.outputs['R'], channel_first.inputs['A channel'])
        self.assertFalse(channel_first.inputs['B texture'].hide)
        self.tree.links.new(source.outputs['Texture'], channel_first.inputs['B texture'])
        self.assertFalse(channel_first.outputs['texture'].hide)
        self.assertTrue(channel_first.outputs['channel'].hide)
        np.testing.assert_allclose(GraphResolver()(channel_first.outputs['texture']),
                                   pixels * red[..., None], rtol=1e-6)

        image_first = self.tree.nodes.new('SourceIOTextureMathNode')
        image_first.operation = 'SUB'
        self.tree.links.new(source.outputs['Texture'], image_first.inputs['A texture'])
        self.tree.links.new(split.outputs['R'], image_first.inputs['B channel'])
        np.testing.assert_allclose(GraphResolver()(image_first.outputs['texture']), pixels - red[..., None], rtol=1e-6)

        channel_first.operation = 'INV'
        inverted = GraphResolver()(channel_first.outputs['texture'])
        np.testing.assert_allclose(inverted, np.broadcast_to(1 - red[..., None], pixels.shape), rtol=1e-6)

        both_channels = self.tree.nodes.new('SourceIOTextureMathNode')
        self.tree.links.new(split.outputs['R'], both_channels.inputs['A channel'])
        self.tree.links.new(split.outputs['G'], both_channels.inputs['B channel'])
        self.assertTrue(both_channels.outputs['texture'].hide)
        self.assertFalse(both_channels.outputs['channel'].hide)
        np.testing.assert_allclose(GraphResolver()(both_channels.outputs['channel']), red + pixels[..., 1], rtol=1e-6)
        current = np.empty(pixels.size, np.float32)
        image.pixels.foreach_get(current)
        np.testing.assert_array_equal(current.reshape(pixels.shape), pixels)

    def test_texture_math_divide_preserves_negative_divisor(self):
        node = self.tree.nodes.new('SourceIOTextureMathNode')
        node.operation = 'DIV'
        numerator = np.full((1, 4), 0.5, np.float32)
        denominator = np.array([[-0.5, 0.5, -0.0, 0.0]], np.float32)
        result = node.process({'A channel': numerator, 'B channel': denominator})['channel']
        np.testing.assert_allclose(result, [[-1.0, 1.0, -500000.0, 500000.0]])
        np.testing.assert_array_equal(denominator, [[-0.5, 0.5, -0.0, 0.0]])

    def test_texture_math_pads_rgb_with_opaque_alpha_for_rgba(self):
        rgb = np.array([[[0.2, 0.4, 0.8]]], np.float32)
        rgba = np.array([[[0.1, 0.1, 0.1, 0.5]]], np.float32)
        node = self.tree.nodes.new('SourceIOTextureMathNode')
        node.operation = 'MUL'
        for a, b in ((rgb, rgba), (rgba, rgb)):
            result = node.process({'A texture': a, 'B texture': b})
            np.testing.assert_allclose(result['texture'], [[[0.02, 0.04, 0.08, 0.5]]], rtol=1e-6)
        self.assertEqual(rgb.shape, (1, 1, 3))

    def test_rgb_split_invert_and_combine_preserve_source_and_default_opaque_alpha(self):
        rgb = np.array([[[0.2, 0.4, 0.8], [0.7, 0.3, 0.1]]], np.float32)
        unchanged = rgb.copy()
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        channels = split.process({'texture': rgb})
        np.testing.assert_array_equal(channels['A'], np.ones((1, 2), np.float32))
        channels['R'][:] = 0
        np.testing.assert_array_equal(rgb, unchanged)
        invert = self.tree.nodes.new('SourceIOTextureInvertChannelsNode')
        invert.invert_a = True
        inverted = invert.process({'texture': rgb})['texture']
        np.testing.assert_array_equal(inverted[..., :3], rgb)
        np.testing.assert_array_equal(inverted[..., 3], np.zeros((1, 2), np.float32))
        np.testing.assert_array_equal(rgb, unchanged)

        image = bpy.data.images.new('combine_rgb_source', width=2, height=1)
        image.pixels[:] = [0.2, 0.4, 0.8, 0.3, 0.7, 0.3, 0.1, 0.5]
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        self.tree.links.new(source.outputs['Texture'], split.inputs['texture'])
        combine = self.tree.nodes.new('SourceIOTextureCombineChannelsNode')
        for channel in 'RGB':
            self.tree.links.new(split.outputs[channel], combine.inputs[channel])
        result = GraphResolver()(combine.outputs['texture'])
        original = texture_pixels(GraphResolver()(source.outputs['Texture']))
        np.testing.assert_array_equal(result[..., :3], original[..., :3])
        np.testing.assert_array_equal(result[..., 3], np.ones((1, 2), np.float32))

    def test_brightness_mode_and_color_space_follow_visible_input_kind(self):
        image = bpy.data.images.new('kind_selection_source', width=2, height=1)
        image.pixels[:] = [0.2, 0.4, 0.8, 0.6, 0.7, 0.3, 0.1, 0.25]
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        self.tree.links.new(source.outputs['Texture'], split.inputs['texture'])
        pixels = texture_pixels(GraphResolver()(source.outputs['Texture']))
        contrast = self.tree.nodes.new('SourceIOTextureBrightnessContrastNode')
        contrast.input_mode = 'TEXTURE'
        contrast.brightness = 0.1
        self.tree.links.new(split.outputs['R'], contrast.inputs['channel'])
        self.assertFalse(contrast.inputs['channel'].hide)
        self.assertTrue(contrast.outputs['channel'].hide)
        self.assertIsNone(contrast.process({'channel': pixels[..., 0]}))
        contrast.input_mode = 'AUTO'
        self.assertTrue(contrast.outputs['texture'].hide)
        np.testing.assert_allclose(GraphResolver()(contrast.outputs['channel']), pixels[..., 0] + 0.1, rtol=1e-6)

        color = self.tree.nodes.new('SourceIOColorSpaceNode')
        channel_link = self.tree.links.new(split.outputs['R'], color.inputs['channel'])
        self.assertTrue(color.outputs['texture'].hide)
        self.assertFalse(color.outputs['channel'].hide)
        linear_channel = GraphResolver()(color.outputs['channel'])
        self.assertLess(linear_channel[0, 0], pixels[0, 0, 0])
        self.tree.links.remove(channel_link)
        self.tree.links.new(source.outputs['Texture'], color.inputs['texture'])
        self.assertFalse(color.outputs['texture'].hide)
        self.assertTrue(color.outputs['channel'].hide)
        linear_image = GraphResolver()(color.outputs['texture'])
        np.testing.assert_array_equal(linear_image[..., 3], pixels[..., 3])
        np.testing.assert_array_equal(texture_pixels(GraphResolver()(source.outputs['Texture'])), pixels)

    def image_source(self, name, rgba):
        """An Image Source node holding the HxWx4 float pixels ``rgba``."""
        rgba = np.asarray(rgba, np.float32)
        image = bpy.data.images.new(name, width=rgba.shape[1], height=rgba.shape[0], alpha=True, float_buffer=True)
        image.pixels.foreach_set(rgba.ravel())
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        return source

    def channel_output(self, name, values):
        """A channel output socket carrying the HxW ``values``."""
        values = np.asarray(values, np.float32)
        rgba = np.ones((*values.shape, 4), np.float32)
        rgba[..., 0] = values
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        self.tree.links.new(self.image_source(name, rgba).outputs['Texture'], split.inputs['texture'])
        return split.outputs['R']

    def test_mix_textures_uses_constant_or_channel_factor(self):
        a = self.image_source('mix_a', [[[1, 0, 0, 1]] * 3])
        b = self.image_source('mix_b', [[[0, 0, 1, 0]] * 3])
        mix = self.tree.nodes.new('SourceIOTextureMixNode')
        self.tree.links.new(a.outputs['Texture'], mix.inputs['A'])
        self.tree.links.new(b.outputs['Texture'], mix.inputs['B'])
        mix.inputs['Factor'].default_value = 0.25
        output = mix.outputs['texture']
        np.testing.assert_array_equal(GraphResolver()(output),
                                      [[[0.75, 0, 0.25, 0.75]] * 3])

        mask = self.channel_output('mix_mask', [[-0.5, 0.5, 1.5]])
        self.tree.links.new(mask, mix.inputs['Factor'])
        np.testing.assert_array_equal(GraphResolver()(output),
                                      [[[1, 0, 0, 1], [0.5, 0, 0.5, 0.5], [0, 0, 1, 0]]])

        wrong_size = self.channel_output('mix_mask_small', [[0.5]])
        self.tree.links.new(wrong_size, mix.inputs['Factor'])
        with self.assertRaisesRegex(GraphBuildError, 'Factor is 1×1, A is 3×1'):
            GraphResolver()(output)

    def test_tile_texture_bakes_repeats_and_resamples_across_tile_boundary(self):
        rgba = np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 24
        source = self.image_source('tile_source', rgba)
        tile = self.tree.nodes.new('SourceIOTextureTileNode')
        self.tree.links.new(source.outputs['Texture'], tile.inputs['texture'])
        pixels = texture_pixels(GraphResolver()(source.outputs['Texture']))
        tile.tiles_x = 2
        tile.tiles_y = 3
        tiled = GraphResolver()(tile.outputs['texture'])
        np.testing.assert_array_equal(tiled, np.tile(pixels, (3, 2, 1)))

        tile.resample = True
        tile.output_width = 3
        tile.output_height = 9
        resized = GraphResolver()(tile.outputs['texture'])
        first, second = tiled[:, 0], tiled[:, 1]
        np.testing.assert_allclose(resized, np.stack(
            ((5 * first + second) / 6, (first + second) / 2,
             (first + 5 * second) / 6), axis=1), atol=1e-7)

    def resize_node(self, width, height, filter='BILINEAR'):
        node = self.tree.nodes.new('SourceIOTextureResizeNode')
        node.inputs['Width'].default_value = width
        node.inputs['Height'].default_value = height
        node.filter = filter
        return node

    def test_resize_channel_bilinear_and_nearest_keep_width_height_and_row_order(self):
        channel = self.channel_output('resize_square', [[0.0, 0.2], [0.6, 1.0]])
        source = GraphResolver()(channel)
        resize = self.resize_node(3, 3)
        self.tree.links.new(channel, resize.inputs['channel'])
        self.assertTrue(resize.outputs['texture'].hide)
        self.assertFalse(resize.outputs['channel'].hide)
        result = GraphResolver()(resize.outputs['channel'])
        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(result[1, 1], source.mean(), places=6)
        np.testing.assert_allclose(result[::2, ::2], source, rtol=1e-6)  # corners clamp to source corners
        np.testing.assert_allclose(result[0, 1], source[0].mean(), rtol=1e-6)
        np.testing.assert_allclose(result[1, 0], source[:, 0].mean(), rtol=1e-6)

        # 2 wide × 3 high -> Width=4, Height=2
        channel = self.channel_output('resize_tall', np.arange(6).reshape(3, 2) / 10)
        source = GraphResolver()(channel)
        resize = self.resize_node(4, 2, 'NEAREST')
        self.tree.links.new(channel, resize.inputs['channel'])
        np.testing.assert_array_equal(GraphResolver()(resize.outputs['channel']), source[[0, 2]][:, [0, 0, 1, 1]])
        resize.filter = 'BILINEAR'
        rows = np.array([0.75 * source[0] + 0.25 * source[1], 0.25 * source[1] + 0.75 * source[2]])
        t = np.array([0.0, 0.25, 0.75, 1.0])
        np.testing.assert_allclose(GraphResolver()(resize.outputs['channel']),
                                   rows[:, [0]] * (1 - t) + rows[:, [1]] * t, rtol=1e-6, atol=1e-7)

    def test_resize_texture_keeps_channels_and_reports_invalid_setup(self):
        resize = self.resize_node(3, 1)
        source = self.image_source('resize_texture', np.ones((1, 1, 4)))
        self.tree.links.new(source.outputs['Texture'], resize.inputs['texture'])
        self.assertTrue(resize.outputs['channel'].hide)
        rgb = np.array([[[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]]], np.float32)
        rgba = np.array([[[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.25, 1.0]]], np.float32)
        for pixels in (rgb, rgba):
            unchanged = pixels.copy()
            result = resize.process({'texture': pixels, 'Width': 3, 'Height': 1})['texture']
            self.assertEqual(result.shape, (1, 3, pixels.shape[-1]))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result[0, 1], pixels[0].mean(axis=0), rtol=1e-6)
            np.testing.assert_array_equal(pixels, unchanged)
        np.testing.assert_allclose(resize.process({'texture': rgba, 'Width': 3, 'Height': 1})['texture'][..., 3],
                                   [[0.0, 0.5, 1.0]], rtol=1e-6)
        self.assertIs(resize.process({'texture': rgba, 'Width': 2, 'Height': 1})['texture'], rgba)

        for width, message in ((0, 'Width must be greater than 0'), (-1, 'Width must be greater than 0'),
                               (2.5, 'Width needs a positive integer, got float'),
                               (True, 'Width needs a positive integer, got bool'),
                               (rgba, 'Width needs a positive integer, got ndarray')):
            with self.assertRaisesRegex(ValueError, message):
                resize.process({'texture': rgba, 'Width': width, 'Height': 1})
        with self.assertRaisesRegex(ValueError, 'GiB limit'):
            resize.process({'texture': rgba, 'Width': 100000, 'Height': 100000})
        resize.inputs['Height'].default_value = 0
        with self.assertRaisesRegex(GraphBuildError, 'Resize .*: Height must be greater than 0'):
            GraphResolver()(resize.outputs['texture'])
        resize.inputs['Height'].default_value = 1
        self.tree.links.new(self.channel_output('resize_both', [[0.5]]), resize.inputs['channel'])
        with self.assertRaisesRegex(GraphBuildError, 'either texture or channel, not both'):
            GraphResolver()(resize.outputs['texture'])
        unlinked = self.resize_node(4, 4)
        with self.assertRaisesRegex(GraphBuildError, 'output texture is unavailable'):
            GraphResolver()(unlinked.outputs['texture'])

    def test_pbr_to_phong_size_mismatch_names_input_and_resize_fixes_it(self):
        normal = self.image_source('pbr_normal', np.full((2, 4, 4), 0.5))
        pbr = self.tree.nodes.new('SourceIOPBRToPhongNode')
        self.tree.links.new(normal.outputs['Texture'], pbr.inputs['Normal'])
        self.assertEqual(GraphResolver()(pbr.outputs['Normal']).shape, (2, 4, 4))  # unlinked scalar broadcasts
        roughness = self.channel_output('pbr_roughness', [[0.2, 0.4], [0.6, 0.8]])
        link = self.tree.links.new(roughness, pbr.inputs['Roughness'])
        expected = (f'{pbr.name}: Roughness is 2×2, Normal is 4×2. Add Resize (Texture/Channel) between Roughness '
                    f'and {pbr.name}, set Width=4 Height=2.')
        with self.assertRaises(GraphBuildError) as caught:
            GraphResolver()(pbr.outputs['Normal'])
        self.assertEqual(str(caught.exception), expected)
        self.assertIn('Roughness is 2×2', pbr.size_error)

        self.tree.links.remove(link)
        resize = self.resize_node(4, 2)
        self.tree.links.new(roughness, resize.inputs['channel'])
        self.tree.links.new(resize.outputs['channel'], pbr.inputs['Roughness'])
        self.assertEqual(GraphResolver()(pbr.outputs['Phong exponent']).shape, (2, 4, 4))
        self.assertEqual(pbr.size_error, '')

    def test_combine_math_and_blend_normals_report_size_mismatch_until_fixed(self):
        small = np.full((2, 2), 0.5, np.float32)
        wide = np.full((2, 4), 0.5, np.float32)
        combine = self.tree.nodes.new('SourceIOTextureCombineChannelsNode')
        with self.assertRaisesRegex(ValueError, r'^G is 2×2, R is 4×2\. .* between G and Combine Channels, '
                                                r'set Width=4 Height=2\.$'):
            combine.process({'R': wide, 'G': small, 'B': 0.0, 'A': 1.0})
        self.assertTrue(combine.size_error)
        self.assertEqual(combine.process({'R': wide, 'G': wide, 'B': 0.0, 'A': 1.0})['texture'].shape, (2, 4, 4))
        self.assertEqual(combine.size_error, '')

        math = self.tree.nodes.new('SourceIOTextureMathNode')
        texture = np.full((2, 4, 4), 0.5, np.float32)
        with self.assertRaisesRegex(ValueError, r'^B channel is 2×2, A texture is 4×2\. .* Width=4 Height=2\.$'):
            math.process({'A texture': texture, 'B channel': small})
        self.assertEqual(math.process({'A texture': texture, 'B channel': wide})['texture'].shape, (2, 4, 4))
        self.assertEqual(math.size_error, '')

        blend = self.tree.nodes.new('SourceIONormalBlendNode')
        with self.assertRaisesRegex(ValueError, r'^Detail is 2×2, Base is 4×2\. .* Width=4 Height=2\.$'):
            blend.process({'Base': texture, 'Detail': np.full((2, 2, 4), 0.5, np.float32)})
        self.assertEqual(blend.process({'Base': texture, 'Detail': texture})['texture'].shape, (2, 4, 4))
        self.assertEqual(blend.size_error, '')

    def test_dropped_image_and_qc_files_create_usable_export_nodes(self):
        if bpy.app.version < (4, 1, 0):
            self.skipTest('Blender file handlers require 4.1')
        from SourceIO.blender_bindings.operators.dragndrop import IMAGE_FH_vtf_import
        from SourceIO.blender_bindings.ui.export_nodes.file_drop import NODE_FH_SourceIOExportAssets

        image = bpy.data.images.new('drop_image', width=2, height=1, alpha=True)
        image.pixels[:] = [0.2, 0.4, 0.8, 1.0, 0.7, 0.3, 0.1, 0.5]
        self.images.append(image)
        png = self.root / 'drop_image.png'
        image.filepath_raw = str(png)
        image.file_format = 'PNG'
        image.save()
        qci = self.root / 'drop_rules.qci'
        qci.write_text('$surfaceprop "metal"\n')

        area = next((item for item in bpy.context.screen.areas if item.type == 'VIEW_3D'), bpy.context.screen.areas[0])
        previous_type = area.type
        area.type = 'NODE_EDITOR'
        self.addCleanup(setattr, area, 'type', previous_type)
        space = area.spaces.active
        space.tree_type = self.tree.bl_idname
        space.node_tree = self.tree
        space.pin = True
        region = next(item for item in area.regions if item.type == 'WINDOW')
        with bpy.context.temp_override(window=bpy.context.window, area=area, region=region):
            self.assertTrue(NODE_FH_SourceIOExportAssets.poll_drop(bpy.context))
            self.assertFalse(IMAGE_FH_vtf_import.poll_drop(bpy.context))
            self.assertEqual(bpy.ops.sourceio.drop_export_asset(
                directory=str(self.root),
                files=[{'name': png.name}, {'name': qci.name}],
                location=(32, 64)), {'FINISHED'})
            image_node = next(node for node in self.tree.nodes if node.bl_idname == 'SourceIOExternalTextureNode')
            qc_node = next(node for node in self.tree.nodes if node.bl_idname == 'SourceIOPropRawQcNode')
            pixels = texture_pixels(GraphResolver()(image_node.outputs['Texture']))
            self.assertEqual(pixels.shape, (1, 2, 4))
            self.assertLess(pixels[0, 0, 0], pixels[0, 1, 0])
            self.assertGreater(pixels[0, 0, 3], pixels[0, 1, 3])
            self.assertEqual(qc_node.recipe().text.strip(), '$surfaceprop "metal"')
            self.texts.append(qc_node.snippet_text)
            node_count = len(self.tree.nodes)
            with self.assertRaisesRegex(RuntimeError, 'Unsupported or missing SourceIO file'):
                bpy.ops.sourceio.drop_export_asset(filepath=str(self.root / 'unsupported.txt'))
            self.assertEqual(len(self.tree.nodes), node_count)
            vrd = self.root / 'helpers.vrd'
            vrd.write_text('')
            text_count = len(bpy.data.texts)
            with self.assertRaisesRegex(RuntimeError, 'connect rigged geometry first'):
                bpy.ops.sourceio.drop_export_asset(
                    directory=str(self.root), files=[{'name': qci.name}, {'name': vrd.name}])
            self.assertEqual(len(self.tree.nodes), node_count)
            self.assertEqual(len(bpy.data.texts), text_count)

    def test_proxy_blocks_and_raw_vmt_parameters_are_ordered(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        geometry = self.mesh_node('reference', material)
        self.model_entry(geometry)
        vmt = self.tree.nodes.new('SourceIOUnlitGenericVMTNode')
        vmt.mat = material
        vmt.vmt_path = 'models/props/base.vmt'
        raw = bpy.data.texts.new('raw VMT')
        raw.from_string('"$additive" "1"\n"$alpha" "0.5"')
        proxies = bpy.data.texts.new('proxy VMT')
        proxies.from_string('\"AnimatedTexture\" { \"animatedtexturevar\" \"$basetexture\" }')
        self.texts.extend([raw, proxies])
        vmt.raw_parameters = raw
        vmt.proxy_parameters = proxies
        self.tree.links.new(vmt.outputs[0], self.output.inputs['Materials'])
        vmt_text = dict(plan_model(self.output).assets)[self.root / 'game/materials/models/props/base.vmt'].decode()
        self.assertLess(vmt_text.index('$additive'), vmt_text.index('$alpha'))
        self.assertIn('animatedtexturevar', vmt_text)
        self.assertIn('unlitgeneric', vmt_text.lower())

    def test_vertexlit_vmt_serializes_grouped_controls_and_shared_textures(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        self.model_entry(self.mesh_node('reference', material))
        vmt = self.material_node(material, 'base')
        vmt.phong = True
        vmt.phong_boost = 2.5
        vmt.phong_fresnel_enabled = True
        vmt.phong_fresnel_ranges = (0.1, 0.5, 0.9)
        vmt.rimlight = True
        vmt.base_uv_enabled = True
        vmt.base_uv_scale = (2, 3)
        for name, color, normal in (
                ('base', (0.2, 0.4, 0.6, 1.0), False),
                ('normal', (0.5, 0.5, 1.0, 1.0), True)):
            image = bpy.data.images.new(name, width=4, height=4)
            image.pixels[:] = list(color) * 16
            self.images.append(image)
            source = self.tree.nodes.new('SourceIOImageSourceNode')
            source.image = image
            texture = self.tree.nodes.new('SourceIOVTFOutputNode')
            texture.vtf_name = f'models/props/{name}.vtf'
            texture.is_normal = normal
            self.tree.links.new(source.outputs[0], texture.inputs['Texture'])
            for slot in (('Base texture', 'Detail texture') if not normal else ('Normal texture',)):
                self.tree.links.new(texture.outputs['VTF'], vmt.inputs[slot])
        self.tree.update()
        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        assets = dict(plan_model(self.output).assets)
        self.assertEqual({path.suffix for path in assets}, {'.vmt', '.vtf'})
        self.assertEqual(sum(path.suffix == '.vtf' for path in assets), 2)
        for name, flag in (('base', TextureFlags.NORMAL), ('normal', TextureFlags.NORMAL)):
            texture = VTFFile()
            texture.load_bytes(assets[self.root / f'game/materials/models/props/{name}.vtf'])
            self.assertEqual(texture.format, ImageFormat.DXT5)
            # Linear space by default: no SRGB flag on either.
            self.assertEqual(texture.get_flag(flag), name == 'normal')
            self.assertFalse(texture.get_flag(TextureFlags.SRGB))
        text = assets[self.root / 'game/materials/models/props/base.vmt'].decode()
        for key in ('$basetexture', '$detail', '$bumpmap', '$phongboost',
                    '$phongfresnelranges', '$rimlight', '$basetexturetransform'):
            self.assertIn(key, text)
        self.assertIn('scale 2 3', text)

    def test_direct_pixel_links_insert_distinct_vtfs_with_normal_map_flags(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        self.model_entry(self.mesh_node('reference', material))
        vmt = self.material_node(material, 'base')
        image = bpy.data.images.new('normal_source', width=4, height=4)
        image.pixels[:] = [0.5, 0.5, 1.0, 1.0] * 16
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        processed = self.tree.nodes.new('SourceIOTextureInvertChannelsNode')
        processed.invert_r = True
        self.tree.links.new(source.outputs['Texture'], processed.inputs['texture'])

        self.tree.links.new(processed.outputs['texture'], vmt.inputs['Base texture'])
        self.tree.links.new(source.outputs['Texture'], vmt.inputs['Normal texture'])
        base = vmt.inputs['Base texture'].links[0].from_node
        normal = vmt.inputs['Normal texture'].links[0].from_node
        self.assertNotEqual(base, normal)
        self.assertEqual(base.bl_idname, 'SourceIOVTFOutputNode')
        self.assertEqual(normal.bl_idname, 'SourceIOVTFOutputNode')
        self.assertEqual(base.vtf_name, 'models/props/base_albedo.vtf')
        self.assertEqual(normal.vtf_name, 'models/props/base_bumpmap.vtf')
        self.assertFalse(base.is_normal)
        self.assertTrue(normal.is_normal)
        self.assertEqual(base.inputs['Texture'].links[0].from_node, processed)
        self.assertEqual(normal.inputs['Texture'].links[0].from_node, source)

        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        assets = dict(plan_model(self.output).assets)
        for name, flag in (('albedo', TextureFlags.NORMAL), ('bumpmap', TextureFlags.NORMAL)):
            vtf = VTFFile()
            vtf.load_bytes(assets[self.root / f'game/materials/models/props/base_{name}.vtf'])
            self.assertEqual(vtf.get_flag(flag), name == 'bumpmap')
            self.assertFalse(vtf.get_flag(TextureFlags.SRGB))
        refract = self.tree.nodes.new('SourceIORefractVMTNode')
        refract.vmt_path = 'models/props/glass.vmt'
        self.tree.links.new(source.outputs['Texture'], refract.inputs['Normal texture'])
        self.assertTrue(refract.inputs['Normal texture'].links[0].from_node.is_normal)

    def test_shader_specific_vmt_inputs_and_typed_raw_conflicts(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        eye = self.tree.nodes.new('SourceIOEyeRefractVMTNode')
        eye.mat = material
        eye.dilation = 0.7
        eye_text = GraphResolver()(eye.outputs['Material']).vmt_text
        self.assertIn('eyerefract', eye_text.lower())
        self.assertIn('$dilation', eye_text)
        refract = self.tree.nodes.new('SourceIORefractVMTNode')
        refract.mat = material
        refract.refract_amount = 0.2
        refract.refract_tint = (0.8, 0.7, 0.6)
        refract_text = GraphResolver()(refract.outputs['Material']).vmt_text
        for key in ('refract', '$refractamount', '$refracttint'):
            self.assertIn(key, refract_text.lower())
        unlit = self.tree.nodes.new('SourceIOUnlitGenericVMTNode')
        unlit.mat = material
        unlit.additive = True
        unlit.vertex_color = True
        unlit.alpha = 0.5
        unlit_text = GraphResolver()(unlit.outputs['Material']).vmt_text
        for key in ('unlitgeneric', '$additive', '$vertexcolor', '$alpha'):
            self.assertIn(key, unlit_text.lower())
        raw = bpy.data.texts.new('duplicate typed key')
        raw.from_string('"$additive" "0"')
        self.texts.append(raw)
        unlit.raw_parameters = raw
        with self.assertRaisesRegex(ValueError, 'conflicts with a generated parameter'):
            GraphResolver()(unlit.outputs['Material'])

    def test_vmt_parameters_override_engine_defaults_even_when_zero(self):
        material = bpy.data.materials.new('engine_defaults')
        self.materials.append(material)
        detail = self.tree.nodes.new('SourceIOVertexLitGenericVMTNode')
        detail.mat = material
        self.assertEqual(detail.detail_scale, 4.0)
        self.assertNotIn('$detailscale', GraphResolver()(detail.outputs['Material']).vmt_text)
        detail.detail_scale = 1.0
        self.assertIn('$detailscale', GraphResolver()(detail.outputs['Material']).vmt_text)

        eye = self.tree.nodes.new('SourceIOEyeRefractVMTNode')
        eye.mat = material
        self.assertEqual((eye.dilation, eye.parallax_strength, eye.eyeball_radius),
                         (0.5, 0.25, 0.5))
        defaults = GraphResolver()(eye.outputs['Material']).vmt_text
        for key in ('$dilation', '$parallaxstrength', '$eyeballradius'):
            self.assertNotIn(key, defaults)
        eye.dilation = eye.parallax_strength = eye.eyeball_radius = 0.0
        overrides = GraphResolver()(eye.outputs['Material']).vmt_text
        for key in ('$dilation', '$parallaxstrength', '$eyeballradius'):
            self.assertIn(key, overrides)

    def test_material_names_must_be_valid_vmt_references(self):
        material = bpy.data.materials.new('base')
        self.materials.append(material)
        vmt = self.material_node(material, 'base')
        for name in ('.vmt', 'models/.vmt', 'models/bad"name.vmt'):
            with self.subTest(name=name):
                vmt.vmt_path = name
                with self.assertRaises(ValueError):
                    GraphResolver()(vmt.outputs['Material'])

    def test_shared_source_is_decoded_once_for_graph_and_preview(self):
        image = bpy.data.images.new('shared_decode', width=2, height=2)
        image.pixels[:] = [0.2, 0.4, 0.6, 1.0] * 4
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        invert = self.tree.nodes.new('SourceIOTextureInvertChannelsNode')
        for node in (split, invert):
            self.tree.links.new(source.outputs['Texture'], node.inputs['texture'])
        from SourceIO.blender_bindings.ui.export_nodes import build_graph
        with patch.object(build_graph, 'texture_pixels', wraps=texture_pixels) as decode:
            resolver = GraphResolver()
            resolver(split.outputs['R'])
            resolver(invert.outputs['texture'])
            self.assertEqual(decode.call_count, 1)

            previews = [self.tree.nodes.new('SourceIOTexturePreviewNode')
                        for _ in range(2)]
            for preview in previews:
                self.tree.links.new(source.outputs['Texture'], preview.inputs['texture'])
            self.tree.nodes.active = source
            area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
            area_type, ui_type = area.type, area.ui_type
            try:
                area.type = 'NODE_EDITOR'
                area.ui_type = 'sourceio.model_definition'
                area.spaces.active.node_tree = self.tree
                with bpy.context.temp_override(area=area):
                    self.assertEqual(bpy.ops.sourceio.evaluate_texture_preview(),
                                     {'FINISHED'})
            finally:
                area.type, area.ui_type = area_type, ui_type
            self.assertEqual(decode.call_count, 2)
            self.images.extend(preview.preview_image for preview in previews)

    def test_texture_preview_accepts_channels_and_keeps_trees_independent(self):
        first = self.tree.nodes.new('SourceIOTexturePreviewNode')
        self.assertIsNone(first.process({'channel': 0.0}))
        before = {image.name for image in bpy.data.images}
        first.process({'channel': np.array([[0.2, 0.8]], dtype=np.float32)})
        first_image = next(image for image in bpy.data.images if image.name not in before)
        self.images.append(first_image)
        self.assertEqual(tuple(first_image.size), (2, 1))
        self.assertAlmostEqual(first_image.pixels[0], 0.2, places=5)
        self.assertAlmostEqual(first_image.pixels[3], 1.0, places=5)

        other_tree = bpy.data.node_groups.new('Other preview graph', 'sourceio.model_definition')
        self.addCleanup(lambda: bpy.data.node_groups.remove(other_tree))
        second = other_tree.nodes.new('SourceIOTexturePreviewNode')
        second.process({'texture': np.array([[[0.9, 0.4, 0.1]]], dtype=np.float32)})
        second_image = next(image for image in bpy.data.images if image.name not in before and image != first_image)
        self.images.append(second_image)
        self.assertAlmostEqual(second_image.pixels[0], 0.9, places=2)
        self.assertAlmostEqual(second_image.pixels[3], 1.0, places=5)
        self.assertAlmostEqual(first_image.pixels[0], 0.2, places=5)

    def test_preview_long_name_reuses_image_when_updated_or_resized(self):
        old_name = self.tree.name
        self.addCleanup(setattr, self.tree, 'name', old_name)
        self.tree.name = 'Long preview graph ' + 'x' * 230
        preview = self.tree.nodes.new('SourceIOTexturePreviewNode')
        preview.name = 'Long preview node'
        pixels = np.full((2, 2), 0.25, np.float32)
        preview.process({'channel': pixels})
        image = preview.preview_image
        self.images.append(image)
        count = len(bpy.data.images)
        preview.process({'channel': np.full((2, 2), 0.75, np.float32)})
        self.assertIs(preview.preview_image, image)
        self.assertEqual(len(bpy.data.images), count)
        self.assertAlmostEqual(image.pixels[0], 0.75, places=2)
        preview.process({'channel': np.full((3, 4), 0.5, np.float32)})
        self.assertIs(preview.preview_image, image)
        self.assertEqual(tuple(image.size), (4, 3))
        self.assertEqual(len(bpy.data.images), count)

    def test_collapsed_texture_inputs_keep_linked_sockets_visible(self):
        node = self.tree.nodes.new('SourceIOVertexLitGenericVMTNode')
        node.show_textures = False
        self.assertTrue(node.inputs['Base texture'].hide)
        self.assertTrue(node.inputs['Detail texture'].hide)
        texture = self.tree.nodes.new('SourceIOVTFOutputNode')
        link = self.tree.links.new(texture.outputs['VTF'], node.inputs['Base texture'])
        self.assertFalse(node.inputs['Base texture'].hide)
        self.assertTrue(node.inputs['Detail texture'].hide)
        node.show_textures = True
        self.assertFalse(node.inputs['Detail texture'].hide)
        self.tree.links.remove(link)
        node.show_textures = False
        self.assertTrue(node.inputs['Base texture'].hide)

    def test_existing_vtf_can_be_copied_or_processed(self):
        image = bpy.data.images.new('external_src', width=4, height=4)
        image.pixels[:] = [0.75, 0.4, 0.7, 1.0] * 8 + [0.25, 0.1, 0.3, 1.0] * 8
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        original_vtf = self.tree.nodes.new('SourceIOVTFOutputNode')
        original_vtf.img_format = 'RGBA8888'
        self.tree.links.new(source.outputs[0], original_vtf.inputs[0])
        data = GraphResolver()(original_vtf.outputs[0]).artifact.data
        vtf = VTFFile()
        vtf.load_bytes(data)
        self.assertEqual(vtf.format, ImageFormat.RGBA8888)
        external_path = self.root / 'existing.vtf'
        external_path.write_bytes(data)
        external = self.tree.nodes.new('SourceIOExternalTextureNode')
        external.source_path = str(external_path)
        decoded = texture_pixels(GraphResolver()(external.outputs['Texture']))
        self.assertAlmostEqual(decoded[0, 0, 0], 0.75, places=2)
        self.assertAlmostEqual(decoded[0, 0, 1], 0.4, places=2)
        self.assertAlmostEqual(decoded[0, 0, 2], 0.7, places=2)
        self.assertAlmostEqual(decoded[-1, 0, 0], 0.25, places=2)
        copied = self.tree.nodes.new('SourceIOVTFOutputNode')
        self.tree.links.new(external.outputs[0], copied.inputs[0])
        self.assertEqual(GraphResolver()(copied.outputs[0]).artifact.data, data)
        invert = self.tree.nodes.new('SourceIOTextureInvertChannelsNode')
        invert.invert_r = True
        processed = self.tree.nodes.new('SourceIOVTFOutputNode')
        processed.img_format = 'RGBA8888'
        self.tree.links.new(external.outputs[0], invert.inputs[0])
        self.tree.links.new(invert.outputs[0], processed.inputs[0])
        changed, _, _, _ = load_vtf_texture(GraphResolver()(processed.outputs[0]).artifact.data)
        baseline, _, _, _ = load_vtf_texture(data)
        self.assertGreater(changed[0], baseline[0])

    def test_vtf_linear_space_keeps_stored_image_values(self):
        image = bpy.data.images.new('stored_src', width=4, height=4)
        image.pixels[:] = [0.5, 0.25, 0.75, 1.0] * 16
        self.images.append(image)
        source = self.tree.nodes.new('SourceIOImageSourceNode')
        source.image = image
        node = self.tree.nodes.new('SourceIOVTFOutputNode')
        node.img_format = 'RGBA8888'
        node.generate_mipmaps = False
        self.tree.links.new(source.outputs[0], node.inputs[0])
        self.assertTrue(node.linear_space)

        def encoded():
            data = GraphResolver()(node.outputs[0]).artifact.data
            vtf = VTFFile()
            vtf.load_bytes(data)
            payload, _, _, _ = load_vtf_texture(data)
            return vtf.get_flag(TextureFlags.SRGB), tuple(payload[:3])

        # The bytes are the image's own values, as vtex writes them.
        self.assertEqual(encoded(), (False, (128, 64, 191)))
        # Off: converted from linear to sRGB and flagged.
        node.linear_space = False
        self.assertEqual(encoded(), (True, (188, 137, 224)))

    def test_split_output_cannot_overwrite_its_external_source_vtf(self):
        image = bpy.data.images.new('split_file_source', width=2, height=2)
        image.pixels[:] = [0.1, 0.4, 0.8, 1.0, 0.7, 0.2, 0.3, 1.0] * 2
        self.images.append(image)
        original_image = self.tree.nodes.new('SourceIOImageSourceNode')
        original_image.image = image
        original_vtf = self.tree.nodes.new('SourceIOVTFOutputNode')
        original_vtf.img_format = 'RGBA8888'
        original_vtf.generate_mipmaps = False
        self.tree.links.new(original_image.outputs['Texture'], original_vtf.inputs['Texture'])
        original = GraphResolver()(original_vtf.outputs['VTF']).artifact.data
        source_file = self.root / 'game/materials/models/props/source.vtf'
        source_file.parent.mkdir(parents=True)
        source_file.write_bytes(original)

        external = self.tree.nodes.new('SourceIOExternalTextureNode')
        external.source_path = str(source_file)
        split = self.tree.nodes.new('SourceIOTextureSplitChannelsNode')
        combine = self.tree.nodes.new('SourceIOTextureCombineChannelsNode')
        self.tree.links.new(external.outputs['Texture'], split.inputs['texture'])
        for output_channel, source_channel in (('R', 'B'), ('G', 'G'), ('B', 'R'), ('A', 'A')):
            self.tree.links.new(split.outputs[source_channel], combine.inputs[output_channel])
        generated = self.tree.nodes.new('SourceIOVTFOutputNode')
        generated.img_format = 'RGBA8888'
        generated.generate_mipmaps = False
        generated.vtf_name = 'models/props/source.vtf'
        self.tree.links.new(combine.outputs['texture'], generated.inputs['Texture'])
        material = bpy.data.materials.new('split_file_material')
        self.materials.append(material)
        self.model_entry(self.mesh_node('reference', material))
        vmt = self.material_node(material, 'split_file_material')
        self.tree.links.new(generated.outputs['VTF'], vmt.inputs['Base texture'])
        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        with self.assertRaisesRegex(GraphBuildError, 'would overwrite source image'):
            plan_model(self.output)
        with self.assertRaisesRegex(GraphBuildError, 'would overwrite source image'):
            export_materials(self.output)
        self.assertEqual(source_file.read_bytes(), original)
        self.assertFalse((self.root / 'content').exists())

        generated.vtf_name = 'models/props/processed.vtf'
        plan = plan_model(self.output)
        processed_file = source_file.with_name('processed.vtf')
        self.assertNotEqual(dict(plan.assets)[processed_file], original)
        self.assertEqual(source_file.read_bytes(), original)

    @staticmethod
    def _preview(name):
        """The preview Text written as ``name``; Blender 4.5 cuts the name short and keeps the full one on the Text.
        """
        found = [text for text in bpy.data.texts if name in (text.name, text.get('sourceio_preview'))]
        assert len(found) == 1, (name, found)
        return found[0]

    def test_preview_writes_vmt_text_for_material_with_different_blender_name(self):
        material = bpy.data.materials.new('Blender material')
        self.materials.append(material)
        self.model_entry(self.mesh_node('reference', material))
        vmt = self.material_node(material, 'custom_skin')
        vmt.phong = True
        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        alternate = self.material_node(material, 'alternate_skin')
        alternate.nocull = True
        self.tree.links.new(alternate.outputs['Material'], self.output.inputs['Materials'])
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        preview_name = f'SourceIO export preview - {self.output.name}'
        vmt_name = f'SourceIO VMT preview - {self.output.name} - materials/models/props/custom_skin.vmt'
        alternate_name = f'SourceIO VMT preview - {self.output.name} - materials/models/props/alternate_skin.vmt'
        try:
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.preview_export_graph(output_node=self.output.name), {'FINISHED'})
                preview = self._preview(preview_name)
                vmt_text = self._preview(vmt_name)
                alternate_text = self._preview(alternate_name)
                self.texts.extend((preview, vmt_text, alternate_text))
                self.assertIn('Asset: ', preview.as_string())
                self.assertIn('"$phong" "1"', vmt_text.as_string())
                self.assertIn('"$nocull" "1"', alternate_text.as_string())
                vmt.phong = False
                self.assertEqual(bpy.ops.sourceio.preview_export_graph(output_node=self.output.name), {'FINISHED'})
                self.assertIs(self._preview(vmt_name), vmt_text)
                self.assertNotIn('"$phong" "1"', vmt_text.as_string())
                self.assertEqual(vmt_text.as_string(),
                                 dict(plan_model(self.output).assets)[
                                     self.root / 'game/materials/models/props/custom_skin.vmt'].decode())
        finally:
            area.type = area_type
            area.ui_type = ui_type
        self.assertFalse((self.root / 'game').exists())
        self.assertFalse((self.root / 'content').exists())

    def test_node_flex_controls_export_stereo_range_and_wrinkle_data(self):
        if bpy.context.window is None:
            self.skipTest('Blender Source Tools requires a window')
        try:
            import io_scene_valvesource
        except ImportError:
            self.skipTest('Blender Source Tools is not installed')
        try:
            bpy.ops.export_scene.smd.get_rna_type()
            owns_tools = False
        except (AttributeError, RuntimeError, KeyError):
            io_scene_valvesource.register()
            owns_tools = True
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            from io_scene_valvesource import datamodel
            geometry = self.mesh_node('face')
            geometry.obj.shape_key_add(name='Basis')
            geometry.obj.shape_key_add(name='smile').data[0].co.x += 0.2
            geometry.obj.shape_key_add(name='blink').data[0].co.z += 0.2
            entry = self.model_entry(geometry)
            self.tree.nodes.active = entry
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.edit_flex_controllers(action='ADD'), {'FINISHED'})
                self.assertEqual(bpy.ops.sourceio.edit_flex_controllers(action='POPULATE'), {'FINISHED'})
                self.assertEqual(len(entry.flex_controllers), 2)
                self.assertEqual(bpy.ops.sourceio.edit_flex_controllers(action='REMOVE'), {'FINISHED'})
            self.assertEqual(len(entry.flex_controllers), 1)
            control = entry.flex_controllers[0]
            control.name = 'face_control'
            control.second_raw_name = 'blink'
            control.stereo = True
            control.eyelid = True
            control.minimum = -0.5
            control.maximum = 0.75
            control.wrinkle_scale = 0.4
            control.second_wrinkle_scale = -0.2
            self.assertIsNone(entry.flex_controller_text)
            plan = plan_model(self.output)
            self.assertIn('DMX encoding/model: 2/1', plan.manifest())
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.preview_export_graph(output_node=self.output.name), {'FINISHED'})
            preview = self._preview(f'SourceIO export preview - {self.output.name}')
            self.texts.append(preview)
            self.assertIn('DMX encoding/model: 2/1', preview.as_string())
            control.raw_name = 'missing'
            with self.assertRaisesRegex(GraphBuildError, 'Flex shape keys not found'):
                plan_model(self.output)
            self.assertFalse(plan.qc_path.exists())
            control.raw_name = 'smile'
            plan = plan_model(self.output)
            export_plan(plan)
            exported = datamodel.load(path=str(plan.geometry[0][1]), element_path=['combinationOperator'])
            flex = exported.root['combinationOperator']['controls'][0]
            self.assertEqual(tuple(flex['rawControlNames']), ('smile', 'blink'))
            self.assertTrue(flex['stereo'])
            self.assertTrue(flex['eyelid'])
            self.assertAlmostEqual(flex['flexMin'], -0.5)
            self.assertAlmostEqual(flex['flexMax'], 0.75)
            for actual, expected in zip(flex['wrinkleScales'], (0.4, -0.2)):
                self.assertAlmostEqual(actual, expected)
            self.assertFalse(any(text.name.startswith('SourceIO_stage_flex_') for text in bpy.data.texts))
        finally:
            area.type = area_type
            area.ui_type = ui_type
            if owns_tools:
                io_scene_valvesource.unregister()

    def test_generated_flex_graph_sync_preserves_authored_rules(self):
        geometry = self.mesh_node('face')
        obj = geometry.obj
        obj.shape_key_add(name='Basis')
        obj.shape_key_add(name='smile')
        obj.shape_key_add(name='blink')
        entry = self.model_entry(geometry)
        self.tree.nodes.active = entry
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.generate_flex_rules(), {'FINISHED'})
            self.assertTrue(entry.inputs['Flex Rules'].is_linked)
            group = entry.inputs['Flex Rules'].links[0].from_node
            self.assertEqual(group.bl_idname, 'SourceIOFlexGroupNode')
            inner = group.node_tree
            collector = next(node for node in inner.nodes if node.bl_idname == 'SourceIOFlexRulesNode')
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.edit_flex_group(action='ENTER'), {'FINISHED'})
            self.assertEqual(area.spaces.active.edit_tree, inner)
            inner.nodes.active = collector
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.sync_flex_rules(), {'FINISHED'})
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.edit_flex_group(action='BACK'), {'FINISHED'})
            self.assertEqual(area.spaces.active.edit_tree, self.tree)
            self.assertEqual({rule.shape_name for rule in inner.nodes if rule.bl_idname == 'SourceIOFlexShapeRuleNode'},
                             {'smile', 'blink'})
            blink = next(rule for rule in inner.nodes
                         if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and rule.shape_name == 'blink')
            blink_name = blink.name
            upstream = blink.inputs['Value'].links[0].from_socket
            constant = inner.nodes.new('SourceIOFlexConstantNode')
            constant.value = 0.5
            math = inner.nodes.new('SourceIOFlexMathNode')
            math.operation = 'MULTIPLY'
            inner.links.new(upstream, math.inputs['A'])
            inner.links.new(constant.outputs['Value'], math.inputs['B'])
            inner.links.new(math.outputs['Value'], blink.inputs['Value'])
            obj.data.shape_keys.key_blocks['blink'].name = 'wink'
            bpy.context.view_layer.update()
            blink = inner.nodes[blink_name]
            self.assertEqual(blink.shape_name, 'wink')
            self.assertEqual(blink.inputs['Value'].links[0].from_node, math)
            self.assertIn('*', next(rule.expression for rule in plan_model(self.output).model.models[0]
                                    .flex_rule_set.rules if rule.shape_name == 'wink'))
            obj.shape_key_add(name='frown')
            bpy.context.view_layer.update()
            # New keys are not given rule nodes automatically: they are listed and exported with an automatic
            # controller.
            self.assertEqual({rule.shape_name for rule in inner.nodes
                              if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and not rule.orphaned},
                             {'smile', 'wink'})
            self.assertIn(f'{obj.name}:frown', collector.missing_shapes.splitlines())
            self.assertEqual(next(rule.expression for rule in plan_model(self.output).model.models[0]
                                  .flex_rule_set.rules if rule.shape_name == 'frown'), 'frown')
            obj.shape_key_remove(obj.data.shape_keys.key_blocks['smile'])
            bpy.context.view_layer.update()
            smile = next(rule for rule in inner.nodes
                         if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and rule.shape_name == 'smile')
            self.assertTrue(smile.orphaned)
            self.assertFalse(smile.outputs['Rule'].is_linked)
            self.assertEqual({rule.shape_name for rule in plan_model(self.output).model.models[0].flex_rule_set.rules},
                             {'wink', 'frown'})
            obj.shape_key_remove(obj.data.shape_keys.key_blocks['wink'])
            obj.shape_key_add(name='squint')
            obj.shape_key_add(name='glare')
            bpy.context.view_layer.update()
            self.assertTrue(collector.sync_warning)
            self.assertEqual(inner.nodes[blink_name].inputs['Value'].links[0].from_node, math)
            # While the ambiguous rule awaits review, the unruled keys export with automatic controllers.
            self.assertEqual({rule.expression for rule in plan_model(self.output).model.models[0].flex_rule_set.rules},
                             {'frown', 'squint', 'glare'})
            inner.nodes[blink_name].shape_name = 'squint'
            with bpy.context.temp_override(area=area):
                bpy.ops.sourceio.sync_flex_rules()
                bpy.ops.sourceio.sync_flex_rules()
            self.assertFalse(collector.sync_warning)
            self.assertEqual(inner.nodes[blink_name].inputs['Value'].links[0].from_node, math)
            self.assertEqual({rule.shape_name for rule in plan_model(self.output).model.models[0].flex_rule_set.rules},
                             {'frown', 'squint', 'glare'})
        finally:
            area.type = area_type
            area.ui_type = ui_type

    def test_generated_flex_graph_tracks_geometry_source_change(self):
        first = self.mesh_node('first')
        first.obj.shape_key_add(name='Basis')
        first.obj.shape_key_add(name='smile')
        second = self.mesh_node('second')
        second.obj.shape_key_add(name='Basis')
        second.obj.shape_key_add(name='blink')
        entry = self.model_entry(first)
        self.tree.nodes.active = entry
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.generate_flex_rules(), {'FINISHED'})
            first.obj = second.obj
            bpy.context.view_layer.update()
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.sync_flex_rules(), {'FINISHED'})
            inner = entry.inputs['Flex Rules'].links[0].from_node.node_tree
            rules = [node for node in inner.nodes if node.bl_idname == 'SourceIOFlexShapeRuleNode']
            self.assertTrue(any(rule.shape_name == 'smile' and rule.orphaned for rule in rules))
            self.assertFalse(any(rule.shape_name == 'blink' for rule in rules))
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.sync_flex_rules(create=True), {'FINISHED'})
            rules = [node for node in inner.nodes if node.bl_idname == 'SourceIOFlexShapeRuleNode']
            self.assertTrue(any(rule.shape_name == 'blink' and not rule.orphaned for rule in rules))
            self.assertEqual({rule.shape_name for rule in plan_model(self.output).model.models[0].flex_rule_set.rules},
                             {'blink'})
        finally:
            area.type = area_type
            area.ui_type = ui_type

    def test_generated_flex_graph_syncs_nested_collection_members(self):
        first = self.mesh_node('first')
        first.obj.shape_key_add(name='Basis')
        first.obj.shape_key_add(name='smile')
        second = self.mesh_node('second')
        second.obj.shape_key_add(name='Basis')
        second.obj.shape_key_add(name='blink')
        root = bpy.data.collections.new('sync_root')
        child = bpy.data.collections.new('sync_child')
        self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['sync_root'], do_unlink=True))
        self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['sync_child'], do_unlink=True))
        bpy.context.scene.collection.children.link(root)
        root.children.link(child)
        child.objects.link(first.obj)
        geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
        geometry.source_type = 'COLLECTION'
        geometry.collection = root
        geometry.basename = 'faces'
        entry = self.model_entry(geometry)
        self.tree.nodes.active = entry
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                bpy.ops.sourceio.generate_flex_rules()
            child.objects.link(second.obj)
            bpy.context.view_layer.update()
            inner = entry.inputs['Flex Rules'].links[0].from_node.node_tree
            collector = next(node for node in inner.nodes if node.bl_idname == 'SourceIOFlexRulesNode')
            self.assertEqual({rule.shape_name for rule in inner.nodes
                              if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and not rule.orphaned},
                             {'smile'})
            self.assertEqual(collector.missing_shapes.splitlines(), [f'{second.obj.name}:blink'])
            child.objects.unlink(first.obj)
            bpy.context.view_layer.update()
            self.assertEqual({rule.shape_name for rule in plan_model(self.output).model.models[0].flex_rule_set.rules},
                             {'blink'})
        finally:
            area.type = area_type
            area.ui_type = ui_type

    def test_connected_flex_graph_exports_expression_rules_for_nested_collection(self):
        if bpy.context.window is None:
            self.skipTest('Blender Source Tools requires a window')
        try:
            import io_scene_valvesource
        except ImportError:
            self.skipTest('Blender Source Tools is not installed')
        try:
            bpy.ops.export_scene.smd.get_rna_type()
            owns_tools = False
        except (AttributeError, RuntimeError, KeyError):
            io_scene_valvesource.register()
            owns_tools = True
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            from io_scene_valvesource import datamodel
            mesh = self.mesh_node('face')
            mesh.obj.shape_key_add(name='Basis')
            mesh.obj.shape_key_add(name='smile').data[0].co.x += 0.2
            mesh.obj.shape_key_add(name='blink').data[0].co.z += 0.2
            source_name = mesh.obj.name
            root = bpy.data.collections.new('rule_root')
            child = bpy.data.collections.new('rule_child')
            self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['rule_root'], do_unlink=True))
            self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['rule_child'], do_unlink=True))
            bpy.context.scene.collection.children.link(root)
            root.children.link(child)
            child.objects.link(mesh.obj)
            geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
            geometry.source_type = 'COLLECTION'
            geometry.collection = root
            geometry.basename = 'face_group'
            entry = self.model_entry(geometry)
            control = entry.flex_controllers.add()
            control.name = 'faceCtrl'
            control.raw_name = 'smile'
            control.stereo = True
            control.eyelid = True
            control.minimum = -0.25
            control.maximum = 0.8
            control.wrinkle_scale = 0.4
            self.tree.nodes.active = entry
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.generate_flex_rules(), {'FINISHED'})
            self.assertFalse(entry.flex_controllers)
            inner = entry.inputs['Flex Rules'].links[0].from_node.node_tree
            smile = next(rule for rule in inner.nodes
                         if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and rule.shape_name == 'smile')
            source = smile.inputs['Value'].links[0].from_socket
            factor = inner.nodes.new('SourceIOFlexConstantNode')
            factor.value = 0.5
            math = inner.nodes.new('SourceIOFlexMathNode')
            math.operation = 'MULTIPLY'
            inner.links.new(source, math.inputs['A'])
            inner.links.new(factor.outputs['Value'], math.inputs['B'])
            inner.links.new(math.outputs['Value'], smile.inputs['Value'])
            blink = next(rule for rule in inner.nodes
                         if rule.bl_idname == 'SourceIOFlexShapeRuleNode' and rule.shape_name == 'blink')
            collector = next(node for node in inner.nodes if node.bl_idname == 'SourceIOFlexRulesNode')
            virtual = inner.nodes.new('SourceIOVirtualControllerNode')
            virtual.control_name = 'expression_driver'
            virtual.stereo = True
            virtual.minimum = -0.5
            virtual.maximum = 0.75
            inner.links.new(virtual.outputs['Controller'], collector.inputs['Controllers'])
            lower = inner.nodes.new('SourceIOFlexConstantNode')
            lower.value = 0.2
            upper = inner.nodes.new('SourceIOFlexConstantNode')
            upper.value = 0.8
            maximum = inner.nodes.new('SourceIOFlexMathNode')
            maximum.operation = 'MAX'
            minimum = inner.nodes.new('SourceIOFlexMathNode')
            minimum.operation = 'MIN'
            inner.links.new(virtual.outputs['Value'], maximum.inputs['A'])
            inner.links.new(lower.outputs['Value'], maximum.inputs['B'])
            inner.links.new(maximum.outputs['Value'], minimum.inputs['A'])
            inner.links.new(upper.outputs['Value'], minimum.inputs['B'])
            clamp = inner.nodes.new('SourceIOFlexClampNode')
            clamp.minimum = 0.3
            clamp.maximum = 0.7
            remap = inner.nodes.new('SourceIOFlexRemapNode')
            remap.input_minimum = 0.3
            remap.input_maximum = 0.7
            remap.clamp_input = True
            invert = inner.nodes.new('SourceIOFlexInvertNode')
            inner.links.new(minimum.outputs['Value'], clamp.inputs['Value'])
            inner.links.new(clamp.outputs['Value'], remap.inputs['Value'])
            inner.links.new(remap.outputs['Value'], invert.inputs['Value'])
            inner.links.new(invert.outputs['Value'], blink.inputs['Value'])
            plan = plan_model(self.output)
            export_plan(plan)
            exported = datamodel.load(path=str(plan.geometry[0][1]), element_path=['combinationOperator'])
            operator = exported.root['combinationOperator']
            controls = {control.name: control for control in operator['controls']}
            # Stereo controllers become adjacent right_/left_ controllers (studiomdl's stereo slider in flex-rule mode);
            # a mono shape they drive is exported as two generated halves.
            names = [control.name for control in operator['controls']]
            for stereo_name in ('faceCtrl', 'expression_driver'):
                self.assertEqual(names.index(f'left_{stereo_name}'), names.index(f'right_{stereo_name}') + 1)
            for side in ('left_', 'right_'):
                face = controls[side + 'faceCtrl']
                self.assertTrue(face['eyelid'])
                self.assertAlmostEqual(face['flexMin'], -0.25)
                self.assertAlmostEqual(face['flexMax'], 0.8)
                driver = controls[side + 'expression_driver']
                self.assertEqual(tuple(driver['rawControlNames']), ())
                self.assertAlmostEqual(driver['flexMin'], -0.5)
                self.assertAlmostEqual(driver['flexMax'], 0.75)
            self.assertAlmostEqual(next(rule.wrinkle_scale for rule in plan.model.models[0].flex_rule_set.rules
                                        if rule.shape_name == 'smile.L'), 0.4)
            target = next(target for target in operator['targets'] if target.type == 'DmeFlexRules')
            expressions = {rule.name: rule['expr'] for rule in target['deltaStates']}
            self.assertEqual(set(expressions), {'smile.L', 'smile.R', 'blink.L', 'blink.R'})
            self.assertIn('left_faceCtrl', expressions['smile.L'])
            self.assertIn('right_faceCtrl', expressions['smile.R'])
            self.assertIn('0.5', expressions['smile.L'])
            for driver, expected in ((-1.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
                actual = eval(expressions['blink.L'], {'__builtins__': {}, 'min': min, 'max': max},
                              {'left_expression_driver': driver})
                self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(target['target'].type, 'DmeMesh')
            self.assertAlmostEqual(bpy.data.objects[source_name].data.shape_keys.key_blocks['smile'].data[0].co.x, 0.2)
        finally:
            area.type = area_type
            area.ui_type = ui_type
            if owns_tools:
                io_scene_valvesource.unregister()

    def test_flex_sequence_eases_between_adjacent_keys_with_one_controller(self):
        mesh = self._flex_object('sequence', {'open': {0: .2}, 'wide': {1: .2}, 'closed': {2: .2}})
        self.model_entry(mesh, 'sequence')
        collector = self.tree.nodes.new('SourceIOFlexRulesNode')
        sequence = self.tree.nodes.new('SourceIOFlexSequenceRuleNode')
        sequence.obj = mesh.obj
        sequence.shapes.add()
        for item, name in zip(sequence.shapes, ('open', 'wide', 'closed')):
            item.shape_name = name
        wide = mesh.obj.data.shape_keys.key_blocks['wide']
        wide.slider_min, wide.slider_max = -1.5, 2.0
        control = self.tree.nodes.new('SourceIOVirtualControllerNode')
        control.control_name = 'transition'
        self.tree.links.new(control.outputs['Value'], sequence.inputs['Value'])
        self.tree.links.new(control.outputs['Controller'], collector.inputs['Controllers'])
        self.tree.links.new(sequence.outputs['Rule'], collector.inputs['Rules'])
        self.tree.links.new(collector.outputs['Flex Rules'], self.output.inputs['Flex Rules'])
        rules = plan_model(self.output).model.models[0].flex_rule_set
        self.assertEqual([control.name for control in rules.controls], ['transition'])
        expressions = {rule.shape_name: rule.expression for rule in rules.rules}
        self.assertEqual(set(expressions), {'open', 'wide', 'closed'})
        for position, expected in (
                (-1, (1, 0, 0)), (0, (1, 0, 0)), (0.125, (0.84375, 0.15625, 0)),
                (0.5, (0, 1, 0)), (0.875, (0, 0.15625, 0.84375)), (1, (0, 0, 1)), (2, (0, 0, 1))):
            actual = [eval(expressions[name], {'__builtins__': {}, 'min': min, 'max': max},
                           {'transition': position}) for name in ('open', 'wide', 'closed')]
            self.assertEqual(tuple(actual), expected)

        sequence.use_shape_range = True
        rules = plan_model(self.output).model.models[0].flex_rule_set
        self.assertEqual([(c.name, c.minimum, c.maximum) for c in rules.controls],
                         [('transition', 0, 1)])
        for position, expected in ((0, (1, 0, 0)), (1 / 3, (0, -1.5, 0)),
                                   (0.5, (0, 0.25, 0)), (2 / 3, (0, 2, 0)), (1, (0, 0, 1))):
            weights = dict.fromkeys(('open', 'wide', 'closed'), 0.0)
            for rule in rules.rules:
                value = eval(rule.expression, {'__builtins__': {}, 'min': min, 'max': max},
                             {'transition': position})
                self.assertTrue(0 <= value <= 1)
                weights[rule.source_shape or rule.shape_name] += value * rule.delta_scale
            np.testing.assert_allclose(tuple(weights.values()), expected)

    def test_flex_crossfade_blends_two_exported_shape_rules(self):
        if bpy.context.window is None:
            self.skipTest('Blender Source Tools requires a window')
        try:
            import io_scene_valvesource
        except ImportError:
            self.skipTest('Blender Source Tools is not installed')
        try:
            bpy.ops.export_scene.smd.get_rna_type()
            owns_tools = False
        except (AttributeError, RuntimeError, KeyError):
            io_scene_valvesource.register()
            owns_tools = True
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            from io_scene_valvesource import datamodel
            mesh = self.mesh_node('crossfade_face')
            mesh.obj.shape_key_add(name='Basis')
            mesh.obj.shape_key_add(name='smile').data[0].co.x += 0.2
            mesh.obj.shape_key_add(name='blink').data[0].co.z += 0.2
            entry = self.model_entry(mesh)
            self.tree.nodes.active = entry
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(bpy.ops.sourceio.generate_flex_rules(), {'FINISHED'})
            inner = entry.inputs['Flex Rules'].links[0].from_node.node_tree
            collector = next(node for node in inner.nodes if node.bl_idname == 'SourceIOFlexRulesNode')
            rules = {rule.shape_name: rule for rule in inner.nodes if rule.bl_idname == 'SourceIOFlexShapeRuleNode'}
            virtual = inner.nodes.new('SourceIOVirtualControllerNode')
            virtual.control_name = 'transition'
            virtual.minimum = -0.5
            virtual.maximum = 0.75
            inner.links.new(virtual.outputs['Controller'], collector.inputs['Controllers'])
            crossfade = inner.nodes.new('SourceIOFlexCrossfadeNode')
            crossfade.minimum = -0.5
            crossfade.maximum = 0.75
            inner.links.new(virtual.outputs['Value'], crossfade.inputs['Value'])
            inner.links.new(crossfade.outputs['From'], rules['smile'].inputs['Value'])
            inner.links.new(crossfade.outputs['To'], rules['blink'].inputs['Value'])
            for interpolation, quarter in (('LINEAR', 0.25), ('SMOOTH', 0.15625)):
                crossfade.interpolation = interpolation
                plan = plan_model(self.output)
                recipes = {rule.shape_name: rule.expression for rule in plan.model.models[0].flex_rule_set.rules}
                if interpolation == 'SMOOTH':
                    export_plan(plan)
                    exported = datamodel.load(path=str(plan.geometry[0][1]), element_path=['combinationOperator'])
                    operator = exported.root['combinationOperator']
                    self.assertEqual(tuple(next(control for control in operator['controls']
                                                if control.name == 'transition')['rawControlNames']), ())
                    target = next(target for target in operator['targets'] if target.type == 'DmeFlexRules')
                    expressions = {rule.name: rule['expr'] for rule in target['deltaStates']}
                else:
                    expressions = recipes
                self.assertEqual(set(expressions), {'smile', 'blink'})
                for driver, to_weight in ((-1.0, 0.0), (-0.5, 0.0), (-0.1875, quarter),
                                          (0.125, 0.5), (0.4375, 1.0 - quarter),
                                          (0.75, 1.0), (1.0, 1.0)):
                    actual = {
                        name: eval(expression, {'__builtins__': {}, 'min': min, 'max': max},
                                   {'transition': driver})
                        for name, expression in expressions.items()
                    }
                    self.assertAlmostEqual(actual['smile'], 1.0 - to_weight, places=5)
                    self.assertAlmostEqual(actual['blink'], to_weight, places=5)
                    self.assertAlmostEqual(actual['smile'] + actual['blink'], 1.0, places=5)
        finally:
            area.type = area_type
            area.ui_type = ui_type
            if owns_tools:
                io_scene_valvesource.unregister()

    def test_flex_helper_bounds_and_reversed_remap(self):
        source = self.tree.nodes.new('SourceIOFlexRawExpressionNode')
        source.expression = 'driver'
        clamp = self.tree.nodes.new('SourceIOFlexClampNode')
        self.tree.links.new(source.outputs['Value'], clamp.inputs['Value'])
        remap = self.tree.nodes.new('SourceIOFlexRemapNode')
        remap.input_minimum = -1.0
        remap.input_maximum = 1.0
        remap.output_minimum = 2.0
        remap.output_maximum = -2.0
        self.tree.links.new(source.outputs['Value'], remap.inputs['Value'])
        invert = self.tree.nodes.new('SourceIOFlexInvertNode')
        self.tree.links.new(source.outputs['Value'], invert.inputs['Value'])

        def evaluate(node, value):
            expression = GraphResolver()(node.outputs['Value'])
            return eval(expression, {'__builtins__': {}, 'min': min, 'max': max}, {'driver': value})

        self.assertEqual([evaluate(clamp, x) for x in (-2.0, 0.25, 2.0)], [0.0, 0.25, 1.0])
        clamp.minimum = 2.0
        with self.assertRaisesRegex(ValueError, 'minimum must not exceed maximum'):
            GraphResolver()(clamp.outputs['Value'])
        self.assertEqual(evaluate(invert, 0.25), 0.75)
        self.assertEqual([evaluate(remap, x) for x in (-2.0, 0.0, 2.0)], [4.0, 0.0, -4.0])
        remap.clamp_input = True
        self.assertEqual([evaluate(remap, x) for x in (-2.0, 0.0, 2.0)], [2.0, 0.0, -2.0])
        remap.clamp_input = False
        remap.clamp_output = True
        self.assertEqual([evaluate(remap, x) for x in (-2.0, 0.0, 2.0)], [2.0, 0.0, -2.0])
        remap.input_maximum = remap.input_minimum
        with self.assertRaisesRegex(ValueError, 'input minimum must be less than input maximum'):
            GraphResolver()(remap.outputs['Value'])

    def test_flex_crossfade_rejects_invalid_input_range(self):
        source = self.tree.nodes.new('SourceIOFlexRawExpressionNode')
        source.expression = 'driver'
        crossfade = self.tree.nodes.new('SourceIOFlexCrossfadeNode')
        self.tree.links.new(source.outputs['Value'], crossfade.inputs['Value'])
        for maximum in (0.0, -1.0):
            crossfade.maximum = maximum
            with self.assertRaisesRegex(ValueError, 'input minimum must be less than input maximum'):
                GraphResolver()(crossfade.outputs['To'])

    def test_advanced_flex_text_controls_export_and_validate(self):
        geometry = self.mesh_node('face')
        source_name = geometry.obj.name
        geometry.obj.shape_key_add(name='Basis')
        smile = geometry.obj.shape_key_add(name='smile')
        smile.data[0].co.x += 0.25
        smile_x = smile.data[0].co.x
        entry = self.model_entry(geometry)
        text = bpy.data.texts.new('advanced flex')
        self.texts.append(text)
        text.from_string(advanced_flex_text('smile', 0.6))
        entry.flex_controller_text = text
        plan = plan_model(self.output)
        self.assertIn(f'Flex controllers: {text.name} -> face', plan.manifest())
        self.assertNotIn('flexcontroller', plan.qc_text.lower())
        text.from_string('not DMX')
        with self.assertRaisesRegex(GraphBuildError, 'Invalid flex controller Text'):
            plan_model(self.output)
        self.assertFalse(plan.qc_path.exists())
        text.from_string(advanced_flex_text('smile', 0.6))
        self.output.geometry_format = 'SMD'
        with self.assertRaisesRegex(GraphBuildError, 'require a DMX'):
            plan_model(self.output)
        self.output.geometry_format = 'DMX'
        plan = plan_model(self.output)
        export_plan(plan)
        exported = datamodel_v2.load(filepath=plan.geometry[0][1])
        control = exported.root['combinationOperator']['controls'][0]
        self.assertEqual(control['rawControlNames'][0], 'smile')
        self.assertAlmostEqual(control['flexMax'], 0.6)
        self.assertEqual(bpy.data.objects[source_name].data.shape_keys.key_blocks['smile'].data[0].co.x, smile_x)

    def test_nested_collection_node_controls_survive_staging(self):
        if bpy.context.window is None:
            self.skipTest('Blender Source Tools requires a window')
        try:
            import io_scene_valvesource
        except ImportError:
            self.skipTest('Blender Source Tools is not installed')
        try:
            bpy.ops.export_scene.smd.get_rna_type()
            owns_tools = False
        except (AttributeError, RuntimeError, KeyError):
            io_scene_valvesource.register()
            owns_tools = True
        try:
            from io_scene_valvesource import datamodel
            mesh = self.mesh_node('face')
            mesh.obj.shape_key_add(name='Basis')
            mesh.obj.shape_key_add(name='blink').data[0].co.z += 0.2
            root = bpy.data.collections.new('node_flex_root')
            child = bpy.data.collections.new('node_flex_child')
            self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['node_flex_root'], do_unlink=True))
            self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['node_flex_child'],
                                                                 do_unlink=True))
            bpy.context.scene.collection.children.link(root)
            root.children.link(child)
            child.objects.link(mesh.obj)
            geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
            geometry.source_type = 'COLLECTION'
            geometry.collection = root
            geometry.basename = 'face_group'
            entry = self.model_entry(geometry)
            control = entry.flex_controllers.add()
            control.name = 'blink_control'
            control.raw_name = 'blink'
            control.stereo = True
            plan = plan_model(self.output)
            export_plan(plan)
            exported = datamodel.load(path=str(plan.geometry[0][1]), element_path=['combinationOperator'])
            flex = exported.root['combinationOperator']['controls'][0]
            self.assertEqual(tuple(flex['rawControlNames']), ('blink',))
            self.assertTrue(flex['stereo'])
            self.assertEqual(bpy.data.collections['node_flex_root'].vs.flex_controller_mode, 'SIMPLE')
        finally:
            if owns_tools:
                io_scene_valvesource.unregister()

    def test_nested_collection_advanced_flex_text_exports(self):
        mesh = self.mesh_node('face')
        mesh.obj.shape_key_add(name='Basis')
        mesh.obj.shape_key_add(name='blink').data[0].co.z += 0.2
        root = bpy.data.collections.new('flex_root')
        child = bpy.data.collections.new('flex_child')
        self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['flex_root'], do_unlink=True))
        self.addCleanup(lambda: bpy.data.collections.remove(bpy.data.collections['flex_child'], do_unlink=True))
        bpy.context.scene.collection.children.link(root)
        root.children.link(child)
        child.objects.link(mesh.obj)
        geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
        geometry.source_type = 'COLLECTION'
        geometry.collection = root
        geometry.basename = 'face_group'
        entry = self.model_entry(geometry)
        text = bpy.data.texts.new('advanced flex')
        self.texts.append(text)
        text.from_string(advanced_flex_text('blink', 0.7))
        entry.flex_controller_text = text
        plan = plan_model(self.output)
        export_plan(plan)
        exported = datamodel_v2.load(filepath=plan.geometry[0][1])
        self.assertAlmostEqual(exported.root['combinationOperator']['controls'][0]['flexMax'], 0.7)

    def test_native_smd_exports_bound_material_without_bst_or_window(self):
        import sys
        from unittest.mock import patch

        material = bpy.data.materials.new('Blender-only name')
        self.materials.append(material)
        geometry = self.mesh_node('reference', material)
        geometry.obj.shape_key_add(name='Basis')
        raised = geometry.obj.shape_key_add(name='Raised')
        raised.data[0].co.z = 2
        raised.value = 1
        self.model_entry(geometry)
        vmt = self.material_node(material, 'exported_skin')
        self.tree.links.new(vmt.outputs['Material'], self.output.inputs['Materials'])
        self.output.geometry_format = 'SMD'
        self.output.dmx_up_axis = 'Y'
        self.output.native_settings_confirmed = False
        with self.assertRaisesRegex(GraphBuildError, 'Native settings reviewed'):
            plan_model(self.output)
        self.output.native_settings_confirmed = True
        scene = bpy.context.scene
        source_name = geometry.obj.name
        object_names = set(bpy.data.objects)
        scene_names = set(bpy.data.scenes)
        self.assertEqual(geometry.obj.material_slots[0].material.name, material.name)
        with patch.dict(sys.modules, {'io_scene_valvesource': None}):
            plan = plan_model(self.output)
            self.assertIn('SMD up axis: Y', plan.manifest())
            export_plan(plan)
        document = smd.load(filepath=plan.geometry[0][1])
        self.assertEqual([document.materials[index] for index in document.triangles['material']], ['exported_skin'])
        self.assertEqual(document.triangles['corners']['position'][0, 0].tolist(), [0., 0., 0.])
        self.assertEqual(raised.value, 1)
        self.assertEqual(bpy.context.scene, scene)
        self.assertEqual(set(bpy.data.scenes), scene_names)
        self.assertEqual(set(bpy.data.objects), object_names)
        self.assertEqual(geometry.obj.name, source_name)
        self.assertEqual(geometry.obj.material_slots[0].material.name, material.name)
        self.assertNotIn('exported_skin', bpy.data.materials)
        self.assertTrue((self.root / 'game/materials/models/props/exported_skin.vmt').is_file())

    def test_native_smd_nested_collection_retains_object_bones_and_scene(self):
        import sys
        from unittest.mock import patch

        first = self.mesh_node('first')
        second = self.mesh_node('second')
        root = bpy.data.collections.new('nested source collection')
        child = bpy.data.collections.new('nested child collection')
        bpy.context.scene.collection.children.link(root)
        root.children.link(child)
        self.addCleanup(lambda: bpy.data.collections.remove(root, do_unlink=True))
        self.addCleanup(lambda: bpy.data.collections.remove(child, do_unlink=True))
        child.objects.link(first.obj)
        child.objects.link(second.obj)
        second.obj.location.x = 3
        geometry = self.tree.nodes.new('SourceIOPropGeometryNode')
        geometry.source_type = 'COLLECTION'
        geometry.collection = root
        geometry.basename = 'nested'
        self.output.geometry_format = 'SMD'
        self.model_entry(geometry)
        names = {first.obj.name, second.obj.name}
        scene = bpy.context.scene
        object_names = set(bpy.data.objects)
        scene_names = set(bpy.data.scenes)
        with patch.dict(sys.modules, {'io_scene_valvesource': None}):
            plan = plan_model(self.output)
            export_plan(plan)
        document = smd.load(filepath=plan.geometry[0][1])
        bones = {node.name: node.id for node in document.nodes}
        self.assertTrue(names <= bones.keys())
        self.assertNotEqual(bones[first.obj.name], bones[second.obj.name])
        self.assertEqual(set(document.triangles['corners']['bone'][:, 0].tolist()),
                         {bones[first.obj.name], bones[second.obj.name]})
        self.assertEqual(bpy.context.scene, scene)
        self.assertEqual(set(bpy.data.scenes), scene_names)
        self.assertEqual(set(bpy.data.objects), object_names)
        self.assertEqual({first.obj.name, second.obj.name}, names)

    def test_native_smd_failure_preserves_previous_geometry_file(self):
        geometry = self.mesh_node('reference')
        self.model_entry(geometry)
        self.output.geometry_format = 'SMD'
        plan = plan_model(self.output)
        target = plan.geometry[0][1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'previous complete SMD')
        geometry.obj.data.clear_geometry()
        with self.assertRaisesRegex(GraphBuildError, 'Native SMD.*no exportable faces'):
            export_plan(plan)
        self.assertEqual(target.read_bytes(), b'previous complete SMD')
        self.assertFalse(plan.qc_path.exists())

    def _biped(self):
        rig_data = bpy.data.armatures.new('sfm_rig_test_rig')
        rig = bpy.data.objects.new('sfm_rig_test_armature', rig_data)
        bpy.context.scene.collection.objects.link(rig)
        self.addCleanup(lambda: (bpy.data.objects.remove(rig, do_unlink=True), bpy.data.armatures.remove(rig_data)))
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        layout = [('Hips', None), ('Spine', 'Hips'), ('Chest', 'Spine'),
                  ('Neck', 'Chest'), ('Head', 'Neck'),
                  ('Tail0', 'Hips'), ('Tail1', 'Tail0')]
        for side in 'LR':
            layout += [(f'Shoulder.{side}', 'Chest'), (f'UpperArm.{side}', f'Shoulder.{side}'),
                       (f'Forearm.{side}', f'UpperArm.{side}'), (f'Hand.{side}', f'Forearm.{side}'),
                       (f'Thigh.{side}', 'Hips'), (f'Shin.{side}', f'Thigh.{side}'),
                       (f'Foot.{side}', f'Shin.{side}'), (f'Toe.{side}', f'Foot.{side}')]
        for index, (name, parent) in enumerate(layout):
            bone = rig_data.edit_bones.new(name)
            bone.head = (index, 0, 0)
            bone.tail = (index, 0, 1)
            if parent:
                bone.parent = rig_data.edit_bones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')
        geometry = self.mesh_node('biped')
        geometry.rig = rig
        geometry.obj.vertex_groups.new(name='Hips').add([0, 1, 2], 1.0, 'REPLACE')
        self.model_entry(geometry)
        return rig

    def test_sfm_ik_rig_script_is_exported_with_renamed_bones(self):
        rig = self._biped()
        node = self.tree.nodes.new('SourceIOSfmIkRigNode')
        self.tree.links.new(node.outputs['SFM rig'], self.output.inputs['SFM rig'])
        self.assertIs(sfm_rig_nodes._connected_armature(self.tree, node), rig)
        created, found, missing = sfm_rig_nodes.detect_rig_bones(self.tree, node, rig)
        self.assertEqual((len(created), found, missing), (4, 4, []))
        self.assertEqual((node.pelvis, node.chest, node.head), ('Hips', 'Chest', 'Head'))
        chains = {(item.kind, item.chain_name): item for item in created}
        arm, leg = chains['ARM', 'L'], chains['LEG', 'R']
        self.assertEqual(
            (arm.label, arm.collar_bone, arm.start_bone, arm.end_bone),
            ('Left arm', 'Shoulder.L', 'UpperArm.L', 'Hand.L'))
        self.assertEqual((leg.start_bone, leg.end_bone, leg.toe_bone), ('Thigh.R', 'Foot.R', 'Toe.R'))
        # Detecting again adds nothing.
        self.assertEqual(sfm_rig_nodes.detect_rig_bones(self.tree, node, rig), ([], 4, []))
        chain = self.tree.nodes.new('SourceIOSfmIkChainNode')
        chain.chain_name = 'spare'
        chain.start_bone, chain.end_bone = 'Shin.L', 'Toe.L'
        self.tree.links.new(chain.outputs['IK chain'], node.inputs['IK chains'])
        with self.assertRaisesRegex(GraphBuildError, 'more than one control'):
            plan_model(self.output)
        self.tree.links.remove(chain.outputs['IK chain'].links[0])

        table = bpy.data.texts.new('sfm_rig_renames')
        self.texts.append(table)
        table.write('Hand.L = hand_left\n')
        rename = self.tree.nodes.new('SourceIOBoneRenameNode')
        rename.table = table
        self.tree.links.new(rename.outputs['Bone names'], self.output.inputs['Bone names'])

        plan = plan_model(self.output)
        script = self.root / 'game' / 'scripts' / 'sfm' / 'animset' / 'rig_ik_props__test.py'
        self.assertIn(f'Asset: {script}', plan.manifest())
        self.assertNotIn('rig_ik', plan.qc_text)
        export_plan(plan)
        text = script.read_text(encoding='utf-8')
        compile(text, str(script), 'exec')
        self.assertIn(
            '("arm", "L", "rig_hand_L", "rig_elbow_L", "UpperArm.L", '
            '"Forearm.L", "hand_left", "", "rig_collar_L", "LeftArm", "")',
            text)
        self.assertIn('("rig_neck", "Neck", "rig_chest", "RigBody")', text)

        node.script_name = 'rig_custom'
        node.head = 'Missing'
        with self.assertRaisesRegex(GraphBuildError, "head bone 'Missing' is not in the exported armature"):
            plan_model(self.output)
        node.head = ''
        arm.end_bone = ''
        with self.assertRaisesRegex(GraphBuildError, 'end bone is required'):
            plan_model(self.output)
        self.tree.nodes.remove(arm)
        plan = plan_model(self.output)
        self.assertTrue(any(path.name == 'rig_custom.py' for path, _ in plan.assets))

    def test_sfm_rig_generic_part_nodes_reach_the_script(self):
        self._biped()
        nodes, links = self.tree.nodes, self.tree.links
        rig = nodes.new('SourceIOSfmIkRigNode')
        rig.pelvis = 'Hips'
        links.new(rig.outputs['SFM rig'], self.output.inputs['SFM rig'])

        tail = nodes.new('SourceIOSfmFkChainNode')
        tail.chain_name = 'tail'
        tail.start_bone, tail.end_bone = 'Tail0', 'Tail1'
        links.new(tail.outputs['FK chain'], rig.inputs['FK chains'])

        leg = nodes.new('SourceIOSfmIkChainNode')
        leg.kind, leg.chain_name = 'LEG', 'L'
        leg.start_bone, leg.end_bone = 'Thigh.L', 'Foot.L'
        leg.toe_bone = 'Toe.L'
        # Hidden for legs, so it must not leak into the recipe.
        leg.collar_bone = 'Hips'
        links.new(leg.outputs['IK chain'], rig.inputs['IK chains'])

        aim = nodes.new('SourceIOSfmConstraintNode')
        aim.kind, aim.slave = 'AIM', 'Head'
        aim.targets = 'rig_tail_1=0.25, Chest'
        links.new(aim.outputs['Constraint'], rig.inputs['Constraints'])

        wag = nodes.new('SourceIOSfmConstraintNode')
        wag.kind, wag.slave = 'ROTATION', 'rig_tail_0'
        wag.slider_name, wag.axis = 'rig_wag', 'Z'
        # Ignored by rotation sliders.
        wag.targets = 'Chest'
        links.new(wag.outputs['Constraint'], rig.inputs['Constraints'])

        plan = plan_model(self.output)
        text = next(data for path, data in plan.assets if path.suffix == '.py').decode('utf-8')
        compile(text, 'rig.py', 'exec')
        self.assertIn('("rig_tail_1", "Tail1", "rig_tail_0", "RigExtra")', text)
        self.assertIn(
            '("leg", "L", "rig_foot_L", "rig_knee_L", "Thigh.L", "Shin.L", '
            '"Foot.L", "rig_toe_L", "rig_foot_L", "LeftLeg", "pvt_heel_L")',
            text)
        self.assertIn(
            '("aim", ("bone", "Head"), ((("handle", "rig_tail_1"), 0.25), '
            '(("bone", "Chest"), 1.0)), True, ', text)
        self.assertIn(
            '("rotation", ("handle", "rig_tail_0"), (), True, "rig_wag", '
            '(0.0, 0.0, 1.0), 90.0, 0.5, "RigExtra")', text)

        aim.targets = 'Chest=heavy'
        with self.assertRaisesRegex(GraphBuildError, "'heavy' is not a number"):
            plan_model(self.output)
        aim.targets = 'rig_missing'
        with self.assertRaisesRegex(GraphBuildError, "'rig_missing' is not in"):
            plan_model(self.output)

    def _detected_rig_node(self, rig):
        """A rig node with the detected limbs as chain nodes by kind and side; the right leg is renamed and has its own
        heel attachment.
        """
        node = self.tree.nodes.new('SourceIOSfmIkRigNode')
        self.tree.links.new(node.outputs['SFM rig'], self.output.inputs['SFM rig'])
        created, _, _ = sfm_rig_nodes.detect_rig_bones(self.tree, node, rig)
        chains = {(item.kind, item.chain_name): item for item in created}
        leg = chains['LEG', 'R']
        leg.chain_name, leg.heel_attachment = 'R2', 'heel_x'
        return node, chains

    def _pole_offsets(self, rig):
        """Offset of every authored pole handle from its joint, found the way the script does: from its weights on a
        triangle of bones.
        """
        plan = plan_model(self.output)
        text = next(data for path, data in plan.assets if path.suffix == '.py').decode('utf-8')
        tables = {}
        exec(text.split('\nROOT_BONE')[0].replace('import vs', ''), tables)
        middles = {limb[3]: limb[5] for limb in tables['LIMBS']}
        result = {}
        for handle, anchor, (first, second, lift) in tables['POLE_ANCHORS']:
            world, bones = rig.matrix_world, rig.data.bones
            origin, *others = (world @ bones[name].head_local for name in anchor)
            u, v = others[0] - origin, others[1] - origin
            normal = u.cross(v)
            target = (origin + first * u + second * v + lift * normal / normal.length ** 0.5)
            middle = rig.data.bones[middles[handle]].head_local
            result[handle] = target - rig.matrix_world @ middle
        return result

    def test_sfm_rig_pole_targets_are_created_and_exported(self):
        rig = self._biped()
        # Off the line the other test bones are on, to span a triangle.
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        rig.data.edit_bones['Head'].head = (4, 0, 3)
        rig.data.edit_bones['Head'].tail = (4, 0, 4)
        bpy.ops.object.mode_set(mode='OBJECT')
        node, chains = self._detected_rig_node(rig)
        self.output.model_scale = 2.0
        self.assertEqual(self._pole_offsets(rig), {})

        made, _, skipped = sfm_rig_nodes.create_pole_targets(node, rig, bpy.context.scene)
        empties = [chain.pole_target for chain in chains.values()]
        self.objects.extend(empty.name for empty in empties)
        self.assertEqual((made, skipped), (4, []))
        self.assertEqual(chains['LEG', 'R'].pole_target.name, 'sfm_rig_test_armature_rig_knee_R2')
        knee = chains['LEG', 'L'].pole_target
        self.assertEqual((knee.parent, knee.parent_bone), (rig, 'Shin.L'))
        bpy.context.view_layer.update()

        # The test limbs are straight lines along X, so the poles go sideways by 40% of the limb length (2); the model
        # scale does not matter.
        offsets = self._pole_offsets(rig)
        self.assertEqual(set(offsets), {'rig_elbow_L', 'rig_elbow_R', 'rig_knee_L', 'rig_knee_R2'})
        for offset in offsets.values():
            for value, wanted in zip(offset, (0.0, 0.8, 0.0)):
                self.assertAlmostEqual(value, wanted, places=5)

        # The markers follow the pose but export their rest position.
        rig.pose.bones['Shin.L'].rotation_quaternion = (0.7071, 0.7071, 0, 0)
        rig.pose.bones['Thigh.R'].location = (0.0, 0.0, 3.0)
        bpy.context.view_layer.update()
        self.assertGreater((knee.matrix_world.translation - Vector((12, 0.8, 0))).length, 0.5)
        for offset in self._pole_offsets(rig).values():
            for value, wanted in zip(offset, (0.0, 0.8, 0.0)):
                self.assertAlmostEqual(value, wanted, places=4)

        # A moved marker changes the handle; a cleared one returns to auto.
        elbow = chains['ARM', 'L'].pole_target
        elbow.matrix_basis = Matrix.Translation((0, 0, 1)) @ elbow.matrix_basis
        chains['LEG', 'L'].pole_target = None
        bpy.context.view_layer.update()
        offsets = self._pole_offsets(rig)
        self.assertNotIn('rig_knee_L', offsets)
        self.assertGreater((offsets['rig_elbow_L'] - offsets['rig_elbow_R']).length, 0.9)

        # Existing markers are kept; only the cleared one is made again.
        chains['ARM', 'R'].end_bone = 'Head'
        self.assertEqual(
            sfm_rig_nodes.create_pole_targets(node, rig, bpy.context.scene),
            (1, 2, ["Right arm: 'Head' must be the grandchild of "
                    "'UpperArm.R'"]))
        self.objects.append(chains['LEG', 'L'].pole_target.name)
        self.assertIs(chains['ARM', 'L'].pole_target, elbow)

    def test_sfm_rig_heel_attachments_are_created_once(self):
        rig = self._biped()
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for side in 'LR':
            toe = rig.data.edit_bones[f'Toe.{side}']
            toe.head.z = -0.5
            toe.tail = (toe.head.x, 1.0, -0.5)
        bpy.ops.object.mode_set(mode='OBJECT')
        node, chains = self._detected_rig_node(rig)
        leg = chains['LEG', 'R']

        scene = bpy.context.scene
        self.assertEqual(sfm_rig_nodes.create_heel_attachments(self.tree, node, rig, scene), (2, 0, []))
        heels = {link.from_node.attachment_name: link.from_node for link in self.output.inputs['Attachments'].links}
        self.objects.extend(heel.empty.name for heel in heels.values())
        # The chain node names its own heel attachment.
        self.assertEqual(set(heels), {'pvt_heel_L', 'heel_x'})
        self.assertEqual(heels['heel_x'].bone, 'Foot.R')
        # Each has an arrows empty on the foot bone.
        empty = heels['pvt_heel_L'].empty
        self.assertEqual((empty.empty_display_type, empty.parent_bone), ('ARROWS', 'Foot.L'))
        bpy.context.view_layer.update()
        # Half a unit below the ankle; the foot bone points up, so that is its negative Y axis.
        qc = plan_model(self.output).qc_text
        line = next(line for line in qc.split('\n') if line.startswith('$attachment "pvt_heel_L" "Foot.L"'))
        for value, wanted in zip(map(float, line.split('"')[4].split()[:3]), (0, -0.5, 0)):
            self.assertAlmostEqual(value, wanted, places=4)
        self.assertIn('"rig_toe_R2", "rig_foot_R2", "Leg_R2", "heel_x")', self._script())

        self.assertEqual(sfm_rig_nodes.create_heel_attachments(self.tree, node, rig, scene), (0, 2, []))
        self.assertEqual(len(self.output.inputs['Attachments'].links), 2)

        # Legs that cannot be used are named with the reason; arms are not the heel button's business.
        leg.chain_name = ''
        chains['LEG', 'L'].toe_bone = ''
        chains['ARM', 'R'].end_bone = 'Head'
        self.assertEqual(sfm_rig_nodes.create_heel_attachments(
            self.tree, node, rig, scene), (0, 0, [
                'Right leg: IK chain name may only contain letters, '
                'digits and underscores',
                'leg L: needs a toe bone']))

    def test_sfm_rig_side_fields_of_saved_graphs_become_chain_nodes(self):
        if bpy.app.version < (5, 0, 0):
            self.skipTest('Saved graphs are only converted on Blender 5')
        rig = self._biped()
        marker = bpy.data.objects.new('saved_knee_marker', None)
        self.objects.append(marker.name)
        node = self.tree.nodes.new('SourceIOSfmIkRigNode')
        node.pelvis = 'Hips'
        self.tree.links.new(node.outputs['SFM rig'], self.output.inputs['SFM rig'])
        # A chain the user had already made for the left arm.
        own = self.tree.nodes.new('SourceIOSfmIkChainNode')
        own.kind, own.chain_name = 'ARM', 'mine'
        own.start_bone, own.end_bone = 'UpperArm.L', 'Hand.L'
        self.tree.links.new(own.outputs['IK chain'], node.inputs['IK chains'])
        # What a file saved with the old arm and leg fields still holds.
        stored = node.bl_system_properties_get(do_create=True)
        for name, value in dict(
                upper_arm_l='UpperArm.L', hand_l='Hand.L',
                collar_l='Shoulder.L', upper_arm_r='UpperArm.R',
                hand_r='Hand.R', collar_r='Shoulder.R',
                thigh_l='Thigh.L', foot_l='Foot.L', toe_l='Toe.L',
                thigh_r='Thigh.R', foot_r='', toe_r='Toe.R').items():
            stored[name] = value
        stored['knee_pole_l'] = marker

        self.tree.update()
        chains = {(item.kind, item.chain_name): item for item in sfm_rig_nodes._chain_nodes(node)}
        # The right leg had no foot and the left arm has its own chain.
        self.assertEqual(set(chains), {('ARM', 'mine'), ('ARM', 'R'), ('LEG', 'L')})
        self.assertEqual(chains['ARM', 'R'].collar_bone, 'Shoulder.R')
        self.assertEqual(chains['LEG', 'L'].toe_bone, 'Toe.L')
        self.assertIs(chains['LEG', 'L'].pole_target, marker)
        # The test bones are on one line, which a pole target cannot use.
        chains['LEG', 'L'].pole_target = None
        script = self._script()
        self.assertIn('"rig_hand_R", "rig_elbow_R"', script)
        self.assertIn('"rig_foot_L", "rig_knee_L"', script)

        # Converted once: a removed chain does not come back.
        self.tree.nodes.remove(chains['ARM', 'R'])
        self.tree.update()
        self.assertEqual(len(sfm_rig_nodes._chain_nodes(node)), 2)

    def test_sfm_finger_node_reaches_the_script(self):
        self._biped()
        rig = self.tree.nodes.new('SourceIOSfmIkRigNode')
        rig.pelvis = 'Hips'
        self.tree.links.new(rig.outputs['SFM rig'], self.output.inputs['SFM rig'])
        finger = self.tree.nodes.new('SourceIOSfmFingerNode')
        finger.finger_name = 'claw'
        finger.start_bone, finger.end_bone = 'Tail0', 'Tail1'
        finger.curl_axis, finger.curl_angle = '-Z', 60.0
        finger.curl_default = 0.0
        constraints = rig.inputs['Constraints']
        self.tree.links.new(finger.outputs['Constraint'], constraints)
        # Next to an ordinary constraint on the same input.
        aim = self.tree.nodes.new('SourceIOSfmConstraintNode')
        aim.kind, aim.slave, aim.targets = 'AIM', 'Head', 'Chest'
        self.tree.links.new(aim.outputs['Constraint'], constraints)
        script = self._script()
        self.assertIn(
            '("rig_curl_claw", "RigExtra", 60.0, 0.0, (0.0, 0.0, -1.0), '
            '("rig_claw_0", "rig_claw_1")),', script)
        self.assertIn('("aim", ("bone", "Head"), ((("bone", "Chest"), 1.0),)', script)
        finger.finger_name = 'bad name'
        with self.assertRaisesRegex(GraphBuildError, 'Finger name may only'):
            plan_model(self.output)
        finger.finger_name = 'claw'

        # As saved while fingers had their own input and socket type; built with the migration off, since every socket
        # change updates the tree.
        from unittest.mock import patch
        with patch.object(sfm_rig_nodes, 'migrate_rig_nodes'):
            self.tree.links.remove(finger.outputs[0].links[0])
            finger.outputs.remove(finger.outputs[0])
            old = finger.outputs.new('SourceIOSfmFkChainSocket', 'Finger')
            fingers = rig.inputs.new('SourceIOSfmFkChainSocket', 'Fingers', use_multi_input=True)
            self.tree.links.new(old, fingers)
        self.tree.update()
        self.assertEqual([socket.name for socket in rig.inputs], ['IK chains', 'FK chains', 'Constraints'])
        self.assertEqual(finger.outputs[0].bl_idname, 'SourceIOSfmConstraintSocket')
        self.assertIn('"rig_curl_claw"', self._script())

    def _script(self):
        return next(data for path, data in plan_model(self.output).assets if path.suffix == '.py').decode('utf-8')

    def _attachment(self, name):
        """Bone, position and angles of one $attachment line of the QC."""
        qc = plan_model(self.output).qc_text
        line = next(line for line in qc.split('\n') if line.startswith(f'$attachment "{name}"'))
        bone, numbers = line.split('"')[3], line.split('"')[4].split()
        return bone, [float(value) for value in numbers if value != 'rotate']

    def assert_attachment(self, name, bone, expected):
        found_bone, values = self._attachment(name)
        self.assertEqual(found_bone, bone)
        for value, wanted in zip(values, expected, strict=True):
            self.assertAlmostEqual(value, wanted, places=3)

    def test_attachment_empty_replaces_position_and_rotation(self):
        rig = self._biped()
        node = self.tree.nodes.new('SourceIOPropAttachmentNode')
        node.attachment_name, node.bone = 'eyes', 'Head'
        node.position = (1.0, 2.0, 3.0)
        node.rotation = (radians(10), radians(20), radians(30))
        self.tree.links.new(node.outputs['Attachment'], self.output.inputs['Attachments'])
        typed = (1, 2, 3, 10, 20, 30)
        self.assert_attachment('eyes', 'Head', typed)

        # The empty starts where the typed values put the attachment.
        empty = prop_nodes.spawn_attachment_empty(self.tree, node, bpy.context.scene)
        self.objects.append(empty.name)
        self.assertIs(node.empty, empty)
        self.assertIs(node.armature, rig)
        self.assertEqual((empty.parent, empty.parent_bone), (rig, 'Head'))
        bpy.context.view_layer.update()
        # The Head bone starts at x=4 and points up, so its Y is world Z.
        for value, wanted in zip(empty.matrix_world.translation, (5, -3, 2)):
            self.assertAlmostEqual(value, wanted, places=4)
        self.assert_attachment('eyes', 'Head', typed)

        # Posing the bone moves the empty but not the exported attachment.
        rig.pose.bones['Head'].rotation_quaternion = (0.7071, 0, 0.7071, 0)
        rig.pose.bones['Neck'].location = (0.0, 2.0, 0.0)
        bpy.context.view_layer.update()
        self.assertGreater((empty.matrix_world.translation - Vector((5, -3, 2))).length, 1.0)
        self.assert_attachment('eyes', 'Head', typed)

        # Moving the empty does.
        empty.matrix_basis = Matrix.Translation((0, 0, 1)) @ empty.matrix_basis
        bpy.context.view_layer.update()
        self.assert_attachment('eyes', 'Head', (1, 2, 4, 10, 20, 30))
        # Another bone of the same armature gives the offset from that bone.
        node.bone = 'Neck'
        self.assert_attachment('eyes', 'Neck', (2, 2, 4, 10, 20, 30))

        node.bone = 'Nope'
        with self.assertRaisesRegex(GraphBuildError, "bone 'Nope' is not in"):
            plan_model(self.output)

    def test_handmade_attachment_uses_connected_armature(self):
        rig = self._biped()
        rig.location = (10, 20, 30)
        bpy.context.view_layer.update()
        node = self.tree.nodes.new('SourceIOPropAttachmentNode')
        node.attachment_name, node.bone = 'eyes', 'Head'
        self.tree.links.new(node.outputs['Attachment'], self.output.inputs['Attachments'])
        empty = bpy.data.objects.new('handmade_attachment', None)
        bpy.context.scene.collection.objects.link(empty)
        self.objects.append(empty.name)
        empty.matrix_world = (rig.matrix_world @ rig.data.bones['Head'].matrix_local
                              @ Matrix.Translation((1, 2, 3)))
        node.empty = empty
        bpy.context.view_layer.update()
        self.assertIsNone(node.armature)
        self.assert_attachment('eyes', 'Head', (1, 2, 3, 0, 0, 0))

    def test_attachment_empty_without_a_bone_uses_world_space(self):
        self.model_entry(self.mesh_node('static_reference'))
        node = self.tree.nodes.new('SourceIOPropAttachmentNode')
        node.attachment_name, node.bone = 'muzzle', 'static_prop'
        node.position = (5.0, 6.0, 7.0)
        self.tree.links.new(node.outputs['Attachment'], self.output.inputs['Attachments'])
        empty = prop_nodes.spawn_attachment_empty(self.tree, node, bpy.context.scene)
        self.objects.append(empty.name)
        self.assertIsNone(empty.parent)
        node.armature = None
        bpy.context.view_layer.update()
        self.assert_attachment('muzzle', 'static_prop', (5, 6, 7, 0, 0, 0))

    def test_saved_output_without_sfm_rig_socket_gets_it_on_update(self):
        self.output.inputs.remove(self.output.inputs['SFM rig'])
        self.model_entry(self.mesh_node('legacy'))
        self.assertIsNone(plan_model(self.output).model.sfm_rig)
        self.tree.update()
        self.assertEqual(self.output.inputs['SFM rig'].bl_idname, 'SourceIOSfmRigSocket')

    def test_saved_output_without_procedural_bones_socket_gets_it_on_update(self):
        self.output.inputs.remove(self.output.inputs['Procedural bones'])
        self.model_entry(self.mesh_node('legacy'))
        self.assertEqual(plan_model(self.output).model.procedural_bones, ())
        self.tree.update()
        self.assertEqual(self.output.inputs['Procedural bones'].bl_idname, 'SourceIOProceduralBoneSocket')

    def test_new_and_saved_nodes_have_each_input_once(self):
        # The tree update runs while init adds sockets; its migration must not leave a second copy that links can be
        # dropped on and ignored.
        entry = self.tree.nodes.new('SourceIOModelEntryNode')
        for node in (self.output, entry):
            names = [socket.name for socket in node.inputs]
            self.assertEqual(len(names), len(set(names)), names)
        rename = self.tree.nodes.new('SourceIOBoneRenameNode')
        early = self.output.inputs.new('SourceIOBoneNamesSocket', 'Bone names')
        self.tree.links.new(rename.outputs['Bone names'], early)
        self.output.inputs.move(len(self.output.inputs) - 1, 1)
        self.tree.update()
        sockets = [socket for socket in self.output.inputs if socket.name == 'Bone names']
        self.assertEqual(len(sockets), 1)
        self.assertEqual(sockets[0].links[0].from_node, rename)

    def _helper_rig(self):
        """Rolled arm with a hand and a twist helper, skinned to all bones."""
        rig_data = bpy.data.armatures.new('helper_test_rig')
        rig = bpy.data.objects.new('helper_test_armature', rig_data)
        bpy.context.scene.collection.objects.link(rig)
        self.addCleanup(lambda: (bpy.data.objects.remove(rig, do_unlink=True), bpy.data.armatures.remove(rig_data)))
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode='EDIT')
        for name, parent, head, tail, roll in (
                ('root', None, (0, 0, 0), (0, 0, 1), 0),
                ('arm', 'root', (0, 0, 1), (1, 0.2, 1.1), 0.4),
                ('hand.R', 'arm', (1, 0.2, 1.1), (1.3, 0.1, 1.0), -0.3),
                ('twist', 'arm', (0.5, 0.1, 1.05), (0.8, 0.15, 1.08), 0.2)):
            bone = rig_data.edit_bones.new(name)
            bone.head, bone.tail, bone.roll = head, tail, roll
            if parent:
                bone.parent = rig_data.edit_bones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')
        geometry = self.mesh_node('helper_body')
        geometry.rig = rig
        for index, name in enumerate(('arm', 'hand.R', 'twist')):
            geometry.obj.vertex_groups.new(name=name).add([index], 1.0, 'REPLACE')
        self.model_entry(geometry)
        self.output.model_scale = 10.0
        node = self.tree.nodes.new('SourceIOHelperBoneNode')
        node.bone, node.control_bone = 'twist', 'hand.R'
        self.tree.links.new(node.outputs['Helper bone'], self.output.inputs['Procedural bones'])
        return rig, node

    def test_helper_bone_triggers_are_captured_poses_written_to_a_vrd(self):
        rig, node = self._helper_rig()
        self.assertIs(procedural_bone_nodes._helper_armature(node), rig)
        procedural_bone_nodes.capture_trigger(node, node.triggers.add(), rig)
        hand, twist = rig.pose.bones['hand.R'], rig.pose.bones['twist']
        hand.rotation_mode = twist.rotation_mode = 'XYZ'
        hand.rotation_euler = (0, radians(90), radians(10))
        twist.rotation_euler = (0, radians(45), 0)
        twist.location = (0, 0.1, 0)
        bpy.context.view_layer.update()
        posed = node.triggers.add()
        posed.tolerance = 60
        procedural_bone_nodes.capture_trigger(node, posed, rig)
        arm = rig.pose.bones['arm'].matrix
        hand_local = np.array(arm.inverted() @ hand.matrix)
        twist_local = np.array(arm.inverted() @ twist.matrix)

        with self.assertRaisesRegex(GraphBuildError, "'hand.R' contains '.'"):
            plan_model(self.output)
        table = bpy.data.texts.new('helper_renames')
        self.texts.append(table)
        table.write('hand.R = hand_R\n')
        rename = self.tree.nodes.new('SourceIOBoneRenameNode')
        rename.table = table
        self.tree.links.new(rename.outputs['Bone names'], self.output.inputs['Bone names'])
        plan = plan_model(self.output)
        vrd_path = plan.qc_path.with_suffix('.vrd')
        self.assertIn('$proceduralbones "test.vrd"', plan.qc_text.splitlines())
        self.assertIn(f'Asset: {vrd_path}', plan.manifest())
        export_plan(plan)
        text = vrd_path.read_text(encoding='utf-8')
        (helper,) = procedural_bones.parse_vrd(text)
        self.assertEqual((helper.bone, helper.parent, helper.control_parent, helper.control),
                         ('twist', 'arm', 'arm', 'hand_R'))
        rest, trigger = helper.triggers
        self.assertEqual((rest.tolerance, trigger.tolerance), (90, 60))
        # The engine compares the control's parent-relative rotation and poses the helper relative to its parent, in
        # exported units.
        trigger_rotation = procedural_bones.euler_matrix(np.radians(trigger.trigger))
        np.testing.assert_allclose(trigger_rotation, hand_local[:3, :3], atol=1e-5)
        np.testing.assert_allclose(trigger.rotation, twist_local[:3, :3], atol=1e-5)
        np.testing.assert_allclose(trigger.position, twist_local[:3, 3] * 10, atol=1e-4)

        for bone in rig.pose.bones:
            bone.matrix_basis = Matrix.Identity(4)
        procedural_bone_nodes.pose_trigger(node, posed, rig)
        bpy.context.view_layer.update()
        np.testing.assert_allclose(np.array(arm.inverted() @ hand.matrix), hand_local, atol=1e-5)
        np.testing.assert_allclose(np.array(arm.inverted() @ twist.matrix), twist_local, atol=1e-5)

        fields = ('control_rotation', 'helper_rotation', 'helper_location')
        recorded = {field: tuple(getattr(posed, field)) for field in fields}
        self.tree.nodes.remove(node)
        (imported,) = procedural_bone_nodes.import_vrd(self.output, text)
        self.assertIs(imported.armature, rig)
        self.assertEqual((imported.bone, imported.control_bone), ('twist', 'hand.R'))
        self.assertEqual(len(imported.triggers), 2)
        for field in fields:
            np.testing.assert_allclose(tuple(getattr(imported.triggers[1], field)), recorded[field], atol=1e-5)
        (again,) = procedural_bones.parse_vrd(dict(plan_model(self.output).assets)[vrd_path].decode('utf-8'))
        self.assertEqual(again.control, 'hand_R')
        for got, expected in zip(again.triggers, helper.triggers, strict=True):
            np.testing.assert_allclose(got.trigger, expected.trigger, atol=1e-3)
            np.testing.assert_allclose(got.rotation, expected.rotation, atol=1e-5)
            np.testing.assert_allclose(got.position, expected.position, atol=1e-4)

    def test_live_preview_poses_the_helper_from_the_control(self):
        rig, node = self._helper_rig()
        hand, twist = rig.pose.bones['hand.R'], rig.pose.bones['twist']
        procedural_bone_nodes.capture_trigger(node, node.triggers.add(), rig)
        hand.rotation_mode = twist.rotation_mode = 'XYZ'
        hand.rotation_euler = (0, radians(90), 0)
        twist.rotation_euler = (0, radians(40), 0)
        bpy.context.view_layer.update()
        procedural_bone_nodes.capture_trigger(node, node.triggers.add(), rig)
        for bone in (hand, twist):
            bone.matrix_basis = Matrix.Identity(4)

        node.live_preview = True
        hand.rotation_euler = (0, radians(45), 0)
        bpy.context.view_layer.update()  # the update handler poses the helper
        bpy.context.view_layer.update()
        self.assertAlmostEqual(twist.matrix_basis.to_euler('XYZ').y, radians(20), places=3)
        shares, error = procedural_bone_nodes._preview_state[node.as_pointer()]
        self.assertIsNone(error)
        np.testing.assert_allclose(shares, (0.5, 0.5), atol=1e-4)

        node.control_bone = 'missing'
        procedural_bone_nodes.update_previews()
        self.assertIn('Set the helper and control bones', procedural_bone_nodes._preview_state[node.as_pointer()][1])
        node.control_bone = 'hand.R'
        node.live_preview = False
        self.assertEqual(twist.matrix_basis, Matrix.Identity(4))

    def test_vrd_import_keeps_file_order_and_rejects_unknown_bones(self):
        rig, node = self._helper_rig()
        self.tree.nodes.remove(node)
        text = ('<helper> twist arm arm hand.R\n<basepos> 0 0 0\n'
                '<trigger> 90 0 0 0 0 0 0 0 0 0\n'
                '<helper> hand.R arm root arm\n'
                '<trigger> 90 0 0 0 0 0 0 0 0 0\n')
        created = procedural_bone_nodes.import_vrd(self.output, text)
        self.assertEqual([item.bone for item in created], ['twist', 'hand.R'])
        model = prop_nodes.collect_model(self.output, GraphResolver())
        self.assertEqual([item.bone for item in model.procedural_bones], ['twist', 'hand.R'])
        with self.assertRaisesRegex(ValueError, "'thumb' is not in the armature"):
            procedural_bone_nodes.import_vrd(
                self.output, '<helper> thumb arm arm hand.R\n<trigger> 90 0 0 0 0 0 0 0 0 0\n')

    def _compiled_summary(self, plan):
        """Structure of a compiled model that the QC route and direct export must agree on."""
        import struct
        from SourceIO.library.models.mdl.v49.mdl_file import MdlV49
        from SourceIO.library.utils import MemoryBuffer
        data = plan.model_file.read_bytes()
        mdl = MdlV49.from_buffer(MemoryBuffer(data))
        names = [bone.name for bone in mdl.bones]
        rules = [bone.procedural_rule_type for bone in mdl.bones if bone.procedural_rule_type]
        return {
            'bones': [(bone.name, names[bone.parent_id] if bone.parent_id >= 0 else None,
                       type(bone.procedural_rule_type).__name__) for bone in mdl.bones],
            'positions': np.array([tuple(bone.position) for bone in mdl.bones]),
            'controls': [names[rule.control_bone_index] for rule in rules],
            'triggers': np.array([(trigger.inverse_tolerance_angle, *trigger.trigger, *trigger.pos, *trigger.quat)
                                  for rule in rules for trigger in rule.triggers]),
            # VTX header numLODs, PHY header solidCount (None without collision)
            'lods': struct.unpack_from('<i', plan.model_file.with_suffix('.dx90.vtx').read_bytes(), 20)[0],
            'solids': (struct.unpack_from('<i', plan.model_file.with_suffix('.phy').read_bytes(), 8)[0]
                       if plan.model_file.with_suffix('.phy').is_file() else None),
            'parts': [(part.name, [(model.vertex_count, len(model.meshes)) for model in part.models])
                      for part in mdl.body_parts],
            'materials': [material.name for material in mdl.materials],
            'skins': [[material.name for material in family] for family in mdl.skin_groups],
            'paths': mdl.materials_paths,
            'attachments': [(item.name, names[item.parent_bone]) for item in mdl.attachments],
            'attachment matrices': np.array([tuple(item.matrix) for item in mdl.attachments]),
            # mstudiohitboxset_t: name, box count, box offset
            'hitboxes': [struct.unpack_from('<i', data, mdl.header.hitbox_set_offset + 12 * index + 4)[0]
                         for index in range(mdl.header.hitbox_set_count)],
            'sequences': [sequence.name for sequence in mdl.sequences],
            'files': sorted(path.name for path in plan.model_file.parent.glob(f'{plan.model_file.stem}.*')),
            'flexes': mdl.flex_names,
            'flex controllers': [(item.name, item.min, item.max) for item in mdl.flex_controllers],
            'flex rules': [(mdl.flex_names[rule.flex_index], [(op.op.name, round(float(op.value), 4))
                                                              for op in rule.flex_ops]) for rule in mdl.flex_rules],
            # flex, partner, vertex animation type and vertex count of every mesh
            'mesh flexes': [[(mdl.flex_names[flex.flex_desc_index],
                              None if flex.partner_index is None else mdl.flex_names[flex.partner_index],
                              flex.vertex_anim_type.name, len(flex.vertex_animations)) for flex in mesh.flexes]
                            for part in mdl.body_parts for model in part.models for mesh in model.meshes],
        }

    def _flex_object(self, name, keys):
        """A 3x3 grid object whose shape keys move vertices: keys maps key -> {vertex: z offset}."""
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([(x, y, 0) for y in range(3) for x in range(-1, 2)], [],
                         [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)])
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        self.meshes.append(mesh.name)
        self.objects.append(obj.name)
        obj.shape_key_add(name='Basis')
        for key, offsets in keys.items():
            block = obj.shape_key_add(name=key)
            for vertex, offset in offsets.items():
                block.data[vertex].co.z += offset
        node = self.tree.nodes.new('SourceIOPropGeometryNode')
        node.basename = name
        node.obj = obj
        return node

    def _compare_routes(self):
        """Compile through the QC route (DMX, built-in studiomdl) and directly; return both summaries."""
        try:
            from SourceIO.library.utils.pylib import studiomdl  # noqa: F401
        except ImportError:
            self.skipTest('this build has no built-in studiomdl')
        self.output.compiler_backend, self.output.geometry_format = 'BUILTIN', 'DMX'
        reference = plan_model(self.output)
        compile_plan(reference)
        expected = self._compiled_summary(reference)
        self.output.content_root = str(self.root / 'direct content')
        self.output.game_root = str(self.root / 'direct game')
        self.output.compiler_backend = 'DIRECT'
        plan = plan_model(self.output)
        self.assertEqual(plan.direct_issues, ())
        compile_plan(plan)
        actual = self._compiled_summary(plan)
        # DMX material names count as relative, so the DMX route stores an empty $cdmaterials path (and skips the
        # default one); direct export bakes materials like the SMD route, compared in the SMD parity test
        del actual['paths'], expected['paths']
        for key in ('positions', 'attachment matrices', 'triggers'):
            np.testing.assert_allclose(actual.pop(key), expected.pop(key), atol=1e-3, err_msg=key)
        self.maxDiff = None
        self.assertEqual(actual, expected)
        return expected

    def test_direct_flexes_from_node_controls_match_the_dmx_route(self):
        face = self._flex_object('flex_face', {'smile': {0: .2, 1: .1, 2: .2}, 'blink.L': {6: .1}, 'blink.R': {8: .1},
                                              'smile_blink.L': {6: .05}})
        entry = self.model_entry(face, 'face')
        for name, raw, second, stereo, wrinkle in (('smile', 'smile', '', True, 1.5),
                                                   ('blink', 'blink.L', 'blink.R', False, 0.0)):
            control = entry.flex_controllers.add()
            control.name, control.raw_name, control.second_raw_name = name, raw, second
            control.stereo, control.wrinkle_scale = stereo, wrinkle
        self.model_entry(self._flex_object('flex_teeth', {'open': {4: .3}}), 'teeth')  # automatic controls
        summary = self._compare_routes()
        self.assertIn(('right_smile', 0.0, 1.0), summary['flex controllers'])
        self.assertIn(('smileL', 'smileR', 'WRINKLE', 6), summary['mesh flexes'][1])

    def test_direct_flexes_from_graph_rules_match_the_dmx_route(self):
        collector, face, teeth = self._flex_rules()
        for obj, moves in ((face, {'smile': 0, 'blink.L': 1, 'blink.R': 2}), (teeth, {'open': 1})):
            for key, vertex in moves.items():
                obj.data.shape_keys.key_blocks[key].data[vertex].co.z += .25
        for name, obj in (('face', face), ('teeth', teeth)):
            node = self.tree.nodes.new('SourceIOPropGeometryNode')
            node.basename, node.obj = f'rules_{name}', obj
            self.model_entry(node, name)
        self.tree.links.new(collector.outputs['Flex Rules'], self.output.inputs['Flex Rules'])
        summary = self._compare_routes()
        self.assertIn('open', summary['flexes'])
        self.assertTrue(summary['flex rules'])

    def test_direct_flexes_move_separate_vertices_at_one_position_apart(self):
        # Closed lips: the upper and lower lip are separate vertices at the same positions; opening moves them apart
        mesh = bpy.data.meshes.new('lips')
        mesh.from_pydata([(-1, 0, .1), (0, 0, 0), (1, 0, .1), (-1, 0, -.1), (0, 0, 0), (1, 0, -.1)], [],
                         [(0, 1, 2), (3, 5, 4)])
        mesh.uv_layers.new()
        obj = bpy.data.objects.new('lips', mesh)
        bpy.context.scene.collection.objects.link(obj)
        self.meshes.append(mesh.name)
        self.objects.append(obj.name)
        obj.shape_key_add(name='Basis')
        opened = obj.shape_key_add(name='open')
        opened.data[1].co.z += .2
        opened.data[4].co.z -= .2
        node = self.tree.nodes.new('SourceIOPropGeometryNode')
        node.basename, node.obj = 'lips', obj
        self.model_entry(node, 'lips')
        summary = self._compare_routes()
        self.assertEqual(summary['mesh flexes'][0][0][0], 'open')

    def test_direct_flexes_from_advanced_text_match_the_dmx_route(self):
        face = self._flex_object('advanced_face', {'smile': {0: .2, 4: .1}})
        entry = self.model_entry(face, 'face')
        text = bpy.data.texts.new('direct advanced flex')
        self.texts.append(text)
        text.from_string(advanced_flex_text('smile', 0.6))
        entry.flex_controller_text = text
        summary = self._compare_routes()
        self.assertIn(('smile', 0.0, 1.0), summary['flex controllers'])

    def test_direct_export_compiles_what_the_qc_route_compiles_without_writing_qc_or_meshes(self):
        try:
            from SourceIO.library.utils.pylib import studiomdl  # noqa: F401
        except ImportError:
            self.skipTest('this build has no built-in studiomdl')
        rig, helper = self._helper_rig()
        procedural_bone_nodes.capture_trigger(helper, helper.triggers.add(), rig)
        table = bpy.data.texts.new('direct_renames')
        self.texts.append(table)
        table.write('hand.R = hand_R\n')
        rename = self.tree.nodes.new('SourceIOBoneRenameNode')
        rename.table = table
        self.tree.links.new(rename.outputs['Bone names'], self.output.inputs['Bone names'])
        skin, cloth = bpy.data.materials.new('direct_skin'), bpy.data.materials.new('direct_cloth')
        self.materials.extend((skin, cloth))
        bpy.data.objects['helper_body_object'].data.materials.append(skin)
        for material in (skin, cloth):
            vmt = self.material_node(material, material.name)
            self.tree.links.new(vmt.outputs[0], self.output.inputs['Materials'])
            family = self.tree.nodes.new('SourceIOPropSkinFamilyNode')
            self.tree.links.new(vmt.outputs[0], family.inputs['Materials'])
            self.tree.links.new(family.outputs[0], self.output.inputs['Skin families'])
        rigged = {}
        for name, material in (('helper_sleeve', cloth), ('helper_low', skin)):
            rigged[name] = node = self.mesh_node(name, material)
            node.rig = rig
            for index, bone in enumerate(('arm', 'hand.R', 'twist')):
                node.obj.vertex_groups.new(name=bone).add([index], 1.0, 'REPLACE')
        bodygroup = self.tree.nodes.new('SourceIOPropBodygroupNode')
        bodygroup.bodygroup_name = 'sleeves'
        self.tree.links.new(self.tree.nodes.new('SourceIOPropBlankNode').outputs[0], bodygroup.inputs['Choices'])
        self.tree.links.new(rigged['helper_sleeve'].outputs[0], bodygroup.inputs['Choices'])
        self.tree.links.new(bodygroup.outputs[0], self.output.inputs['Bodygroups'])
        body = next(node for node in self.tree.nodes if getattr(node, 'basename', '') == 'helper_body')
        lod = self.tree.nodes.new('SourceIOPropLodNode')
        lod.distance = 25
        self.tree.links.new(body.outputs[0], lod.inputs['Original mesh'])
        self.tree.links.new(rigged['helper_low'].outputs[0], lod.inputs['Replacement mesh'])
        self.tree.links.new(lod.outputs[0], self.output.inputs['LODs'])
        collision = self.tree.nodes.new('SourceIOPropCollisionNode')
        collision.mass, collision.concave = 5, True
        hull = self.mesh_node('helper_hull')
        hull.obj.data.clear_geometry()
        hull.obj.data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], [],
                                  [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        self.tree.links.new(hull.outputs[0], collision.inputs[0])
        self.tree.links.new(collision.outputs[0], self.output.inputs['Collision'])
        attachment = self.tree.nodes.new('SourceIOPropAttachmentNode')
        attachment.attachment_name, attachment.bone = 'grip', 'hand.R'
        attachment.position, attachment.rotation = (0.1, 0.2, 0.3), (radians(90), 0, 0)
        self.tree.links.new(attachment.outputs['Attachment'], self.output.inputs['Attachments'])
        hitbox = self.tree.nodes.new('SourceIOPropHitboxNode')
        hitbox.group, hitbox.bone, hitbox.minimum, hitbox.maximum = 1, 'arm', (-1, -1, -1), (1, 1, 1)
        self.tree.links.new(hitbox.outputs[0], self.output.inputs['Hitboxes'])
        self.output.dmx_up_axis = 'Y'

        self.output.compiler_backend, self.output.geometry_format = 'BUILTIN', 'SMD'
        reference = plan_model(self.output)
        compile_plan(reference)
        expected = self._compiled_summary(reference)
        self.assertIn(('twist', 'arm', 'QuatInterpRule'), expected['bones'])
        self.assertEqual((expected['lods'], expected['solids']), (2, 1))

        self.output.content_root = str(self.root / 'direct content')
        self.output.game_root = str(self.root / 'direct game')
        self.output.compiler_backend = 'DIRECT'
        plan = plan_model(self.output)
        self.assertEqual(plan.direct_issues, ())
        self.assertIn('Direct export: built in memory', plan.manifest())
        self.assertNotIn('Geometry: ', plan.manifest())
        log = compile_plan(plan)
        self.assertIn('writing', log)
        actual = self._compiled_summary(plan)
        self.assertFalse(plan.qc_path.exists())
        self.assertFalse(any(path.exists() for _, path in plan.geometry))
        self.assertTrue(all(path.is_file() for path, _ in plan.assets))  # VMTs and the VRD as on the QC route
        for key in ('positions', 'attachment matrices', 'triggers'):
            np.testing.assert_allclose(actual.pop(key), expected.pop(key), atol=1e-3, err_msg=key)
        self.maxDiff = None
        self.assertEqual(actual, expected)

    def test_check_direct_export_does_not_write_cache_or_content(self):
        self.model_entry(self.mesh_node('reference'))
        self.output.compiler_backend = 'DIRECT'
        area = next(area for area in bpy.context.screen.areas if area.type == 'VIEW_3D')
        area_type, ui_type = area.type, area.ui_type
        try:
            area.type = 'NODE_EDITOR'
            area.ui_type = 'sourceio.model_definition'
            area.spaces.active.node_tree = self.tree
            with bpy.context.temp_override(area=area):
                self.assertEqual(
                    bpy.ops.sourceio.check_direct_export(output_node=self.output.name),
                    {'FINISHED'})
        finally:
            area.type, area.ui_type = area_type, ui_type
        self.assertFalse((self.root / 'content').exists())
        self.assertFalse((self.root / 'game').exists())

    def test_direct_export_refuses_unsupported_features_before_writing(self):
        self.model_entry(self.mesh_node('reference'))
        raw = self.tree.nodes.new('SourceIOPropRawQcNode')
        raw.snippet = '$surfaceprop "metal"'
        self.tree.links.new(raw.outputs[0], self.output.inputs['Raw QC'])
        face = self.mesh_node('face')
        face.obj.shape_key_add(name='Basis')
        face.obj.shape_key_add(name='smile')
        face.vertex_animation_ranges = 'pulse 1 3 sequence'
        self.model_entry(face, 'face')
        self.output.compiler_backend = 'DIRECT'
        plan = plan_model(self.output)
        self.assertEqual([issue.severity for issue in plan.direct_issues], ['ERROR'] * 2)
        messages = '\n'.join(issue.message for issue in plan.direct_issues)
        for expected in ("Raw QC '$surfaceprop \"metal\"'", "Vertex animation 'pulse' of 'face'"):
            self.assertIn(expected, messages)
            self.assertIn(expected, plan.manifest())
        with self.assertRaises(CompilationError) as caught:
            compile_plan(plan)
        self.assertIn("Direct export can't build this model", str(caught.exception))
        self.assertIn("Raw QC '$surfaceprop \"metal\"'", caught.exception.log)
        self.assertFalse((self.root / 'game').exists())
        self.assertFalse((self.root / 'content').exists())
        store_direct_issues(self.output, plan.direct_issues)
        self.assertEqual(self.output.direct_issues.splitlines()[0].split('\t')[0], 'ERROR')


if __name__ == '__main__':
    unittest.main()
