"""Run with a Python environment providing bpy: python -m unittest ...test_skins."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import bpy

from SourceIO.blender_bindings.shared.skins import set_skin, prop_skin
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material
from SourceIO.blender_bindings.operators import shared_operators as operators


class SkinTests(unittest.TestCase):
    def setUp(self):
        self.a = bpy.data.materials.new('skin_a')
        self.b = bpy.data.materials.new('skin_b')
        self.c = bpy.data.materials.new('skin_c')

    def mesh(self, materials):
        mesh = bpy.data.meshes.new('skin_mesh')
        obj = bpy.data.objects.new('skin_test', mesh)
        for mat in materials:
            mesh.materials.append(mat)
        return obj

    def source1(self, pointers=True):
        obj = self.mesh([self.a, self.b, self.c])
        obj['model_type'] = 's1'
        obj['active_skin'] = '0'
        groups = {'0': [self.a, self.b], '1': [self.c, self.c]}
        obj['skin_groups'] = groups if pointers else {k: [m.name for m in v] for k, v in groups.items()}
        return obj

    def test_round_trip_preserves_slots_with_shared_replacements(self):
        for pointers in (True, False):
            obj = self.source1(pointers)
            self.assertTrue(set_skin(obj, '1'))
            self.assertEqual([s.material for s in obj.material_slots], [self.c] * 3)
            self.assertTrue(set_skin(obj, 0))
            self.assertEqual([s.material for s in obj.material_slots], [self.a, self.b, self.c])

    def test_linked_meshes_are_independent(self):
        obj = self.source1()
        other = obj.copy()
        set_skin(obj, '1')
        self.assertEqual(other.material_slots[0].material, self.a)
        self.assertEqual(obj.material_slots[0].material, self.c)

    def test_invalid_skin_does_not_change_state(self):
        obj = self.source1()
        self.assertFalse(set_skin(obj, 'missing'))
        self.assertEqual(obj['active_skin'], '0')
        self.assertEqual(obj.material_slots[0].material, self.a)

    def test_source2_full_paths_and_numeric_groups(self):
        a = get_or_create_material('same', 'materials/one/same.vmat')
        b = get_or_create_material('same', 'materials/two/same.vmat')
        obj = self.mesh([a])
        obj['model_type'] = 'S2'
        obj['active_skin'] = 'original'
        obj['skin_groups'] = {'original': 'materials/one/same.vmat', 'alternate': 'materials/two/same.vmat'}
        self.assertTrue(set_skin(obj, 1))
        self.assertEqual(obj.material_slots[0].material, b)
        self.assertEqual(obj['active_skin'], 'alternate')
        set_skin(obj, 'original')
        self.assertEqual(obj.material_slots[0].material, a)

    def test_blender_skin_operator(self):
        obj = self.source1()
        bpy.context.scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        bpy.utils.register_class(operators.SOURCEIO_OT_ChangeSkin)
        try:
            self.assertEqual(bpy.ops.sourceio.select_skin(skin_name='1'), {'FINISHED'})
            self.assertEqual(obj.material_slots[0].material, self.c)
            self.assertEqual(bpy.ops.sourceio.select_skin(skin_name='0'), {'FINISHED'})
            self.assertEqual(obj.material_slots[0].material, self.a)
        finally:
            bpy.utils.unregister_class(operators.SOURCEIO_OT_ChangeSkin)

    def test_prop_skin_locations(self):
        for data in ({'skin': 2}, {'entity': {'skin': '2'}}):
            obj = {'entity_data': data}
            self.assertEqual(prop_skin(obj, '0'), '2')
        self.assertEqual(prop_skin({'skin': 0}, 'default'), '0')

    def test_prop_import_cache_separates_skins(self):
        for source2 in (False, True):
            bpy.context.scene['INSTANCE_CACHE'] = {}
            imported = []
            def import_model(*args):
                obj = self.source1()
                container = ModelContainer([obj], {'body': [obj]})
                imported.append(container)
                return container
            context = SimpleNamespace(scene=SimpleNamespace(
                use_instances=True, import_materials=False, replace_entity=False,
                use_bvlg=False, import_physics=False))
            loader = SimpleNamespace(apply_prop_pose=lambda *args: None,
                                     report=lambda *args: self.fail(str(args)))
            content = SimpleNamespace(find_file=lambda path: object(), get_steamid_from_asset=lambda path: None)
            placeholders = []
            with patch.object(operators, 'import_model', side_effect=import_model), \
                 patch.object(operators, 'load_model', side_effect=import_model), \
                 patch.object(operators.CompiledModelResource, 'from_buffer', return_value=SimpleNamespace(name='cache_test')):
                for skin in ('0', '1', '1'):
                    obj = bpy.data.objects.new('placeholder', None)
                    bpy.context.scene.collection.objects.link(obj)
                    obj['entity_data'] = {'prop_path': 'cache_test.vmdl_c' if source2 else 'cache_test.mdl',
                                          'skin': skin, 'entity': {}, 'type': 'static_prop', 'scale': 1.0}
                    method = operators.SourceIO_OT_LoadEntity.load_vmdl if source2 else operators.SourceIO_OT_LoadEntity.load_mdl
                    method(loader, content, context, obj)
                    placeholders.append(obj)
            self.assertEqual(len(imported), 2)
            self.assertNotEqual(placeholders[0].instance_collection, placeholders[1].instance_collection)
            self.assertEqual(placeholders[1].instance_collection, placeholders[2].instance_collection)
            self.assertEqual(imported[0].objects[0].material_slots[0].material, self.a)
            self.assertEqual(imported[1].objects[0].material_slots[0].material, self.c)


if __name__ == '__main__':
    unittest.main()
