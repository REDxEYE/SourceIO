import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from SourceIO.blender_bindings.models.materials import get_model_material_names, resolve_model_material


class MaterialPathTests(unittest.TestCase):
    def model(self, *names):
        return SimpleNamespace(materials=[SimpleNamespace(name=name, bpy_material=None) for name in names],
                               materials_paths=['', 'models/props', 'models/props'])

    def test_mesh_import_reuses_successes_and_misses_without_lookups(self):
        mdl = self.model('tube', 'missing')
        cm = Mock()
        cm.check.side_effect = lambda p: str(p) == 'materials/models/props/tube.vmt'
        for mat in mdl.materials:
            resolve_model_material(cm, mdl, mat.name)
        self.assertEqual([str(c.args[0]) for c in cm.check.call_args_list], [
            'materials/tube.vmt', 'materials/models/props/tube.vmt',
            'materials/missing.vmt', 'materials/models/props/missing.vmt'])
        cm.check.reset_mock()
        self.assertEqual(get_model_material_names(cm, mdl), {'tube': 'models/props/tube', 'missing': 'missing'})
        get_model_material_names(cm, mdl)
        resolve_model_material(cm, mdl, 'missing')
        cm.check.assert_not_called()
        cm.find_file.assert_not_called()

    def test_without_material_import_resolves_once(self):
        mdl = self.model('tube', 'tube')
        cm = Mock()
        cm.check.return_value = True
        self.assertEqual(get_model_material_names(cm, mdl), {'tube': 'tube'})
        get_model_material_names(cm, mdl)
        self.assertEqual(cm.check.call_count, 1)
        cm.find_file.assert_not_called()

    def test_full_paths_and_dotted_names(self):
        mdl = self.model('models/props/tube.blue')
        cm = Mock()
        cm.check.return_value = True
        resolve_model_material(cm, mdl, mdl.materials[0].name)
        resolve_model_material(cm, mdl, mdl.materials[0].name)
        self.assertEqual([str(c.args[0]) for c in cm.check.call_args_list],
                         ['materials/models/props/tube.blue.vmt'])

    def test_cache_does_not_leak_between_models(self):
        cm = Mock()
        cm.check.return_value = False
        get_model_material_names(cm, self.model('tube'))
        cm.check.return_value = True
        self.assertEqual(get_model_material_names(cm, self.model('tube')), {'tube': 'tube'})
        self.assertEqual(cm.check.call_count, 3)

    def test_already_loaded_material_retains_pointer_and_resolved_path(self):
        from SourceIO.blender_bindings.models.mdl36 import import_mdl
        mdl = self.model('tube')
        cm = Mock()
        cm.check.side_effect = lambda p: str(p) == 'materials/models/props/tube.vmt'
        material = {'source1_loaded': True}
        with patch.object(import_mdl, 'get_or_create_material', return_value=material):
            import_mdl.import_materials(cm, mdl)
        self.assertIs(mdl.materials[0].bpy_material, material)
        cm.check.reset_mock()
        self.assertEqual(get_model_material_names(cm, mdl), {'tube': 'models/props/tube'})
        cm.check.assert_not_called()
        cm.find_file.assert_not_called()

    def test_shader_import_opens_only_resolved_file(self):
        from SourceIO.blender_bindings.models.mdl36 import import_mdl
        mdl = self.model('tube')
        cm = Mock()
        cm.check.side_effect = lambda p: str(p) == 'materials/models/props/tube.vmt'
        material = {}
        with patch.object(import_mdl, 'get_or_create_material', return_value=material), \
             patch.object(import_mdl, 'VMT') as vmt, \
             patch.object(import_mdl.ShaderRegistry, 'source1_create_nodes', return_value=material):
            import_mdl.import_materials(cm, mdl)
        self.assertEqual(cm.find_file.call_count, 1)
        self.assertEqual(str(cm.find_file.call_args.args[0]), 'materials/models/props/tube.vmt')
        self.assertIs(vmt.call_args.args[0], cm.find_file.return_value)

    def test_generic_path_collection_does_not_open_files(self):
        from SourceIO.library.utils.path_utilities import collect_full_material_names
        cm = Mock()
        cm.check.side_effect = lambda p: str(p) == 'materials/models/props/tube.vmt'
        self.assertEqual(collect_full_material_names(['tube', 'missing'], ['models/props'], cm),
                         {'tube': 'models/props/tube', 'missing': 'missing'})
        cm.find_file.assert_not_called()


if __name__ == '__main__':
    unittest.main()
