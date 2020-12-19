from ..content_manager import ContentManager
from ...utilities.keyvalues import KVParser


class VMT:
    def __init__(self, file_object):
        kv_parser = KVParser('VMT', file_object.read(-1).decode().replace('`', ''))
        self.shader, self.material_data = kv_parser.parse()

    def get_param(self, name, default):
        return self.material_data.get(name, default)

    def parse(self):
        content_manager = ContentManager()
        if self.shader.lower() == 'patch':
            original_material = content_manager.find_file(self.material_data['include'])
            if original_material:
                kv_parser = KVParser('VMT', original_material.read(-1).decode())
                old_data = self.material_data
                self.shader, self.material_data = kv_parser.parse()
                if 'insert' in old_data:
                    patch_data = old_data['insert']
                    self.material_data.update(patch_data)
                if 'replace' in old_data:
                    patch_data = old_data['replace']
                    self.material_data.update(patch_data)
            else:
                print('Failed to find original material')
                return
