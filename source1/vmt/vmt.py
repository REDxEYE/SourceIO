from ..content_manager import ContentManager
from ...utilities.keyvalues import KVParser


class VMT:
    def __init__(self, file_object):

        kv_parser = KVParser('VMT', file_object.read(-1).decode().replace('`', ''))
        self.shader, self.material_data = kv_parser.parse()
        self.textures = {}

    def parse(self):
        content_manager = ContentManager()
        if self.shader.lower() == 'patch':
            original_material = content_manager.find_file(self.material_data['include'])
            if original_material:
                kv_parser = KVParser('VMT', original_material.read(-1).decode())
                self.shader, self.material_data = kv_parser.parse()
                if 'insert' in self.material_data:
                    patch_data = self.material_data['insert']
                    self.material_data.update(patch_data)
                if 'replace' in self.material_data:
                    patch_data = self.material_data['replace']
                    self.material_data.update(patch_data)
            else:
                print('Failed to find original material')
                return
