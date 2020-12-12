from io import StringIO
from pathlib import Path

from ..content_manager import ContentManager
from ...utilities.gameinfo import Gameinfo
from ...utilities.keyvalues import KVParser


class VMT:
    def __init__(self, file_object):

        kv_parser = KVParser('VMT', StringIO(file_object.read(-1).decode()))
        self.shader, self.material_data = kv_parser.parse()
        self.textures = {}

    def parse(self):
        content_manager = ContentManager()
        if self.shader.lower() == 'patch':
            original_material = content_manager.find_file(self.material_data['include'])
            if original_material:
                kv_parser = KVParser('VMT', StringIO(original_material.read(-1).decode()))
                self.shader, self.material_data = kv_parser.parse()
            else:
                print('Failed to find original material')
                return
        for key, value in self.material_data.items():
            if isinstance(value, str):
                texture = content_manager.find_texture(value)
                if texture:
                    print(key, value)
                    self.textures[key] = Path(value).stem, texture
