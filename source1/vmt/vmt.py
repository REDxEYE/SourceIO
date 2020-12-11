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
        for key, value in self.material_data.items():
            if isinstance(value, str):
                texture = content_manager.find_texture(value)
                if texture:
                    self.textures[key] = Path(value).stem, texture
