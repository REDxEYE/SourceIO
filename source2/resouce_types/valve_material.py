from pathlib import Path

from ..source2 import ValveFile
from .valve_texture import ValveTexture


class ValveMaterial:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()
        self.valve_file.check_external_resources()

    def load(self, flip_textures):
        textures = {}
        data_block = self.valve_file.get_data_block(block_name='DATA')[0]
        if data_block:
            for tex in data_block.data['m_textureParams']:
                texture = self.valve_file.get_child_resource(tex['m_pValue'])
                if texture is not None:
                    print(f"Loading {texture.filepath.stem} texture")
                    tex_file = ValveTexture('', valve_file=texture)
                    tex_file.load(flip_textures)
                    textures[tex['m_name']] = tex_file
                else:
                    textures[tex['m_name']] = None
                    print(f"missing {tex['m_pValue']} texture")
