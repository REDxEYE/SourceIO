from pathlib import Path
from typing import Dict, Union

from .....logger import SLoggingManager
from ...shader_base import Nodes
from ..source2_shader_base import Source2ShaderBase
import bpy

log_manager = SLoggingManager()


class Skybox(Source2ShaderBase):
    SHADER: str = 'sky.vfx'

    def __init__(self, source2_material, resources: Dict[Union[str, int], Path]):
        super().__init__(source2_material, resources)
        self.logger = log_manager.get_logger(f'Shaders::{self.SHADER}')
        self.do_arrange = True

    @property
    def sky_texture(self):
        texture_path = self.get_texture('g_tSkyTexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            return image
        return None

    def create_nodes(self, material_name):
        self.logger.info(f'Creating material {repr(material_name)}')
        self.bpy_material = bpy.data.worlds.get(material_name, False) or bpy.data.worlds.new(material_name)

        if self.bpy_material is None:
            self.logger.error('Failed to get or create material')
            return 'UNKNOWN'

        if self.bpy_material.get('source_loaded'):
            return 'LOADED'

        self.bpy_material.use_nodes = True
        self.clean_nodes()
        self.bpy_material['source_loaded'] = True

        material_output = self.create_node(Nodes.ShaderNodeOutputWorld)
        shader = self.create_node(Nodes.ShaderNodeBackground, self.SHADER)
        self.connect_nodes(shader.outputs['Background'], material_output.inputs['Surface'])

        texture = self.create_node(Nodes.ShaderNodeTexEnvironment)
        texture.image = self.sky_texture
        self.connect_nodes(texture.outputs['Color'], shader.inputs['Color'])
