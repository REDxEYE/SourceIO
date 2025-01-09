from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()

class Skybox(Source1ShaderBase):
    SHADER: str = 'sky'

    def __init__(self, skybox_texture, skybox_texture_hdr,skybox_texture_hdr_alpha):
        self.logger = log_manager.get_logger(f'Shaders::{self.SHADER}')
        self.skybox_texture = skybox_texture
        self.skybox_texture_hdr = skybox_texture_hdr
        self.skybox_texture_hdr_alpha = skybox_texture_hdr_alpha
        self.do_arrange = True

    def create_nodes(self, material):
        self.logger.info(f'Creating material {repr(material.name)}')
        self.bpy_material = material

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
        texture.image = self.skybox_texture
        self.connect_nodes(texture.outputs['Color'], shader.inputs['Color'])


