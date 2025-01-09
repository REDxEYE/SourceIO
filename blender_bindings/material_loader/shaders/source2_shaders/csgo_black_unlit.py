from pprint import pformat

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class CSGOBlackUnlit(Source2ShaderBase):
    SHADER: str = 'csgo_black_unlit.vfx'

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_black_unlit.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data = self._material_resource.get_block(KVBlock,block_name='DATA')
        self.logger.info(pformat(dict(data)))
