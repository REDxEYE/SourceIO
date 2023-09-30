from pprint import pformat

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOBlackUnlit(Source2ShaderBase):
    SHADER: str = 'csgo_black_unlit.vfx'

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_black_unlit.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data, = material_data.get_data_block(block_name='DATA')
        self.logger.info(pformat(dict(data)))
