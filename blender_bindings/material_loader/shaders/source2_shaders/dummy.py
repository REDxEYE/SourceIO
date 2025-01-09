from pprint import pformat

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.library.source2.blocks.kv3_block import KVBlock


class DummyShader(Source2ShaderBase):
    SHADER: str = 'DUMMY'

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        data = self._material_resource.get_block(KVBlock,block_name='DATA')
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, data["m_shaderName"])
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        self.logger.info(pformat(dict(data)))
        if data:
            for i, param in enumerate(data['m_textureParams']):
                x = i % 4
                y = i // 4
                texture_path = self._material_resource.get_texture_property(param['m_name'], None)
                if texture_path is not None:
                    image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
                    self.create_texture_node(image, param['m_name'], (-600 - x * 200, y * 280))
            for i, param in enumerate(data['m_intParams']):
                x = i % 4
                y = i // 4
                node = self.create_node(Nodes.ShaderNodeValue, param['m_name'], (-600 - x * 200, -400 - y * 200))
                node.outputs[0].default_value = param['m_nValue']

            for i, param in enumerate(data['m_floatParams']):
                x = i % 4
                y = i // 4
                node = self.create_node(Nodes.ShaderNodeValue, param['m_name'], (-600 - x * 200, -600 - y * 200))
                node.outputs[0].default_value = param['m_flValue']

            for i, param in enumerate(data['m_vectorParams']):
                x = i % 4
                y = i // 4
                node = self.create_node(Nodes.ShaderNodeRGB, param['m_name'], (-600 - x * 200, -900 - y * 200))
                node.outputs[0].default_value = param['m_value']
