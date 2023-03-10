from ..shader_base import Nodes
from .source2_shader_base import Source2ShaderBase


class DebugMaterial(Source2ShaderBase):
    SHADER = 'vr_xen_foliage.vfx'

    def get_or_default(self, texture_path):
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        data_block, = self._material_resource.get_data_block(block_name='DATA')

        for int_param in data_block['m_intParams']:
            node = self.create_node(Nodes.ShaderNodeValue, int_param['m_name'])
            node.outputs[0].default_value = int_param['m_nValue']

        for float_param in data_block['m_floatParams']:
            node = self.create_node(Nodes.ShaderNodeValue, float_param['m_name'])
            node.outputs[0].default_value = float_param['m_flValue']

        for vector_param in data_block['m_vectorParams']:
            node = self.create_node(Nodes.ShaderNodeRGB, vector_param['m_name'])
            node.outputs[0].default_value = vector_param['m_value']

        for texture_param in data_block['m_textureParams']:
            self.create_texture_node(self.get_or_default(texture_param['m_pValue']), texture_param['m_name'])
