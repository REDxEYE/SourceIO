from typing import Dict, List

from .resource import CompiledResource


class CompiledMaterialResource(CompiledResource):
    def get_used_textures(self):
        data_block, = self.get_data_block(block_name='DATA')
        used_textures = {}
        for texture in data_block['m_textureParams']:
            used_textures[texture['m_name']] = texture['m_pValue']
        return used_textures

    def get_int_property(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        return self._get_prop(prop_name, data['m_intParams'], 'm_nValue') or default

    def get_float_property(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        return self._get_prop(prop_name, data['m_floatParams'], 'm_flValue') or default

    def get_vector_property(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        value = self._get_prop(prop_name, data['m_vectorParams'], 'm_value')
        if value is None:
            return default
        return value

    def get_texture_property(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        return self._get_prop(prop_name, data['m_textureParams'], 'm_pValue') or default

    def get_dynamic_property(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        return self._get_prop(prop_name, data['m_dynamicParams'], 'error') or default

    def get_dynamic_texture(self, prop_name, default=None):
        data, = self.get_data_block(block_name='DATA')
        return self._get_prop(prop_name, data['m_dynamicTextureParams'], 'error') or default

    @staticmethod
    def _get_prop(prop_name: str, prop_array: List[Dict], prop_value_name):
        for prop in prop_array:
            if prop['m_name'] == prop_name:
                return prop[prop_value_name]
