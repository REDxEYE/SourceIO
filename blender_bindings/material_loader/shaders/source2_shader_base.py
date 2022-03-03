# from pprint import pformat

import bpy
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

from ..shader_base import ShaderBase
from ...source2.vtex.loader import ValveCompiledTextureLoader
from ....logger import SLoggingManager
from ....library.source2.resource_types import ValveCompiledTexture
from ....library.shared.content_providers.content_manager import ContentManager

logger = SLoggingManager().get_logger("Source2::Shader")


class Source2ShaderBase(ShaderBase):
    def __init__(self, source2_material, resources: Dict[Union[str, int], Path]):
        super().__init__()
        self._material_data: Dict[str, Any] = source2_material
        # logger.print(pformat(self._material_data))
        self.resources: Dict[Union[str, int], Path] = resources

    def _get_param(self, param_type, name, value_type, default):
        for param in self._material_data[param_type]:
            if param['m_name'] == name:
                return param[value_type]
        return default

    def get_int(self, name, default):
        return self._get_param('m_intParams', name, 'm_nValue', default)

    def get_float(self, name, default):
        return self._get_param('m_floatParams', name, 'm_flValue', default)

    def get_vector(self, name, default):
        return self._get_param('m_vectorParams', name, 'm_value', default)

    def get_texture(self, name, default):
        return self._get_param('m_textureParams', name, 'm_pValue', default)

    def get_dynamic(self, name, default):
        return self._get_param('m_dynamicParams', name, 'error', default)

    def get_dynamic_texture(self, name, default):
        return self._get_param('m_dynamicTextureParams', name, 'error', default)

    def split_normal(self, image: bpy.types.Image):
        roughness_name = self.new_texture_name_with_suffix(image.name, 'roughness', 'tga')
        if image.get('normalmap_converted', None):
            return image, bpy.data.images.get(roughness_name, None)
        if bpy.app.version > (2, 83, 0):
            buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(buffer)
        else:
            buffer = np.array(image.pixels[:])

        mask = buffer[2::4]
        roughness_rgb = np.dstack((mask, mask, mask, np.ones_like(mask)))

        roughness_texture = Source2ShaderBase.make_texture(roughness_name, image.size, roughness_rgb, True)
        buffer[1::4] = np.subtract(1, buffer[1::4])
        buffer[2::4] = 1.0
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(buffer.tolist())
        else:
            image.pixels[:] = buffer.tolist()
        image.pack()
        image['normalmap_converted'] = True
        return image, roughness_texture

    def load_texture(self, texture_name, texture_path):

        if texture_path in self.resources:
            proper_path = self.resources[texture_path]
            texture_path = ContentManager().find_file(proper_path)
            if texture_path:
                texture = ValveCompiledTextureLoader(texture_path)
                return texture.import_texture(proper_path.stem, True)
        return None
