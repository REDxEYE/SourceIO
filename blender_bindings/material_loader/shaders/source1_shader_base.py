import bpy
from typing import Dict, Any
import numpy as np

from ..shader_base import ShaderBase
from ...source1.vtf import import_texture
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source1.vmt import VMT


class Source1ShaderBase(ShaderBase):
    def __init__(self, vmt):
        super().__init__()
        self._vmt: VMT = vmt
        self.textures = {}

    def load_texture(self, texture_name, texture_path):
        if bpy.data.images.get(texture_name, False):
            self.logger.debug(f'Using existing texture {texture_name}')
            return bpy.data.images.get(texture_name)

        content_manager = ContentManager()
        texture_file = content_manager.find_texture(texture_path)
        if texture_file is not None:
            return import_texture(texture_name, texture_file)
        return None

    @staticmethod
    def convert_ssbump(image: bpy.types.Image):
        if image.get('ssbump_converted', None):
            return image
        if bpy.app.version > (2, 83, 0):
            buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(buffer)
        else:
            buffer = np.array(image.pixels[:])
        buffer[0::4] *= 0.5
        buffer[0::4] += 0.33
        buffer[1::4] *= 0.5
        buffer[1::4] += 0.33
        buffer[2::4] *= 0.2
        buffer[2::4] += 0.8
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(buffer.tolist())
        else:
            image.pixels[:] = buffer.tolist()
        image.pack()
        image['ssbump_converted'] = True
        return image

    @staticmethod
    def convert_normalmap(image: bpy.types.Image):
        if image.get('normalmap_converted', None):
            return image
        if bpy.app.version > (2, 83, 0):
            buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
            image.pixels.foreach_get(buffer)
        else:
            buffer = np.array(image.pixels[:])

        buffer[1::4] = np.subtract(1, buffer[1::4])
        if bpy.app.version > (2, 83, 0):
            image.pixels.foreach_set(buffer.tolist())
        else:
            image.pixels[:] = buffer.tolist()
        image.pack()
        image['normalmap_converted'] = True
        return image
