from pathlib import Path
from typing import Any, Dict

import bpy
import numpy as np

from ...utils.texture_utils import check_texture_cache, create_and_cache_texture
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source1.vmt import VMT
from ...source1.vtf import import_texture
from ..shader_base import ShaderBase


class Source1ShaderBase(ShaderBase):
    def __init__(self, vmt):
        super().__init__()
        self.load_bvlg_nodes()
        self._vmt: VMT = vmt
        self.textures = {}

    def load_texture(self, texture_name: str, texture_path: Path):
        image = check_texture_cache(texture_path / texture_name)
        if image is not None:
            return image
        if bpy.data.images.get(texture_name, False):
            self.logger.debug(f'Using existing texture {texture_name}')
            return bpy.data.images.get(texture_name)

        content_manager = ContentManager()
        texture_file = content_manager.find_texture(texture_path/texture_name)
        if texture_file is not None:
            return import_texture(texture_path / texture_name, texture_file)
        return None

    @staticmethod
    def convert_ssbump(image: bpy.types.Image):
        if image.get('ssbump_converted', None):
            return image
        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)
        buffer[0::4] *= 0.5
        buffer[0::4] += 0.33
        buffer[1::4] *= 0.5
        buffer[1::4] += 0.33
        buffer[2::4] *= 0.2
        buffer[2::4] += 0.8
        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['ssbump_converted'] = True
        return image

    @staticmethod
    def convert_normalmap(image: bpy.types.Image):
        if image.get('normalmap_converted', None):
            return image

        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)

        buffer[1::4] = np.subtract(1, buffer[1::4])
        image.pixels.foreach_set(buffer.ravel())
        image.pack()
        del buffer
        image['normalmap_converted'] = True
        return image
