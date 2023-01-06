from array import array
from pathlib import Path
from typing import Dict, Union

import bpy
import numpy as np

from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source2.resource_types2 import (CompiledMaterialResource,
                                                 CompiledTextureResource)
from ....logger import SLoggingManager
from ...source2.vtex_loader import import_texture
from ..shader_base import ShaderBase

logger = SLoggingManager().get_logger("Source2::Shader")


class Source2ShaderBase(ShaderBase):
    def __init__(self, source2_material: CompiledMaterialResource):
        super().__init__()
        self._material_resource = source2_material

    def load_texture_or_default(self, name_or_id: Union[str, int], default_color: tuple = (1.0, 1.0, 1.0, 1.0)):
        print(f'Loading texture {name_or_id}')
        resource = self._material_resource.get_child_resource(name_or_id, ContentManager(), CompiledTextureResource)
        texture_name: str
        if isinstance(name_or_id, int):
            texture_name = f"0x{name_or_id:08}"
        elif isinstance(name_or_id, str):
            texture_name = name_or_id
        else:
            raise Exception(f"Invalid name or id: {name_or_id}")

        return self.load_texture(resource, Path(texture_name)) or self.get_missing_texture(f'missing_{texture_name}',
                                                                                           default_color)

    def split_normal(self, image: bpy.types.Image):
        roughness_name = self.new_texture_name_with_suffix(image.name, 'roughness', 'tga')
        if image.get('normalmap_converted', None):
            return image, bpy.data.images.get(roughness_name, None)

        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)

        mask = buffer[2::4]
        roughness_rgb = np.dstack((mask, mask, mask, np.ones_like(mask)))

        roughness_texture = self.make_texture(roughness_name, image.size, roughness_rgb, True)
        buffer[1::4] = np.subtract(1, buffer[1::4])
        buffer[2::4] = 1.0

        image.pixels.foreach_set(buffer.ravel())

        image.pack()
        image['normalmap_converted'] = True
        return image, roughness_texture

    def load_texture(self, texture_resource, texture_path):
        if texture_resource:
            texture = import_texture(texture_resource, texture_path.stem, True)
            return texture
        return None
