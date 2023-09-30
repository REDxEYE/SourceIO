from pathlib import Path
from typing import Union, Optional, Tuple
import bpy
import numpy as np

from ...utils.texture_utils import check_texture_cache
from ....library.shared.content_providers.content_manager import ContentManager
from ....library.source2.resource_types import (CompiledMaterialResource,
                                                CompiledTextureResource)
from ....logger import SLoggingManager
from ...source2.vtex_loader import import_texture
from ..shader_base import ShaderBase, Nodes

logger = SLoggingManager().get_logger("Source2::Shader")


class Source2ShaderBase(ShaderBase):
    def __init__(self, source2_material: CompiledMaterialResource, tinted: bool = False):
        super().__init__()
        self.load_source2_nodes()
        self._material_resource = source2_material
        self.tinted = tinted

    def _have_texture(self, slot_name: str) -> Optional[bpy.types.Node]:
        texture_path = self._material_resource.get_texture_property(slot_name, None)
        if texture_path is not None:
            return self._material_resource.get_child_resource(texture_path, ContentManager()) is not None
        return False

    def _get_texture(self, slot_name: str, default_color: Tuple[float, float, float, float],
                     is_data=False,
                     invert_y: bool = False):
        texture_path = self._material_resource.get_texture_property(slot_name, None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, default_color, invert_y)
            if is_data:
                image.colorspace_settings.is_data = True
                image.colorspace_settings.name = 'Non-Color'
        else:
            image = self.get_missing_texture(slot_name, default_color)
        texture_node = self.create_node(Nodes.ShaderNodeTexImage, slot_name)
        texture_node.image = image
        return texture_node

    def load_texture_or_default(self, name_or_id: Union[str, int], default_color: tuple = (1.0, 1.0, 1.0, 1.0),
                                invert_y: bool = False):
        print(f'Loading texture {name_or_id}')
        resource = self._material_resource.get_child_resource(name_or_id, ContentManager(), CompiledTextureResource)
        texture_name: str
        if isinstance(name_or_id, int):
            texture_name = f"0x{name_or_id:08}"
        elif isinstance(name_or_id, str):
            texture_name = name_or_id
        else:
            raise Exception(f"Invalid name or id: {name_or_id}")

        return self.load_texture(resource, Path(texture_name), invert_y) or self.get_missing_texture(
            f'missing_{texture_name}',
            default_color)

    def split_normal(self, image: bpy.types.Image):
        roughness_name = self.new_texture_name_with_suffix(image.name, 'roughness', 'tga')
        if image.get('normalmap_converted', None):
            return image, bpy.data.images.get(roughness_name, None)

        buffer = np.zeros(image.size[0] * image.size[1] * 4, np.float32)
        image.pixels.foreach_get(buffer)

        mask = buffer[3::4]
        roughness_rgb = np.dstack((mask, mask, mask, np.ones_like(mask)))

        roughness_texture = self.make_texture(roughness_name, image.size, roughness_rgb, True)
        buffer[3::4] = 1.0

        image.pixels.foreach_set(buffer.ravel())

        image.pack()
        image['normalmap_converted'] = True
        return image, roughness_texture

    def load_texture(self, texture_resource: Optional[CompiledTextureResource], texture_path, invert_y: bool = False):
        if texture_resource is not None:
            texture = check_texture_cache(texture_path)
            if texture is not None:
                return texture
            texture = import_texture(texture_resource, texture_path, True, invert_y)
            return texture
        return None
