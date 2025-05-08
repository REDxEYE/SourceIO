from typing import Union, Optional, Any
import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import ShaderBase, Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.source2.vtex_loader import import_texture
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3
from SourceIO.blender_bindings.utils.texture_utils import check_texture_cache
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.source2.resource_types import CompiledMaterialResource, CompiledTextureResource
from SourceIO.library.utils.perf_sampler import timed
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

logger = SourceLogMan().get_logger("Source2::Shader")


class Source2ShaderBase(ShaderBase):
    def __init__(self, content_manager: ContentManager, source2_material: CompiledMaterialResource,
                 tinted: bool = False):
        super().__init__()
        self.content_manager = content_manager
        self.load_source2_nodes()
        self._material_resource = source2_material
        self.unused_textures = set(self._material_resource.get_used_textures().keys())
        self.tinted = tinted

    def _have_texture(self, slot_name: str) -> Optional[bpy.types.Node]:
        texture_path = self._material_resource.get_texture_property(slot_name, None)
        if texture_path is not None:
            return self._material_resource.has_child_resource(texture_path, self.content_manager)
        return None

    @timed
    def _get_texture(self, slot_name: str, default_color: tuple[float, float, float, float],
                     is_data=False,
                     invert_y: bool = False):
        if slot_name in self.unused_textures:
            self.unused_textures.remove(slot_name)
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

    def _skip_texture(self, slot_name: str):
        if slot_name in self.unused_textures:
            self.unused_textures.remove(slot_name)

    def load_texture_or_default(self, name_or_id: Union[str, int], default_color: tuple = (1.0, 1.0, 1.0, 1.0),
                                invert_y: bool = False):
        # print(f'Loading texture {name_or_id}')
        resource = self._material_resource.get_child_resource(name_or_id, self.content_manager,
                                                              CompiledTextureResource)
        texture_name: str
        if isinstance(name_or_id, int):
            texture_name = f"0x{name_or_id:08}"
        elif isinstance(name_or_id, str):
            texture_name = name_or_id
        else:
            raise Exception(f"Invalid name or id: {name_or_id}")

        return self.load_texture(resource, TinyPath(texture_name), invert_y) or self.get_missing_texture(
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
            texture = import_texture(texture_resource, texture_path, invert_y)
            return texture
        return None

    def create_transform(self, uv_slot: str, scale: tuple[float, ...], offset: tuple[float, ...],
                         center: tuple[float, ...]):
        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        uv_node.uv_map = uv_slot  # "TEXCOORD_1" if self._material_resource.get_int_property("F_SECONDARY_UV", 0) else "TEXCOORD"
        uv_transform = self.create_node_group("UVTransform")
        uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]
        uv_transform.inputs["g_vTexCoordOffset"].default_value = offset[:3]
        uv_transform.inputs["g_vTexCoordCenter"].default_value = center[:3]
        self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

        return uv_transform

    def _check_flag(self, name: str, default: int = 0):
        return self._material_resource.get_int_property(name, default) == 1

    def _handle_alpha_modes(self,
                            alpha_mode: str,
                            alpha_test_ref: float,
                            alpha_output_socket,
                            alpha_input_socket):
        if alpha_mode == "TEST":
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'CLIP'
                self.bpy_material.shadow_method = 'CLIP'
                self.bpy_material.alpha_threshold = alpha_test_ref
            self.connect_nodes(alpha_output_socket, alpha_input_socket)
        elif alpha_mode == "TRANSLUCENT":
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output_socket, alpha_input_socket)
        elif alpha_mode == "OVERLAY":
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output_socket, alpha_input_socket)

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        return super().create_nodes(material, extra_parameters)

    def _handle_self_illum(self, albedo_output, self_illum_mask_output: None | object,
                           self_illum_tint: None | object,
                           albedo_factor: float,
                           emission_strength: float,
                           emission_color_input, emission_strength_input
                           ):
        if self_illum_mask_output is not None:
            color_multiply_node = self.create_node(Nodes.ShaderNodeMixRGB)
            color_multiply_node.blend_type = 'MULTIPLY'
            color_multiply_node.inputs[0].default_value = albedo_factor
            self.connect_nodes(self_illum_mask_output, color_multiply_node.inputs[1])
            self.connect_nodes(albedo_output, color_multiply_node.inputs[2])
            emission_color_output = color_multiply_node.outputs[0]
        else:
            emission_color_output = albedo_output

        if self_illum_tint is not None:
            color_multiply_node = self.create_node(Nodes.ShaderNodeMixRGB)
            color_multiply_node.blend_type = 'MULTIPLY'
            color_multiply_node.inputs[0].default_value = 1.0
            color_multiply_node.inputs[1].default_value = self.ensure_length(self_illum_tint, 4, 1.0)
            self.connect_nodes(emission_color_output, color_multiply_node.inputs[2])
            emission_color_output = color_multiply_node.outputs[0]
        self.connect_nodes(emission_color_output, emission_color_input)
        emission_strength_input.default_value = emission_strength
