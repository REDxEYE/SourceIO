from typing import Any

import bpy
import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3


class VrSkin(Source2ShaderBase):
    SHADER: str = 'vr_skin.vfx'

    # @property
    # def ambient_occlusion(self):
    #     return self._get_texture('g_tAmbientOcclusion', (1.0, 1.0, 1.0, 1.0), True)

    # @property
    # def combined_masks(self):
    #     return self._get_texture('g_tCombinedMasks', (1.0, 1.0, 1.0, 1.0), True)

    @property
    def color(self):
        return self._material_resource.get_vector_property('g_vColorTint', np.ones(4, dtype=np.float32))

    @property
    def alpha_test(self):
        return self._material_resource.get_int_property('F_ALPHA_TEST', 0)

    @property
    def metalness(self):
        return self._material_resource.get_int_property('F_METALNESS_TEXTURE', 0)

    @property
    def translucent(self):
        return self._material_resource.get_int_property('F_TRANSLUCENT', 0)

    @property
    def specular(self):
        return self._material_resource.get_int_property('F_SPECULAR', 0)

    @property
    def roughness_value(self):
        value = self._material_resource.get_vector_property('TextureRoughness', None)
        if value is None:
            return
        return value[0]

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        self.bpy_material = material
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        normal_node = self._get_texture('g_tNormal', (0.5, 0.5, 1.0, 1.0), True)
        albedo_node = self._get_texture('g_tColor', (0.3, 0.3, 0.3, 1.0), False)
        color_tint = self.color
        base_color_input = shader.inputs['Base Color']
        if color_tint[0] != 1.0 and color_tint[1] != 1.0 and color_tint[2] != 1.0:
            color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
            color_mix.blend_type = 'MULTIPLY'
            self.connect_nodes(albedo_node.outputs['Color'], color_mix.inputs['Color1'])
            color = color_tint
            if sum(color) > 3:
                color = list(np.divide(color, 255))
            color_mix.inputs['Color2'].default_value = color
            color_mix.inputs['Fac'].default_value = 1.0
            base_color_output = color_mix.outputs['Color']
        else:
            base_color_output = albedo_node.outputs['Color']

        if extra_parameters.get(ExtraMaterialParameters.USE_OBJECT_TINT, False):
            base_color_output = self.insert_object_tint(base_color_output, 1.0)
        self.connect_nodes(base_color_output, base_color_input)

        if self.translucent or self.alpha_test:
            if not is_blender_4_3():
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'HASHED'
            self.connect_nodes(albedo_node.outputs['Alpha'], shader.inputs['Alpha'])
        elif self.metalness:
            self.connect_nodes(albedo_node.outputs['Alpha'], shader.inputs['Metallic'])

        normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

        self.connect_nodes(normal_node.outputs['Color'], normalmap_node.inputs['Color'])
        self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

        if self.roughness_value is None:
            self.connect_nodes(normal_node.outputs[1], shader.inputs['Roughness'])
        else:
            shader.inputs['Roughness'].default_value = self.roughness_value

        self._skip_texture("g_tOcclusion")
        self._skip_texture("g_tShadowFalloff")
        self._skip_texture("g_tDiffuseFalloff")
        self._skip_texture("g_tCombinedMasks")