import numpy as np
from typing import Iterable

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class Refract(Source1ShaderBase):
    SHADER: str = 'refract'

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$normalmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def color2(self):
        color_value, value_type = self._vavle_material.get_vector('$color2', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def bluramount(self):
        return self._vavle_material.get_float('$bluramount', 0)

    @property
    def color(self):
        color_value, value_type = self._vavle_material.get_vector('$color', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        elif len(color_value) > 3:
            color_value = color_value[:3]
        return color_value

    @property
    def refracttint(self):
        color_value, value_type = self._vavle_material.get_vector('$refracttint', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return color_value

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        self.bpy_material.blend_method = 'OPAQUE'
        self.bpy_material.shadow_method = 'NONE'
        self.bpy_material.use_screen_refraction = True
        self.bpy_material.use_backface_culling = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture

            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])

        bumpmap = self.bumpmap
        if bumpmap:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            bumpmap_node.image = bumpmap

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
            shader.inputs['Transmission'].default_value = 1.0
            shader.inputs['Roughness'].default_value = self.bluramount
