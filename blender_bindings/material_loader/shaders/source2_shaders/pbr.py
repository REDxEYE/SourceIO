import numpy as np

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3


class PBR(Source2ShaderBase):
    SHADER: str = 'pbr.vfx'

    @property
    def flag_self_illum(self):
        return self._material_resource.get_int_property('F_SELF_ILLUM', 0)

    @property
    def flag_alpha_test(self):
        return self._material_resource.get_int_property('F_ALPHA_TEST', 0)

    @property
    def flag_render_backfaces(self):
        return self._material_resource.get_int_property('F_RENDER_BACKFACES', 0)

    @property
    def color_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tColor', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0.3, 0.3, 1.0))
            return image
        return None

    @property
    def ambient_occlusion(self):
        texture_path = self._material_resource.get_texture_property('g_tAmbientOcclusion', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def normal_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tNormalRoughness', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 0.5))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def tint_mask_displacement_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tTintMaskDisplacement', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def selfillum_mask_texture(self):
        texture_path = self._material_resource.get_texture_property('g_tSelfIllumMask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            return image
        return None

    @property
    def selfillum_color_tint(self):
        return self._material_resource.get_vector_property('g_vSelfIllumTint1', np.ones(4, dtype=np.float32))

    @property
    def selfillum_scale(self):
        return self._material_resource.get_float_property('g_flSelfIllumScale1', 0)

    @property
    def selfillum_albedo_factor(self):
        return self._material_resource.get_float_property('g_flSelfIllumAlbedoFactor1', 0)

    @property
    def color_tint(self):
        return self._material_resource.get_vector_property('g_vColorTint1', np.ones(4, dtype=np.float32))

    @property
    def albedo_texcoord_scale1(self):
        return self._material_resource.get_vector_property('g_vAlbedoTexcoordScale1', np.asarray((1., 1., 0., 0.)))

    @property
    def albedo_texcoord_rotation1(self):
        return self._material_resource.get_float_property('g_flAlbedoTexcoordRotation1', 0)

    @property
    def albedo_texcoord_offset1(self):
        return self._material_resource.get_vector_property('g_vAlbedoTexcoordOffset1', np.zeros(4, dtype=np.float32))

    @property
    def normal_texcoord_scale1(self):
        return self._material_resource.get_vector_property('g_vNormalTexcoordScale1', np.asarray((1., 1., 0., 0.)))

    @property
    def normal_texcoord_offset1(self):
        return self._material_resource.get_vector_property('g_vNormalTexcoordOffset1', np.zeros(4, dtype=np.float32))

    @property
    def selfillum_texcoord_scale1(self):
        return self._material_resource.get_vector_property('g_vSelfillumTexcoordScale1', np.asarray((1., 1., 0., 0.)))

    @property
    def selfillum_texcoord_offset1(self):
        return self._material_resource.get_vector_property('g_vSelfillumTexcoordOffset1', np.zeros(4, dtype=np.float32))

    def setup_uv_transform(self, offset: tuple[float, ...], scale: tuple[float, ...]):
        if all(a == 0 for a in offset) and (scale == (1., 1., 0.0, 0.0)).all():
            return None
        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        mapping_node = self.create_node(Nodes.ShaderNodeMapping)
        self.connect_nodes(uv_node.outputs[0], mapping_node.inputs[0])
        mapping_node.inputs[1].default_value = offset[:3]
        mapping_node.inputs[3].default_value = scale[:3]
        return mapping_node

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        color_texture = self.color_texture
        if color_texture:
            color_map_node = self.create_texture_node(color_texture, "color")
            if self.flag_alpha_test:
                self.connect_nodes(color_map_node.outputs[1], shader.inputs["Alpha"])
                if not is_blender_4_3():
                    self.bpy_material.blend_method = 'HASHED'
                    self.bpy_material.shadow_method = 'HASHED'
            else:
                self.connect_nodes(color_map_node.outputs[1],shader.inputs["Metallic"])
            uv_mapping = self.setup_uv_transform(self.albedo_texcoord_offset1, self.albedo_texcoord_scale1)
            if uv_mapping:
                self.connect_nodes(uv_mapping.outputs[0], color_map_node.inputs[0])
            if (self.color_tint != (1.0, 1.0, 1.0, 0.0)).all():
                color_mix_node = self.create_node(Nodes.ShaderNodeMix)
                color_mix_node.data_type = "RGBA"
                color_mix_node.blend_type = "MULTIPLY"
                color_mix_node.inputs[0].default_value = 1
                self.connect_nodes(color_map_node.outputs[0], color_mix_node.inputs[6])
                color_mix_node.inputs[7].default_value = self.color_tint
                color_map_output = color_mix_node.outputs[2]
            else:
                color_map_output = color_map_node.outputs[0]
        else:
            rgb_node = self.create_node(Nodes.ShaderNodeRGB)
            rgb_node.outputs[0].default_value = self.color_tint
            color_map_output = rgb_node.outputs[0]
        self.connect_nodes(color_map_output, shader.inputs['Base Color'])

        normal_texture = self.normal_texture
        if normal_texture:
            normal_map_texture = self.create_texture_node(normal_texture, 'normal')
            uv_mapping = self.setup_uv_transform(self.normal_texcoord_offset1, self.normal_texcoord_scale1)
            if uv_mapping:
                self.connect_nodes(uv_mapping.outputs[0], normal_map_texture.inputs[0])
            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)
            self.connect_nodes(normal_map_texture.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])
            self.connect_nodes(normal_map_texture.outputs['Alpha'], shader.inputs['Roughness'])

        if self.flag_self_illum and self.selfillum_scale > 0:
            selfillum_mask = self.selfillum_mask_texture
            if is_blender_4():
                e_color_socket = shader.inputs["Emission Color"]
            else:
                e_color_socket = shader.inputs["Emission"]
            if self.selfillum_albedo_factor == 1.0:
                self.connect_nodes(color_map_output, e_color_socket)
            elif self.selfillum_albedo_factor > 0:
                mix_node = self.create_node(Nodes.ShaderNodeMix, "selfillum tint")
                mix_node.data_type = "RGBA"
                mix_node.blend_type = "MULTIPLY"
                self.connect_nodes(color_map_output, mix_node.inputs[6])
                mix_node.inputs[0].default_value = self.selfillum_albedo_factor
                mix_node.inputs[7].default_value = self.selfillum_color_tint
                self.connect_nodes(mix_node.outputs[2], e_color_socket)
            else:
                e_color_socket.default_value = self.selfillum_color_tint

            if selfillum_mask:
                selfillum_mask_node = self.create_texture_node(selfillum_mask, 'selfillum mask')
                uv_mapping = self.setup_uv_transform(self.selfillum_texcoord_offset1, self.selfillum_texcoord_scale1)
                if uv_mapping:
                    self.connect_nodes(uv_mapping.outputs[0], selfillum_mask_node.inputs[0])
                split_node = self.create_node(Nodes.ShaderNodeSeparateRGB)
                self.connect_nodes(selfillum_mask_node.outputs[0], split_node.inputs[0])
                if self.selfillum_scale == 1.0:
                    self.connect_nodes(split_node.outputs[0], shader.inputs["Emission Strength"])
                else:
                    math_node = self.create_node(Nodes.ShaderNodeMath)
                    math_node.operation = "MULTIPLY"
                    self.connect_nodes(split_node.outputs[0], math_node.inputs[0])
                    math_node.inputs[1].default_value = self.selfillum_scale
                    self.connect_nodes(math_node.outputs[0], shader.inputs["Emission Strength"])

        if self.tint_mask_displacement_texture:
            self.create_texture_node(self.tint_mask_displacement_texture, "tint_mask_displacement_texture")
