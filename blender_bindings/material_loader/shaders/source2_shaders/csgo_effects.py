from pprint import pformat

import bpy

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOEffects(Source2ShaderBase):
    SHADER: str = 'csgo_effects.vfx'

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node_group("csgo_effects.vfx", name=self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        material_data = self._material_resource
        data, = material_data.get_data_block(block_name='DATA')
        self.logger.info(pformat(dict(data)))

        if self._have_texture("g_tColor"):
            scale = material_data.get_vector_property("g_vTexCoordScale", None)
            offset = material_data.get_vector_property("g_vTexCoordOffset", None)
            center = material_data.get_vector_property("g_vTexCoordCenter", None)

            color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))
            if scale is not None or offset is not None or center is not None:
                uv_node = self.create_node(Nodes.ShaderNodeUVMap)
                uv_transform = self.create_node_group("UVTransform")
                if scale is not None:
                    uv_transform.inputs["g_vTexCoordScale"].default_value = scale[:3]
                if offset is not None:
                    uv_transform.inputs["g_vTexCoordOffset"].default_value = offset[:3]
                if center is not None:
                    uv_transform.inputs["g_vTexCoordCenter"].default_value = center[:3]

                self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])

                self.connect_nodes(uv_transform.outputs[0], color_texture.inputs[0])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["TextureColor"])
            alpha_output = color_texture.outputs[1]
        else:
            alpha_output = None

        if self._have_texture("g_tTintMask") and material_data.get_int_property("F_TINT_MASK", 0) == 1:
            tint_texture = self._get_texture("g_tTintMask", (1, 0, 0, 1), True, True)
            self.connect_nodes(tint_texture.outputs[0], shader.inputs["TextureTintMask"])

        tint = material_data.get_vector_property("g_vColorTint", None)
        if tint is not None and (tint[0] != 1.0 or tint[1] != 1.0 or tint[2] != 1.0):
            shader.inputs["GlobalTint"].default_value = tint

        uv_node = self.create_node(Nodes.ShaderNodeUVMap)

        value_node = self.create_node(Nodes.ShaderNodeValue)
        driver = value_node.outputs[0].driver_add("default_value").driver

        # Set driver type
        driver.type = 'SCRIPTED'

        # The driver expression: frame divided by fps
        driver.expression = "current_frame / fps"

        # Add the current frame variable
        frame_var = driver.variables.new()
        frame_var.name = "current_frame"
        frame_var.type = 'SINGLE_PROP'
        frame_var.targets[0].id_type = 'SCENE'
        frame_var.targets[0].id = bpy.context.scene
        frame_var.targets[0].data_path = "frame_current"

        # Add the fps variable
        fps_var = driver.variables.new()
        fps_var.name = "fps"
        fps_var.type = 'SINGLE_PROP'
        fps_var.targets[0].id_type = 'SCENE'
        fps_var.targets[0].id = bpy.context.scene
        fps_var.targets[0].data_path = "render.fps"

        if self._have_texture("g_tMask1"):
            color_texture = self._get_texture("g_tMask1", (1, 1, 1, 1))
            uv_transform = self.create_node_group("UVPan")
            uv_transform.inputs["PanSpeed"].default_value = material_data.get_vector_property("g_vMask1PanSpeed",
                                                                                              (0, 0, 0))[:3]
            uv_transform.inputs["g_vTexCoordScale"].default_value = material_data.get_vector_property("g_vMask1Scale",
                                                                                                      (1, 1, 1))[:3]
            self.connect_nodes(value_node.outputs[0], uv_transform.inputs["Time"])
            self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])
            self.connect_nodes(uv_transform.outputs[0], color_texture.inputs[0])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["Mask1"])

        if self._have_texture("g_tMask2"):
            color_texture = self._get_texture("g_tMask2", (1, 1, 1, 1))
            uv_transform = self.create_node_group("UVPan")
            uv_transform.inputs["PanSpeed"].default_value = material_data.get_vector_property("g_vMask2PanSpeed",
                                                                                              (0, 0, 0))[:3]
            uv_transform.inputs["g_vTexCoordScale"].default_value = material_data.get_vector_property("g_vMask2Scale",
                                                                                                      (1, 1, 1))[:3]
            self.connect_nodes(value_node.outputs[0], uv_transform.inputs["Time"])
            self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])
            self.connect_nodes(uv_transform.outputs[0], color_texture.inputs[0])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["Mask2"])

        if self._have_texture("g_tMask3"):
            color_texture = self._get_texture("g_tMask3", (1, 1, 1, 1))
            uv_transform = self.create_node_group("UVPan")
            uv_transform.inputs["PanSpeed"].default_value = material_data.get_vector_property("g_vMask3PanSpeed",
                                                                                              (0, 0, 0))[:3]
            uv_transform.inputs["g_vTexCoordScale"].default_value = material_data.get_vector_property("g_vMask3Scale",
                                                                                                      (1, 1, 1))[:3]
            self.connect_nodes(value_node.outputs[0], uv_transform.inputs["Time"])
            self.connect_nodes(uv_node.outputs[0], uv_transform.inputs[0])
            self.connect_nodes(uv_transform.outputs[0], color_texture.inputs[0])

            self.connect_nodes(color_texture.outputs[0], shader.inputs["Mask3"])

        if self.tinted:
            vcolor_node = self.create_node(Nodes.ShaderNodeVertexColor)
            vcolor_node.layer_name = "TINT"
            self.connect_nodes(vcolor_node.outputs[0], shader.inputs["ModelTint"])

        if material_data.get_int_property("F_ALPHA_TEST", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'CLIP'
            self.bpy_material.shadow_method = 'CLIP'
            self.bpy_material.alpha_threshold = material_data.get_float_property("g_flAlphaTestReference", 0.5)
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
        elif material_data.get_int_property("S_TRANSLUCENT", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
        elif material_data.get_int_property("F_ADDITIVE_BLEND", 0) and alpha_output is not None:
            self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'CLIP'
            self.connect_nodes(alpha_output, shader.inputs["Alpha"])
