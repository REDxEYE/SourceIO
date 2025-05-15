from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import ExtraMaterialParameters, Nodes
from SourceIO.blender_bindings.material_loader.shaders.source2_shader_base import Source2ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3


class CSGOEnvironment(Source2ShaderBase):
    SHADER: str = 'csgo_environment.vfx'

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader_node = self.create_node(Nodes.ShaderNodeBsdfPrincipled)
        self.connect_nodes(shader_node.outputs['BSDF'], material_output.inputs['Surface'])

        material_data = self._material_resource

        center = material_data.get_vector_property("g_vTexCoordCenter1", (0.5, 0.5, 0.0))
        offset = material_data.get_vector_property("g_vTexCoordOffset1", (0.0, 0.0, 0.0))
        scale = material_data.get_vector_property("g_vTexCoordScale1", (1.0, 1.0, 0.0))
        transform_node = self.create_transform("TEXCOORD", scale, offset, center)

        albedo_node = self._get_texture("g_tColor1", (1, 1, 1, 1))
        normal_node = self._get_texture("g_tNormal1", (0.5, 0.5, 1, 1), True)

        self.connect_nodes(transform_node.outputs[0], albedo_node.inputs[0])
        self.connect_nodes(transform_node.outputs[0], normal_node.inputs[0])

        normal_convert_node = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_node.outputs[0], normal_convert_node.inputs['Color'])
        self.connect_nodes(normal_convert_node.outputs[0], shader_node.inputs['Normal'])
        self.connect_nodes(normal_node.outputs[1], shader_node.inputs['Roughness'])

        self.connect_nodes(albedo_node.outputs['Color'], shader_node.inputs['Base Color'])

        if material_data.get_int_property("F_ALPHA_TEST", 0):
            alpha_test_reference = material_data.get_float_property("g_flAlphaTestReference", 0.5)
            self._handle_alpha_modes("TEST", alpha_test_reference,
                                     albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])
        elif material_data.get_int_property("S_TRANSLUCENT", 0):
            self._handle_alpha_modes("TRANSLUCENT", 0.5,
                                     albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])
        elif material_data.get_int_property("F_OVERLAY", 0):
            self._handle_alpha_modes("OVERLAY", 0.5,
                                     albedo_node.outputs['Alpha'], shader_node.inputs['Alpha'])
