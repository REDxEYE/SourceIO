from typing import Tuple

from ..source2_shader_base import Source2ShaderBase
from ...shader_base import Nodes


class CSGOWeapon(Source2ShaderBase):
    SHADER: str = 'csgo_weapon.vfx'

    def _get_texture(self, slot_name: str, default_color: Tuple[float, float, float, float], is_data=False):
        texture_path = self._material_resource.get_texture_property(slot_name, None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, default_color)
            if is_data:
                image.colorspace_settings.is_data = True
                image.colorspace_settings.name = 'Non-Color'
        else:
            image = self.get_missing_texture(slot_name, default_color)
        texture_node = self.create_node(Nodes.ShaderNodeTexImage, slot_name)
        texture_node.image = image
        return texture_node

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        color_texture = self._get_texture("g_tColor", (1, 1, 1, 1))

        color_output = color_texture.outputs[0]

        self.connect_nodes(color_output, shader.inputs["Base Color"])

        normal_texture = self._get_texture("g_tNormal", (1, 1, 1, 1), True)
        normal_conv = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_texture.outputs[0], normal_conv.inputs[1])
        self.connect_nodes(normal_conv.outputs[0], shader.inputs["Normal"])
        self.connect_nodes(normal_texture.outputs[1], shader.inputs["Specular"])

        metalness_texture = self._get_texture("g_tMetalness", (1, 1, 1, 1), True)
        metalness_conv = self.create_node(Nodes.ShaderNodeSeparateRGB)
        self.connect_nodes(metalness_texture.outputs[0], metalness_conv.inputs[0])
        self.connect_nodes(metalness_conv.outputs[0], shader.inputs["Roughness"])
        metallic_invert = self.create_node(Nodes.ShaderNodeInvert)
        self.connect_nodes(metalness_conv.outputs[1], metallic_invert.inputs[1])

        self.connect_nodes(metallic_invert.outputs[0], shader.inputs["Metallic"])
