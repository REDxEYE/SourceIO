from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class Teeth(Source1ShaderBase):
    SHADER: str = 'teeth'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    def create_nodes(self, material:bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfDiffuse, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            self.create_and_connect_texture_node(basetexture, shader.inputs['Color'], name='$basetexture')