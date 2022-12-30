from typing import Iterable

import numpy as np

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class DecalModulate(Source1ShaderBase):
    SHADER: str = 'decalmodulate'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def decalscale(self):
        return self._vmt.get_float('$decalscale', 0)

    @property
    def decal(self):
        return self._vmt.get_int('$decal', 0)

    @property
    def vertexcolor(self):
        return self._vmt.get_int('$vertexcolor', 0)

    @property
    def vertexalpha(self):
        return self._vmt.get_int('$vertexalpha', 0)

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture
            basetexture_node.id_data.nodes.active = basetexture_node

            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
