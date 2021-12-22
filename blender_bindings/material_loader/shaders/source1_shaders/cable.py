import math

from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class Cable(Source1ShaderBase):
    SHADER = 'cable'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    def create_nodes(self, material_name: str):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture,
                                                                    shader.inputs['Base Color'],
                                                                    name='$basetexture')

            tex_coord_node = self.create_node(Nodes.ShaderNodeTexCoord)
            tex_mapping_node = self.create_node(Nodes.ShaderNodeMapping)
            tex_mapping_node.inputs['Rotation'].default_value = (0, 0, math.radians(90))
            tex_mapping_node.inputs['Scale'].default_value = (25, 1, 1)

            self.connect_nodes(tex_coord_node.outputs['UV'], tex_mapping_node.inputs['Vector'])
            self.connect_nodes(tex_mapping_node.outputs['Vector'], basetexture_node.inputs['Vector'])


class SplineRope(Cable):
    SHADER = 'splinerope'
