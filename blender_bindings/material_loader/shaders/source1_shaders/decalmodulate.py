from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3


class DecalModulate(Source1ShaderBase):
    SHADER: str = 'decalmodulate'

    @property
    def basetexture(self):
        texture_path = self._vmt.get_string('$basetexture', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
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

    def create_nodes(self, material):
        if super().create_nodes(material) in ['UNKNOWN', 'LOADED']:
            return
        if not is_blender_4_3():
            self.bpy_material.blend_method = 'BLEND'
            self.bpy_material.shadow_method = 'NONE'

        self.bpy_material['DECAL'] = True

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfTransparent, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        print(basetexture)
        if basetexture:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
            basetexture_node.image = basetexture
            basetexture_node.id_data.nodes.active = basetexture_node

            scale = self.create_node(Nodes.ShaderNodeVectorMath)
            scale.operation = 'SCALE'
            scale.inputs[3].default_value = 2.0

            self.connect_nodes(basetexture_node.outputs[0], scale.inputs[0])
            self.connect_nodes(scale.outputs[0], shader.inputs['Color'])
