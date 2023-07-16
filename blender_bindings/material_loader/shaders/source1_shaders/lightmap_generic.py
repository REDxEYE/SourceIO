from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class LightmapGeneric(Source1ShaderBase):
    SHADER = 'lightmappedgeneric'

    @property
    def isskybox(self):
        return self._vmt.get_int('%' + 'compilesky', 0) + self._vmt.get_int('%' + 'compile2Dsky', 0)

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
            image = self.load_texture_or_default(texture_path, (0.6, 0.0, 0.6, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def ssbump(self):
        return self._vmt.get_int('$ssbump', 0) == 1

    @property
    def phong(self):
        return self._vmt.get_int('$phong', 0) == 1

    @property
    def alpha(self):
        return self._vmt.get_int('$alpha', 0) == 1

    @property
    def alphatest(self):
        return self._vmt.get_int('$alphatest', 0) == 1

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        if self.isskybox:
            self.bpy_material.shadow_method = 'NONE'
            self.bpy_material.use_backface_culling = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        basetexture = self.basetexture

        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture,
                                                                    shader.inputs['Base Color'],
                                                                    name='$basetexture')

            if self.alphatest:
                self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'HASHED'
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])
            if self.translucent:
                self.bpy_material.blend_method = 'BLEND'
                self.bpy_material.shadow_method = 'HASHED'
                self.bpy_material.use_backface_culling = True
                self.bpy_material.show_transparent_back = False
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

        bumpmap = self.bumpmap
        if bumpmap and not self.ssbump:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            bumpmap_node.image = bumpmap

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

        if not self.phong:
            shader.inputs['Specular'].default_value = 0


class ReflectiveLightmapGeneric(LightmapGeneric):
    SHADER = 'lightmappedreflective'

class SDKLightmapGeneric(LightmapGeneric):
    SHADER = 'sdk_lightmappedgeneric'

