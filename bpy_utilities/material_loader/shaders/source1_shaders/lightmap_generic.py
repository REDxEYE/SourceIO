
from .detail import DetailSupportMixin
from ...shader_base import Nodes
from ..source1_shader_base import Source1ShaderBase


class LightmapGeneric(DetailSupportMixin):
    SHADER = 'lightmappedgeneric'

    @property
    def isskybox(self):
        return self._vavle_material.get_int('%' + 'compilesky', 0) + self._vavle_material.get_param(
            '%' + 'compile2Dsky', 0)

    @property
    def basetexture(self):
        texture_path = self._vavle_material.get_param('$basetexture', None)
        if texture_path is not None:
            return self.load_texture_or_default(texture_path, (0.3, 0, 0.3, 1.0))
        return None

    @property
    def basetexturetransform(self):
        return self._vavle_material.get_transform_matrix('$basetexturetransform', {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0), 'translate': (0, 0, 0)})

    @property
    def ssbump(self):
        return self._vavle_material.get_param('$ssbump', None)

    @property
    def bumpmap(self):
        texture_path = self._vavle_material.get_param('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            if self.ssbump:
                image = self.convert_ssbump(image)
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def bumptransform(self):
        return self._vavle_material.get_transform_matrix('$bumptransform', {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0), 'translate': (0, 0, 0)})

    @property
    def selfillummask(self):
        texture_path = self._vavle_material.get_param('$selfillummask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.0, 0.0, 0.0, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def color(self):
        color_value, value_type = self._vavle_material.get_vector('$color', None)
        if color_value is None:
            return None
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]
        return self.ensure_length(color_value, 4, 1.0)

    @property
    def translucent(self):
        return self._vavle_material.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vavle_material.get_int('$alphatest', 0) == 1

    @property
    def alphatestreference(self):
        return self._vavle_material.get_float('$alphatestreference', 0.5)

    @property
    def allowalphatocoverage(self):
        return self._vavle_material.get_int('$allowalphatocoverage', 0) == 1

    @property
    def phong(self):
        return self._vavle_material.get_int('$phong', 0) == 1

    @property
    def selfillum(self):
        return self._vavle_material.get_int('$selfillum', 0) == 1

    @property
    def basealphaenvmapmask(self):
        return self._vavle_material.get_int('$basealphaenvmapmask', 1) == 1

    @property
    def normalmapalphaenvmapmask(self):
        return self._vavle_material.get_int('$normalmapalphaenvmapmask', 0) == 1

    @property
    def envmap(self):
        return self._vavle_material.get_string('$envmap', None) is not None

    @property
    def envmapmask(self):
        texture_path = self._vavle_material.get_param('$envmapmask', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1, 1, 1, 1.0))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def envmaptint(self):
        color_value, value_type = self._vavle_material.get_vector('$envmaptint', [1, 1, 1])
        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]

        return self.ensure_length(color_value, 4, 1.0)

    def create_nodes(self, material_name):
        if super().create_nodes(material_name) in ['UNKNOWN', 'LOADED']:
            return

        if self.isskybox:
            self.bpy_material.shadow_method = 'NONE'
            self.bpy_material.use_backface_culling = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        if self.alphatest or self.translucent:
            if self.translucent:
                self.bpy_material.blend_method = 'BLEND'
            else:
                self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'

        if self.use_bvlg_status:
            if self.phong:
                #please use VertexLitGeneric instead
                pass

            self.do_arrange = False
            group_node = self.create_node_group("LightMappedGeneric", [-200, 0])
            parentnode = material_output
            if self.alphatest or self.translucent:
                alphatest_node = self.create_node_group("$alphatest", [250, 0])
                parentnode = alphatest_node
                material_output.location = [450, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                self.connect_nodes(alphatest_node.outputs['BSDF'], material_output.inputs['Surface'])

            self.connect_nodes(group_node.outputs['BSDF'], parentnode.inputs[0])
            if self.basetexture:
                basetexture_node = self.create_and_connect_texture_node(self.basetexture,
                                                                        group_node.inputs['$basetexture [texture]'],
                                                                        name='$basetexture')
                basetexture_node.location = [-800, 0]
                if self.basetexturetransform:
                    UV, self.UVmap = self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
                else:
                    UV = None
                albedo = basetexture_node.outputs['Color']
                if self.basealphaenvmapmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                if self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       alphatest_node.inputs['Alpha [basemap texture alpha]'])

                if self.detail:
                    albedo, detail = self.handle_detail(group_node.inputs['$basetexture [texture]'], albedo, UV=UV)

            if self.color:
                group_node.inputs['$color [RGB field]'].default_value = self.color or self.color2

            if self.envmap:
                group_node.inputs['$envmap [boolean]'].default_value = 1
                if self.envmaptint:
                    group_node.inputs['$envmaptint [RGB field]'].default_value = self.envmaptint

            if self.bumpmap:
                bumpmap_node = self.create_and_connect_texture_node(self.bumpmap,
                                                                    group_node.inputs['$bumpmap [texture]'],
                                                                    name='$bumpmap',
                                                                    UV=UV)
                if self.bumptransform:
                    self.handle_transform(self.bumptransform, bumpmap_node.inputs[0])
                bumpmap_node.location = [-800, -220]
                if self.normalmapalphaenvmapmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])

            if self.selfillum:
                group_node.inputs['$selfillum [bool]'].default_value = 1
                if self.selfillummask:
                    selfillummask_node = self.create_and_connect_texture_node(self.selfillummask, group_node.inputs[
                        '$selfillummask [texture alpha]'], UV=UV)
                    selfillummask_node.location = [-500, -510]
                elif self.basetexture is not None:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['$selfillummask [texture alpha]'])
        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

            basetexture = self.basetexture
            if basetexture:
                basetexture_node = self.create_and_connect_texture_node(basetexture,
                                                                        shader.inputs['Base Color'],
                                                                        name='$basetexture')
                if self.envmap:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Specular'])
                    shader.inputs['Roughness'].default_value = 0.2
                if self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])
                if self.translucent:
                    self.bpy_material.use_backface_culling = True
                    self.bpy_material.show_transparent_back = False
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

            if not self.phong:
                shader.inputs['Specular'].default_value = 0


class ReflectiveLightmapGeneric(LightmapGeneric):
    SHADER = 'lightmappedreflective'
