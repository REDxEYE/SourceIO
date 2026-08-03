from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_3


class HeroesArmor(Source1ShaderBase):
    """Shared implementation for the Dota 2 ``heroes_*`` shader family.

    ``heroes_faceskin`` uses it verbatim; ``heroes_pbs`` and ``heroes_hair`` only
    add one extra texture slot each via :attr:`EXTRA_TEXTURE`.
    """
    SHADER: str = 'heroes_armor'

    #: Optional ``(vmt_key, node_name)`` for a family-specific texture slot.
    EXTRA_TEXTURE: tuple[str, str] | None = None

    @property
    def bumpmap(self):
        return self._texture_property('$bumpmap', (0.5, 0.5, 1.0, 1.0), normal_map=True)

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0, 0.3, 1.0))

    @property
    def selfillummask(self):
        return self._texture_property('$selfillummask', (0.0, 0.0, 0.0, 1.0), is_data=True)

    @property
    def phongexponenttexture(self):
        return self._texture_property('$phongexponenttexture', (0.5, 0.0, 0.0, 1.0), is_data=True)

    @property
    def envmapmask(self):
        return self._texture_property('$envmapmask', (1, 1, 1, 1.0), is_data=True)

    @property
    def extra_texture(self):
        if self.EXTRA_TEXTURE is None:
            return None
        return self._texture_property(self.EXTRA_TEXTURE[0], (0.3, 0, 0.3, 1.0))

    @property
    def color2(self):
        return self._color_property('$color2')

    @property
    def color(self):
        return self._color_property('$color')

    @property
    def envmaptint(self):
        return self._color_property('$envmaptint', [1, 1, 1])

    @property
    def phongtint(self):
        return self._color_property('$phongtint')

    @property
    def phongfresnelranges(self):
        return self._color_property('$phongfresnelranges', length=3, filler=0.1)

    @property
    def translucent(self):
        return self._bool_property('$translucent')

    @property
    def alphatest(self):
        return self._bool_property('$alphatest')

    @property
    def allowalphatocoverage(self):
        return self._bool_property('$allowalphatocoverage')

    @property
    def additive(self):
        return self._bool_property('$additive')

    @property
    def phong(self):
        return self._bool_property('$phong_enable')

    @property
    def selfillum(self):
        return self._bool_property('$selfillum')

    @property
    def basealphaenvmapmask(self):
        return self._bool_property('$basealphaenvmapmask', 1)

    @property
    def basemapalphaphongmask(self):
        return self._bool_property('$basemapalphaphongmask')

    @property
    def normalmapalphaphongmask(self):
        return self._bool_property('$normalmapalphaphongmask', 1)

    @property
    def normalmapalphaenvmapmask(self):
        return self._bool_property('$normalmapalphaenvmapmask')

    @property
    def phongalbedotint(self):
        return self._bool_property('$phongalbedotint', 1)

    @property
    def envmap(self):
        return self._vmt.get_string('$envmap', None) is not None

    @property
    def alphatestreference(self):
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def phongexponent(self):
        return self._vmt.get_float('$phongexponent', None)

    @property
    def phongboost(self):
        return self._vmt.get_float('$phongboost', 1)

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        if self._vmt.get('proxies', None):
            proxies = self._vmt.get('proxies')
            for proxy_name, proxy_data in proxies.items():
                if proxy_name == 'selectfirstifnonzero':
                    result_var = proxy_data.get('resultvar')
                    src1_var = proxy_data.get('srcvar1')
                    src2_var = proxy_data.get('srcvar2')
                    src1_value, src1_type = self._vmt.get_vector(src1_var, [0])
                    if all([val > 0 for val in src1_value]):
                        self._vmt[result_var] = self._vmt[src1_var]
                    else:
                        self._vmt[result_var] = self._vmt[src2_var]

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0]
        parentnode = material_output

        if self.alphatest or self.translucent:
            if not is_blender_4_3():
                if self.translucent:
                    self.bpy_material.blend_method = 'BLEND'
                else:
                    self.bpy_material.blend_method = 'HASHED'
                self.bpy_material.shadow_method = 'HASHED'

        if self.use_bvlg_status:
            self.do_arrange = False
            if self.alphatest or self.translucent:
                alphatest_node = self.create_node_group("$alphatest", [250, 0])
                parentnode = alphatest_node
                material_output.location = [450, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                self.connect_nodes(alphatest_node.outputs['BSDF'], material_output.inputs['Surface'])

            group_node = self.create_node_group("VertexLitGeneric", [-200, 0])
            self.connect_nodes(group_node.outputs['BSDF'], parentnode.inputs[0])
            if self.basetexture:
                basetexture_node = self.create_and_connect_texture_node(self.basetexture,
                                                                        group_node.inputs['$basetexture [texture]'],
                                                                        name='$basetexture')
                basetexture_node.location = [-800, 0]
                if self.basealphaenvmapmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                if self.basemapalphaphongmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])
                if self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       alphatest_node.inputs['Alpha [basemap texture alpha]'])
            if self.color or self.color2:
                group_node.inputs['$color2 [RGB field]'].default_value = self.color or self.color2

            if self.envmap:
                group_node.inputs['$envmap [boolean]'].default_value = 1
                if self.envmaptint:
                    group_node.inputs['$envmaptint [RGB field]'].default_value = self.envmaptint

            if self.bumpmap:
                bumpmap_node = self.create_and_connect_texture_node(self.bumpmap,
                                                                    group_node.inputs['$bumpmap [texture]'],
                                                                    name='$bumpmap')
                bumpmap_node.location = [-800, -220]
                if self.normalmapalphaenvmapmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                elif self.normalmapalphaphongmask and not self.basemapalphaphongmask:
                    self.connect_nodes(bumpmap_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])

            if self.phong:
                group_node.inputs['$phong [bool]'].default_value = 1
                if self.phongboost:
                    group_node.inputs['$phongboost [value]'].default_value = self.phongboost
                if self.phongexponent:
                    group_node.inputs['$phongexponent [value]'].default_value = self.phongexponent
                elif self.phongexponenttexture:
                    phongexponent_group_node = self.create_node_group('$phongexponenttexture splitter', [-500, -300])
                    self.connect_nodes(phongexponent_group_node.outputs['$phongexponent [value]'],
                                       group_node.inputs['$phongexponent [value]'])
                    self.connect_nodes(phongexponent_group_node.outputs['rimlight mask'],
                                       group_node.inputs['rimlight mask'])
                    phongexponenttexture_node = self.create_and_connect_texture_node(self.phongexponenttexture,
                                                                                     phongexponent_group_node.inputs[
                                                                                         '$phongexponenttexture [texture]'],
                                                                                     phongexponent_group_node.inputs[
                                                                                         'alpha'],
                                                                                     name='$phongexponenttexture')
                    phongexponenttexture_node.location = [-800, -470]

                    if self.phongalbedotint is not None and not self.phongtint:
                        phongexponent_group_node.location = [-550, -300]
                        phongalbedo_node = self.create_node_group("$phongalbedotint", [-350, -345])
                        self.connect_nodes(phongexponent_group_node.outputs['phongalbedotint amount'],
                                           phongalbedo_node.inputs['phongalbedotint amount'])
                        self.connect_nodes(phongalbedo_node.outputs['$phongtint [RGB field]'],
                                           group_node.inputs['$phongtint [RGB field]'])
                        if self.basetexture is not None:
                            self.connect_nodes(basetexture_node.outputs['Color'],
                                               phongalbedo_node.inputs['$basetexture [texture]'])
                else:
                    group_node.inputs['$phongexponent [value]'].default_value = 10

                if self.phongtint is not None:
                    group_node.inputs['$phongtint [RGB field]'].default_value = self.phongtint

                if self.phongfresnelranges:
                    group_node.inputs['$phongfresnelranges [value field]'].default_value = self.phongfresnelranges

            if self.selfillum:
                group_node.inputs['$selfillum [bool]'].default_value = 1
                if self.selfillummask:
                    selfillummask_node = self.create_and_connect_texture_node(self.selfillummask, group_node.inputs[
                        '$selfillummask [texture alpha]'])
                    selfillummask_node.location = [-500, -510]
                elif self.basetexture is not None:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['$selfillummask [texture alpha]'])
        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

            basetexture = self.basetexture
            if basetexture:
                basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$basetexture')
                basetexture_node.image = basetexture
                basetexture_node.id_data.nodes.active = basetexture_node

                if self.color or self.color2:
                    color_mix = self.create_node(Nodes.ShaderNodeMixRGB)
                    color_mix.blend_type = 'MULTIPLY'
                    self.connect_nodes(basetexture_node.outputs['Color'], color_mix.inputs['Color1'])
                    color_mix.inputs['Color2'].default_value = (self.color or self.color2)
                    color_mix.inputs['Fac'].default_value = 1.0
                    self.connect_nodes(color_mix.outputs['Color'], shader.inputs['Base Color'])
                else:
                    self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
                if self.translucent or self.alphatest:
                    self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

                if self.additive:
                    basetexture_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                    basetexture_additive_mix_node = self.create_node(Nodes.ShaderNodeMixRGB)
                    self.insert_node(basetexture_node.outputs['Color'], basetexture_additive_mix_node.inputs['Color1'],
                                     basetexture_additive_mix_node.outputs['Color'])
                    basetexture_additive_mix_node.inputs['Color2'].default_value = (1.0, 1.0, 1.0, 1.0)

                    self.connect_nodes(basetexture_node.outputs['Color'], basetexture_invert_node.inputs['Color'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                       basetexture_additive_mix_node.inputs['Fac'])

            if self.EXTRA_TEXTURE is not None and (extra_texture := self.extra_texture):
                self.create_texture_node(extra_texture, self.EXTRA_TEXTURE[1])

            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.image = bumpmap

                normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

                self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
                self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

            if self.selfillum:
                selfillummask = self.selfillummask
                basetexture_node = self.get_node('$basetexture')
                if selfillummask is not None:
                    selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                    selfillummask_node.image = selfillummask
                    if 'Emission Strength' in shader.inputs:
                        self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])

                else:
                    if 'Emission Strength' in shader.inputs:
                        self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
                self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

            if not self.phong:
                shader.inputs['Specular'].default_value = 0
            elif self.phongboost is not None:
                shader.inputs['Specular'].default_value = self.clamp_value(self.phongboost / 64)
            phongexponenttexture = self.phongexponenttexture
            if self.phongexponent is not None and phongexponenttexture is None:
                shader.inputs['Roughness'].default_value = self.clamp_value(self.phongexponent / 256)
            elif self.phongexponenttexture is not None:
                phongexponenttexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$phongexponenttexture')
                phongexponenttexture_node.image = phongexponenttexture
                phongexponenttexture_split_node = self.create_node(Nodes.ShaderNodeSeparateColor)
                phongexponenttexture_split_node.mode = "RGB"
                self.connect_nodes(phongexponenttexture_node.outputs['Color'],
                                   phongexponenttexture_split_node.inputs['Image'])

                phongexponenttexture_r_invert_node = self.create_node(Nodes.ShaderNodeInvert)
                self.connect_nodes(phongexponenttexture_split_node.outputs['R'],
                                   phongexponenttexture_r_invert_node.inputs['Color'])
                self.connect_nodes(phongexponenttexture_split_node.outputs['G'],
                                   shader.inputs['Metallic'])

                self.connect_nodes(phongexponenttexture_r_invert_node.outputs['Color'], shader.inputs['Roughness'])
