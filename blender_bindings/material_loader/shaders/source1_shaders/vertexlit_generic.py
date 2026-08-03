import math
from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3
from .detail import DetailSupportMixin


def phong_exponent_to_roughness(exponent: float) -> float:
    """Convert a Source Phong exponent to a Blender GGX roughness.

    Source evaluates ``pow(saturate(dot(R, L)), $phongexponent)``. The standard
    Blinn-Phong -> microfacet-roughness mapping is ``sqrt(2 / (exponent + 2))``,
    which is what Blender's GGX lobe expects.
    """
    return min(1.0, max(0.0, math.sqrt(2.0 / (max(exponent, 0.0) + 2.0))))


class VertexLitGeneric(DetailSupportMixin, Source1ShaderBase):
    SHADER: str = 'vertexlitgeneric'

    @property
    def bumpmap(self):
        texture_path = self._vmt.get_string('$bumpmap', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            if image == self.basetexture:
                return image
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def compress(self):
        texture_path = self._vmt.get_string('$compress', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            if image == self.basetexture:
                return image
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def stretch(self):
        texture_path = self._vmt.get_string('$stretch', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (0.5, 0.5, 1.0, 1.0))
            if image == self.basetexture:
                return image
            image = self.convert_normalmap(image)
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0, 0.3, 1.0))

    @property
    def lightwarptexture(self):
        return self._texture_property('$lightwarptexture', (0.3, 0, 0.3, 1.0), is_data=True)

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
                                               'translate': (0, 0, 0)})

    @property
    def selfillummask(self):
        return self._texture_property('$selfillummask', (0.0, 0.0, 0.0, 1.0), is_data=True)

    @property
    def phongexponenttexture(self):
        return self._texture_property('$phongexponenttexture', (0.5, 0.0, 0.0, 1.0), is_data=True)

    @property
    def color2(self):
        return self._color_property('$color2')

    @property
    def color(self):
        return self._color_property('$color')

    @property
    def colortint_base(self):
        return self._color_property('$colortint_base')

    @property
    def translucent(self):
        return self._vmt.get_int('$translucent', 0) == 1

    @property
    def alphatest(self):
        return self._vmt.get_int('$alphatest', 0) == 1

    @property
    def alphatestreference(self):
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def selfillummaskscale(self):
        return self._vmt.get_float('$selfillummaskscale', 1.0)

    @property
    def allowalphatocoverage(self):
        return self._vmt.get_int('$allowalphatocoverage', 0) == 1

    @property
    def additive(self):
        return self._vmt.get_int('$additive', 0) == 1

    @property
    def phong(self):
        return self._vmt.get_int('$phong', 0) == 1

    @property
    def selfillum(self):
        return self._vmt.get_int('$selfillum', 0) == 1

    @property
    def selfillumtint(self):
        return self._color_property('$selfillumtint')

    @property
    def basealphaenvmapmask(self):
        return self._vmt.get_int('$basealphaenvmapmask', 1) == 1

    @property
    def basemapalphaphongmask(self):
        return self._vmt.get_int('$basemapalphaphongmask', 0) == 1

    @property
    def normalmapalphaphongmask(self):
        return self._vmt.get_int('$normalmapalphaphongmask', 1) == 1

    @property
    def normalmapalphaenvmapmask(self):
        return self._vmt.get_int('$normalmapalphaenvmapmask', 0) == 1

    @property
    def envmap(self):
        return self._vmt.get_string('$envmap', None) is not None

    @property
    def envmapmask(self):
        return self._texture_property('$envmapmask', (1, 1, 1, 1.0), is_data=True)

    @property
    def envmaptint(self):
        return self._color_property('$envmaptint', [1, 1, 1])

    @property
    def phongfresnelranges(self):
        return self._color_property('$phongfresnelranges', length=3, filler=0.1)

    @property
    def phongexponent(self):
        value = self._vmt.get_float('$phongexponent', None)
        return value

    @property
    def phongboost(self):
        value = self._vmt.get_float('$phongboost', 1)
        return value

    @property
    def rimlight(self):
        value = self._vmt.get_int('$rimlight', 0)
        return value

    @property
    def rimlightexponent(self):
        value = self._vmt.get_int('$rimlightexponent', 0)
        return value

    @property
    def rimlightboost(self):
        value = self._vmt.get_float('$rimlightboost', 0)
        return value

    @property
    def phongalbedotint(self):
        return self._vmt.get_int('$phongalbedotint', 1) == 1

    @property
    def blendtintbybasealpha(self):
        return self._vmt.get_float('$blendtintbybasealpha', 0)

    @property
    def blendtintcoloroverbase(self):
        return self._vmt.get_float('$blendtintcoloroverbase', 0)

    @property
    def nocull(self):
        return self._vmt.get_int('$nocull', 0) == 1

    @property
    def phongtint(self):
        return self._color_property('$phongtint')

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        sio_diffuse = None
        selfillum = False

        # print(f"BVLG: {self.use_bvlg_status}")

        if 'proxies' in self._vmt:
            proxies = self._vmt.get('proxies')
            for proxy_name, proxy_data in proxies.items():
                if proxy_name == 'selectfirstifnonzero':
                    result_var = proxy_data.get('resultvar')
                    if result_var not in self._vmt:
                        continue
                    src1_var = proxy_data.get('srcvar1')
                    src2_var = proxy_data.get('srcvar2')
                    src1_value, src1_type = self._vmt.get_vector(src1_var, [0])
                    if all([val > 0 for val in src1_value]):
                        if src1_var not in self._vmt:
                            continue
                        self._vmt[result_var] = self._vmt[src1_var]
                    else:
                        if src2_var not in self._vmt:
                            continue
                        self._vmt[result_var] = self._vmt[src2_var]

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        material_output.location = [250, 0]
        if not is_blender_4_3() and (self.alphatest or self.translucent):
            if self.translucent:
                self.bpy_material.blend_method = 'BLEND'
                if hasattr(self.bpy_material, 'surface_render_method'):
                    self.bpy_material.surface_render_method = 'BLENDED'
            else:
                self.bpy_material.blend_method = 'HASHED'
            self.bpy_material.shadow_method = 'HASHED'
        if self.additive:
            self.bpy_material.blend_method = 'BLEND'
            if hasattr(self.bpy_material, 'surface_render_method'):
                self.bpy_material.surface_render_method = 'BLENDED'
        uv = None

        compress = self.compress
        if compress:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$compress')
            basetexture_node.image = compress

        stretch = self.stretch
        if stretch:
            basetexture_node = self.create_node(Nodes.ShaderNodeTexImage, '$stretch')
            basetexture_node.image = stretch

        basetexture = self.basetexture
        if self.use_bvlg_status:
            self.do_arrange = True

            self.bpy_material.use_backface_culling = not self.nocull

            group_node = self.create_node_group("VertexLitGeneric", [-200, 0])

            # so we can toggle backface culling across eevee and cycles with one boolean property
            driv = group_node.inputs['$nocull'].driver_add('default_value')
            driv.driver.expression = '1-var'
            var = driv.driver.variables.new()
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'MATERIAL'
            var.targets[0].id = self.bpy_material
            var.targets[0].data_path = 'use_backface_culling'

            final = group_node.outputs[0]

            if basetexture:
                basetexture_node = self.create_and_connect_texture_node(basetexture,
                                                                        group_node.inputs['$basetexture [texture]'],
                                                                        name='$basetexture')
                basetexture_node.location = [-800, 0]
                if self.basetexturetransform:
                    uv, self.uv_map = self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
                albedo = basetexture_node.outputs['Color']
                if self.basealphaenvmapmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['envmapmask [basemap texture alpha]'])
                if self.basemapalphaphongmask:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['phongmask [bumpmap texture alpha]'])
                # if self.alphatest or self.translucent:
                #    self.connect_nodes(basetexture_node.outputs['Alpha'],
                #                       alphatest_node.inputs['Alpha [basemap texture alpha]'])

                if self.detail:
                    albedo, detail = self.handle_detail(group_node.inputs['$basetexture [texture]'], albedo, uv_node=uv)
            elif self.color:
                group_node.inputs['$basetexture [texture]'].default_value = self.color

            if self.color2 or self.colortint_base:
                group_node.inputs['$color2 [RGB field]'].default_value = self.color2 or self.colortint_base

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

            if self.rimlight:
                group_node.inputs['$rimlight [bool]'].default_value = self.rimlight
            if self.rimlightexponent:
                group_node.inputs['$rimlightexponent [value]'].default_value = self.rimlightexponent
            if self.rimlightboost:
                group_node.inputs['$rimlightboost [value]'].default_value = self.rimlightboost

            if self.blendtintbybasealpha:
                group_node.inputs['$blendtintbybasealpha'].default_value = self.blendtintbybasealpha

            if self.blendtintcoloroverbase:
                group_node.inputs['$blendtintcoloroverbase'].default_value = self.blendtintcoloroverbase

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
                                                                                     name='$phongexponenttexture',
                                                                                     uv_out=uv.output[0])
                    phongexponenttexture_node.location = [-800, -470]

                    if self.phongalbedotint is not None and not self.phongtint:
                        phongexponent_group_node.location = [-550, -300]
                        phongalbedo_node = self.create_node_group("$phongalbedotint", [-350, -345])
                        self.connect_nodes(phongexponent_group_node.outputs['phongalbedotint amount'],
                                           phongalbedo_node.inputs['phongalbedotint amount'])
                        self.connect_nodes(phongalbedo_node.outputs['$phongtint [RGB field]'],
                                           group_node.inputs['$phongtint [RGB field]'])
                        if basetexture is not None:
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
                        '$selfillummask [texture alpha]'], uv_out=uv.output[0])
                    selfillummask_node.location = [-500, -510]
                elif basetexture is not None:
                    self.connect_nodes(basetexture_node.outputs['Alpha'],
                                       group_node.inputs['$selfillummask [texture alpha]'])
                if self.selfillummaskscale:
                    group_node.inputs['$selfillummaskscale'].default_value = self.selfillummaskscale
                if self.selfillumtint:
                    group_node.inputs['$selfillumtint'].default_value = self.selfillumtint

            if self.lightwarptexture:
                material_output.target = 'EEVEE'
                cycles_output = self.create_node(Nodes.ShaderNodeOutputMaterial, name='Cycles Output')
                cycles_output.target = 'CYCLES'

                lightwarp = self.create_texture_node(self.lightwarptexture, 'lightwarptexture')
                lightwarp.extension = 'EXTEND'
                self.connect_nodes(group_node.outputs['lightwarpvector'], lightwarp.inputs[0])
                sio_diffuse = self.create_node_group('SIO-Diffuse', name='Diffuse')
                self.connect_nodes(lightwarp.outputs[0], sio_diffuse.inputs['Lightwarp'])
                self.connect_nodes(group_node.outputs['$bumpmap'], sio_diffuse.inputs['$bumpmap'])
                self.connect_nodes(group_node.outputs['Final Albedo'], sio_diffuse.inputs['Albedo'])
                self.connect_nodes(group_node.outputs['Lighting'], sio_diffuse.inputs['Light Pass'])

                # so we can toggle backface culling across eevee and cycles with one boolean property
                driv = sio_diffuse.inputs['$nocull'].driver_add('default_value')
                driv.driver.expression = '1-var'
                var = driv.driver.variables.new()
                var.type = 'SINGLE_PROP'
                var.targets[0].id_type = 'MATERIAL'
                var.targets[0].id = self.bpy_material
                var.targets[0].data_path = 'use_backface_culling'

                final = sio_diffuse.outputs['EEVEE Pass']
                final_cycles = sio_diffuse.outputs['Cycles Pass']

            if self.additive:
                if sio_diffuse:
                    add_eevee = self.create_node(Nodes.ShaderNodeAddShader)
                    add_cycles = self.create_node(Nodes.ShaderNodeAddShader)
                    transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                    self.connect_nodes(sio_diffuse.outputs['EEVEE Pass'], add_eevee.inputs[1])
                    self.connect_nodes(sio_diffuse.outputs['Cycles Pass'], add_cycles.inputs[1])
                    self.connect_nodes(transparent.outputs[0], add_eevee.inputs[0])
                    self.connect_nodes(transparent.outputs[0], add_cycles.inputs[0])
                    final = add_eevee.outputs[0]
                    final_cycles = add_cycles.outputs[0]
                else:
                    add = self.create_node(Nodes.ShaderNodeAddShader)
                    transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
                    self.connect_nodes(final, add.inputs[1])
                    self.connect_nodes(transparent.outputs[0], add.inputs[0])
                    final = add.outputs[0]

            if self.alphatest or self.translucent:
                alphatest_node = self.create_node_group("$alphatest", [250, 0])
                material_output.location = [450, 0]
                alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                alphatest_node.inputs['$translucent'].default_value = self.translucent
                if basetexture:
                    self.connect_nodes(basetexture_node.outputs[1],
                                       alphatest_node.inputs['Alpha [basemap texture alpha]'])
                self.connect_nodes(final, alphatest_node.inputs[0])
                final = alphatest_node.outputs[0]
                if sio_diffuse:
                    alphatest_node = self.create_node_group("$alphatest", [250, 0])
                    material_output.location = [450, 0]
                    alphatest_node.inputs['$alphatestreference [value]'].default_value = self.alphatestreference
                    alphatest_node.inputs['$allowalphatocoverage [boolean]'].default_value = self.allowalphatocoverage
                    alphatest_node.inputs['$translucent'].default_value = self.translucent
                    if basetexture:
                        self.connect_nodes(basetexture_node.outputs[1],
                                           alphatest_node.inputs['Alpha [basemap texture alpha]'])
                    self.connect_nodes(final_cycles, alphatest_node.inputs[0])
                    final_cycles = alphatest_node.outputs[0]

            # Finalize outputs
            self.connect_nodes(final, material_output.inputs[0])
            if sio_diffuse:
                self.connect_nodes(final_cycles, cycles_output.inputs[0])

            '''
            END OF BVLG
            '''

        else:
            shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
            self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

            basetexture = basetexture
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
                    if is_blender_4():
                        self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                           shader.inputs['Transmission Weight'])
                    else:
                        self.connect_nodes(basetexture_invert_node.outputs['Color'], shader.inputs['Transmission'])
                    self.connect_nodes(basetexture_invert_node.outputs['Color'],
                                       basetexture_additive_mix_node.inputs['Fac'])

            bumpmap = self.bumpmap
            if bumpmap:
                bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
                bumpmap_node.image = bumpmap

                normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

                self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
                self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

            if self.selfillum and basetexture:
                basetexture_node = self.get_node('$basetexture')
                if basetexture_node:
                    selfillummask = self.selfillummask
                    if selfillummask is not None:
                        selfillummask_node = self.create_node(Nodes.ShaderNodeTexImage, '$selfillummask')
                        selfillummask_node.image = selfillummask
                        if 'Emission Strength' in shader.inputs:
                            self.connect_nodes(selfillummask_node.outputs['Color'], shader.inputs['Emission Strength'])
                    else:
                        if 'Emission Strength' in shader.inputs:
                            self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])
                    if is_blender_4():
                        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission Color'])
                    else:
                        self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Emission'])

            self._setup_phong(shader, self.get_node('$basetexture'))

    def _setup_phong(self, shader, basetexture_node):
        """Approximate Source's Phong specular on a Principled BSDF.

        Mirrors ``skin_ps20b.fxc`` / ``common_vertexlitgeneric_dx9.h`` from the
        Source SDK 2013, which computes::

            fSpecExp = (const >= 0) ? const : (1.0 + 149.0 * vSpecExpMap.r)
            specularLighting = pow( saturate(dot(vReflect, vLightDir)), fSpecExp )
            fSpecMask = lerp( normalTexel.a, baseColor.a, $basemapalphaphongmask )
            fSpecMask *= fFresnelRanges
            specularLighting *= fSpecMask * $phongboost
            vSpecularTint = lerp( 1, baseColor.rgb, vSpecExpMap.g )
            vSpecularTint = ($phongtint.r >= 0) ? $phongtint : vSpecularTint
            result = specularLighting * vSpecularTint + ...

        Blender has no Phong lobe and its specular inputs are 0..1 factors, so:

        * exponent -> Roughness via :func:`phong_exponent_to_roughness`
        * mask x fresnel x boost -> Specular IOR Level / Roughness modulation
        * tint -> Specular Tint (same lerp as the SDK)
        """
        spec_input = 'Specular IOR Level' if is_blender_4() else 'Specular'

        if not self.phong:
            shader.inputs[spec_input].default_value = 0.0
            return

        # --- Specular strength: $phongboost x fresnel ---
        # In the SDK, boost is an unclamped multiplier into an HDR accumulator
        # (`specularLighting *= fSpecMask * g_SpecularBoost`), so values in the
        # hundreds are legitimate. Blender's Specular IOR Level is a hard 0..1
        # factor (0.5 == IOR 1.45), so map the boost perceptually: 1.0 stays
        # neutral at 0.5 and the practical 1..255 range spans 0.5..1.0 instead of
        # saturating almost immediately.
        boost = self.phongboost if self.phongboost is not None else 1.0
        boost = max(boost, 0.0)
        if boost <= 1.0:
            strength = 0.5 * boost
        else:
            # log-scaled so boost 1 -> 0.5, 16 -> ~0.75, 255 -> 1.0
            strength = 0.5 + 0.5 * min(1.0, math.log(boost, 255.0))

        fresnel_ranges = self.phongfresnelranges
        if fresnel_ranges and len(fresnel_ranges) > 1:
            # vRanges = (low, mid, high); the SDK's piecewise curve is centred on
            # `mid`, which is the value seen across most of the surface, so use it
            # as the constant stand-in for the per-pixel fresnel term.
            strength *= max(fresnel_ranges[1], 0.0)
        shader.inputs[spec_input].default_value = self.clamp_value(strength)

        # --- Roughness from the exponent (scalar or texture red channel) ---
        exponent_texture = self.phongexponenttexture
        roughness_output = None
        albedo_tint_amount = None  # socket carrying the per-pixel albedo-tint mask
        rim_mask_output = None  # socket carrying the per-pixel rimlight mask
        if exponent_texture is not None:
            exponent_node = self.create_texture_node(exponent_texture, '$phongexponenttexture')
            split_node = self.create_node(Nodes.ShaderNodeSeparateColor, 'phongexponent split')
            split_node.mode = "RGB"
            self.connect_nodes(exponent_node.outputs['Color'], split_node.inputs[0])

            # The SDK decodes the red channel as `fSpecExp = 1.0 + factor * r`
            # (factor is 149.0 unless $phongexponentfactor overrides it), so r=0
            # is a very broad highlight and r=1 is tight. Map Range is linear in
            # `r`, which cannot reproduce the sqrt() curve exactly, but matching
            # both endpoints keeps the highlight size right at the extremes.
            exponent_factor = self._vmt.get_float('$phongexponentfactor', None) or 149.0
            remap = self.create_node(Nodes.ShaderNodeMapRange, 'phongexponent -> roughness')
            remap.inputs['From Min'].default_value = 0.0
            remap.inputs['From Max'].default_value = 1.0
            remap.inputs['To Min'].default_value = phong_exponent_to_roughness(1.0)
            remap.inputs['To Max'].default_value = phong_exponent_to_roughness(1.0 + exponent_factor)
            self.connect_nodes(split_node.outputs[0], remap.inputs['Value'])
            roughness_output = remap.outputs[0]

            # G is the per-pixel $phongalbedotint amount -- NOT metalness, which is
            # what this used to be wired to.
            albedo_tint_amount = split_node.outputs[1]
            # A is the rim mask: `fRimMask = lerp(1.0, vSpecExpMap.a, $rimmask)`.
            rim_mask_output = exponent_node.outputs['Alpha']
        elif self.phongexponent is not None:
            shader.inputs['Roughness'].default_value = phong_exponent_to_roughness(self.phongexponent)
        else:
            # No exponent map and no explicit $phongexponent. The SDK falls back
            # to `max(g_EyePos_SpecExponent.w, 0)`; VertexLitGeneric's own node
            # group uses 12 as its neutral default, which reads well in practice.
            shader.inputs['Roughness'].default_value = phong_exponent_to_roughness(12.0)

        # --- Specular tint, resolved once ---
        # Source precedence: an explicit $phongtint wins; otherwise, when
        # $phongalbedotint is set the specular tints toward the albedo, by the
        # per-pixel amount in the exponent texture's green channel if present.
        if 'Specular Tint' in shader.inputs:
            phongtint = self.phongtint
            if phongtint is not None:
                shader.inputs['Specular Tint'].default_value = phongtint
            elif self.phongalbedotint and basetexture_node is not None and albedo_tint_amount is not None:
                # Only tint per-pixel, driven by the exponent texture's green
                # channel. Without that channel Source has no tint amount to
                # apply, and forcing full-albedo tint here reads far too strong.
                albedo_tint_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'phongalbedotint')
                albedo_tint_mix.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)
                self.connect_nodes(albedo_tint_amount, albedo_tint_mix.inputs['Fac'])
                self.connect_nodes(basetexture_node.outputs['Color'], albedo_tint_mix.inputs['Color2'])
                self.connect_nodes(albedo_tint_mix.outputs['Color'], shader.inputs['Specular Tint'])

        # --- Phong mask ---
        # SDK: `fSpecMask = lerp( normalTexel.a, baseColor.a, $basemapalphaphongmask )`
        # i.e. normal-map alpha by default, base-map alpha when the flag is set.
        mask_output = None
        if self.basemapalphaphongmask and basetexture_node is not None:
            mask_output = basetexture_node.outputs['Alpha']
        elif self.normalmapalphaphongmask:
            bumpmap_node = self.get_node('$bumpmap')
            if bumpmap_node is not None:
                mask_output = bumpmap_node.outputs['Alpha']

        if mask_output is not None:
            # The SDK scales the specular *intensity* by the mask. Blender's
            # Specular IOR Level is already set to a constant, so approximate the
            # same effect by driving masked-out areas fully rough instead:
            # Mix(Fac=mask, A=1.0, B=roughness)
            mask_mix = self.create_node(Nodes.ShaderNodeMix, 'phongmask')
            mask_mix.data_type = 'FLOAT'
            self.connect_nodes(mask_output, mask_mix.inputs['Factor'])
            mask_mix.inputs['A'].default_value = 1.0
            if roughness_output is not None:
                self.connect_nodes(roughness_output, mask_mix.inputs['B'])
            else:
                mask_mix.inputs['B'].default_value = shader.inputs['Roughness'].default_value
            roughness_output = mask_mix.outputs[0]

        if roughness_output is not None:
            self.connect_nodes(roughness_output, shader.inputs['Roughness'])

        # --- Rimlight ---
        # SDK: `rimLighting = pow(LdotR, g_RimExponent)` masked by N.L and scaled
        # by $rimlightboost, with the exponent texture's alpha as a per-pixel mask.
        # Sheen is the closest Principled analogue for a broad edge-facing term.
        if self.rimlight and 'Sheen Weight' in shader.inputs:
            boost = self.rimlightboost
            weight = self.clamp_value(boost if boost else 1.0)
            if rim_mask_output is not None:
                rim_mix = self.create_node(Nodes.ShaderNodeMix, 'rimlight mask')
                rim_mix.data_type = 'FLOAT'
                self.connect_nodes(rim_mask_output, rim_mix.inputs['Factor'])
                rim_mix.inputs['A'].default_value = 0.0
                rim_mix.inputs['B'].default_value = weight
                self.connect_nodes(rim_mix.outputs[0], shader.inputs['Sheen Weight'])
            else:
                shader.inputs['Sheen Weight'].default_value = weight
            if 'Sheen Roughness' in shader.inputs:
                shader.inputs['Sheen Roughness'].default_value = phong_exponent_to_roughness(
                    self.rimlightexponent or 4.0)


class SDKVertexLitGeneric(VertexLitGeneric):
    SHADER: str = 'sdk_vertexlitgeneric'
