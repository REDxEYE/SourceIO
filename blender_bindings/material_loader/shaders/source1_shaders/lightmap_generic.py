from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3
from .detail import DetailSupportMixin, TCOMBINE_MOD2X_SELECT_TWO_PATTERNS


class LightmapGeneric(DetailSupportMixin, Source1ShaderBase):
    """LightmappedGeneric -- the world/brush geometry shader.

    This class implements the whole ``LightmappedGeneric_DX9`` family. In the SDK,
    ``worldvertextransition.cpp`` is not a separate implementation at all: it fills
    a ``LightmappedGeneric_DX9_Vars_t`` and calls straight into
    ``InitParamsLightmappedGeneric_DX9`` / ``InitLightmappedGeneric_DX9`` /
    ``DrawLightmappedGeneric_DX9``. Both shaders therefore support the same
    features -- including the two-way ``$basetexture2`` blend -- so both live here
    and :class:`WorldVertexTransition` only changes the shader name.

    Unlike VertexLitGeneric there is **no Phong**; specular comes from a cubemap
    (``lightmappedgeneric_ps2_3_x.h``)::

        fresnel = pow( 1.0 - dot( worldSpaceNormal, eyeVect ), 5.0 );
        fresnel = fresnel * g_OneMinusFresnelReflection + g_FresnelReflection;

        specularFactor = 1.0
        if ( $normalmapalphaenvmapmask ) specularFactor *= vNormal.a;
        if ( $envmapmask )               specularFactor *= tex2D( EnvmapMaskSampler, ... ).xyz;
        if ( $basealphaenvmapmask )      specularFactor *= 1.0 - blendedAlpha;

        specularLighting  = ENV_MAP_SCALE * texCUBE( EnvmapSampler, reflectVect );
        specularLighting *= specularFactor * g_EnvmapTint * fresnel;

    and the two-way blend is::

        float blendfactor = i.vertexBlendX_fogFactorW.r;      // vertex alpha
        if ( bBaseTexture2 ) {
            float4 modt = tex2D( BlendModulationSampler, ... );
            float minb  = saturate( modt.g - modt.r );
            float maxb  = saturate( modt.g + modt.r );
            blendfactor = smoothstep( minb, maxb, blendfactor );
            baseColor.rgb = lerp( baseColor, baseColor2.rgb, blendfactor );
        }

    Blender's Principled BSDF has no cubemap slot, so the environment reflection
    itself is left to the scene world and only its strength/tint are translated.
    """
    SHADER = 'lightmappedgeneric'

    #: ``worldvertextransition.cpp`` sets ``m_nAlphaTestReference = -1``, i.e. the
    #: parameter does not exist for that shader. Subclasses can opt out.
    SUPPORTS_ALPHATESTREFERENCE = True

    # ------------------------------------------------------------------ textures

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0.0, 0.3, 1.0))

    @property
    def basetexture2(self):
        return self._texture_property('$basetexture2', (0.3, 0.3, 0.0, 1.0))

    @property
    def blendmodulatetexture(self):
        return self._texture_property('$blendmodulatetexture', (0.3, 0.0, 0.3, 1.0), is_data=True)

    @property
    def bumpmap(self):
        return self._texture_property('$bumpmap', (0.5, 0.5, 1.0, 1.0),
                                      normal_map=True, ssbump=self.ssbump)

    @property
    def bumpmap2(self):
        return self._texture_property('$bumpmap2', (0.5, 0.5, 1.0, 1.0),
                                      normal_map=True, ssbump=self.ssbump)

    @property
    def envmapmask(self):
        return self._texture_property('$envmapmask', (1, 1, 1, 1.0), is_data=True)

    @property
    def selfillummask(self):
        return self._texture_property('$selfillummask', (0.0, 0.0, 0.0, 1.0), is_data=True)

    # -------------------------------------------------------------------- colours

    @property
    def envmaptint(self):
        return self._color_property('$envmaptint', [1, 1, 1])

    @property
    def selfillumtint(self):
        return self._color_property('$selfillumtint', [1, 1, 1])

    # ---------------------------------------------------------------------- flags

    @property
    def isskybox(self):
        return self._vmt.get_int('%' + 'compilesky', 0) + self._vmt.get_int('%' + 'compile2Dsky', 0)

    @property
    def ssbump(self):
        return self._bool_property('$ssbump')

    @property
    def alphatest(self):
        return self._bool_property('$alphatest')

    @property
    def translucent(self):
        return self._bool_property('$translucent')

    @property
    def selfillum(self):
        return self._bool_property('$selfillum')

    @property
    def maskedblending(self):
        return self._bool_property('$maskedblending')

    @property
    def envmap(self):
        return self._vmt.get_string('$envmap', None) is not None

    @property
    def basealphaenvmapmask(self):
        return self._bool_property('$basealphaenvmapmask')

    @property
    def normalmapalphaenvmapmask(self):
        return self._bool_property('$normalmapalphaenvmapmask')

    @property
    def basetexturenoenvmap(self):
        return self._bool_property('$basetexturenoenvmap')

    @property
    def basetexture2noenvmap(self):
        return self._bool_property('$basetexture2noenvmap')

    # --------------------------------------------------------------------- floats

    @property
    def alpha(self):
        return self._vmt.get_float('$alpha', 1.0)

    @property
    def alphatestreference(self):
        if not self.SUPPORTS_ALPHATESTREFERENCE:
            return 0.5
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def fresnelreflection(self):
        # "1.0 == mirror, 0.0 == water" per the SDK parameter description.
        return self._vmt.get_float('$fresnelreflection', 1.0)

    @property
    def envmapcontrast(self):
        return self._vmt.get_float('$envmapcontrast', 0.0)

    @property
    def envmapsaturation(self):
        return self._vmt.get_float('$envmapsaturation', 1.0)

    @property
    def seamless_scale(self):
        # 0 means "use ordinary mapping" per the SDK parameter description.
        return self._vmt.get_float('$seamless_scale', 0.0)

    # ----------------------------------------------------------------- transforms

    _DEFAULT_TRANSFORM = {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1),
                          'rotate': (0, 0, 0), 'translate': (0, 0, 0)}

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform', self._DEFAULT_TRANSFORM)

    @property
    def bumptransform(self):
        return self._vmt.get_transform_matrix('$bumptransform', self._DEFAULT_TRANSFORM)

    @property
    def bumptransform2(self):
        return self._vmt.get_transform_matrix('$bumptransform2', self._DEFAULT_TRANSFORM)

    @property
    def envmapmasktransform(self):
        return self._vmt.get_transform_matrix('$envmapmasktransform', self._DEFAULT_TRANSFORM)

    @property
    def blendmasktransform(self):
        return self._vmt.get_transform_matrix('$blendmasktransform', self._DEFAULT_TRANSFORM)

    # ------------------------------------------------------------------ node tree

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        if self.isskybox:
            if not is_blender_4_3():
                # 4.3 removed shadow_method; disabling shadow casting moved to the
                # object level (Object.visible_shadow), which a material-level
                # shader cannot reach, so there is no 4.3+ equivalent here.
                self.bpy_material.shadow_method = 'NONE'
            self.bpy_material.use_backface_culling = True

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        uv_out = uv_node.outputs[0]

        seamless_scale = self.seamless_scale
        if seamless_scale:
            # $seamless_scale rescales the (world-space projected) UVs.
            scale_node = self.create_node(Nodes.ShaderNodeVectorMath, 'seamless_scale')
            scale_node.operation = 'SCALE'
            self.connect_nodes(uv_out, scale_node.inputs[0])
            scale_node.inputs['Scale'].default_value = seamless_scale
            uv_out = scale_node.outputs[0]

        # `blend_output` carries the SDK's `blendfactor`, shared by the albedo and
        # normal blends. It stays None when there is only one layer.
        blend_output = None
        basetexture_node = None
        albedo_output = None

        basetexture = self.basetexture
        basetexture2 = self.basetexture2

        if basetexture and basetexture2:
            basetexture_node, albedo_output, blend_output = self._build_blended_albedo(
                shader, basetexture, basetexture2, uv_node, uv_out)
        elif basetexture:
            basetexture_node = self.create_texture_node(basetexture, '$basetexture')
            self._apply_transform(self.basetexturetransform, basetexture_node, uv_node, uv_out)
            self.connect_nodes(basetexture_node.outputs['Color'], shader.inputs['Base Color'])
            albedo_output = basetexture_node.outputs['Color']
            if self.detail:
                albedo_output, _ = self.handle_detail(shader.inputs['Base Color'], albedo_output,
                                                      uv_node=uv_node)

        if basetexture_node is not None:
            self._setup_alpha(shader, basetexture_node)

        bumpmap_node = self._setup_normals(shader, uv_node, uv_out, blend_output)

        if self.selfillum and albedo_output is not None:
            self._setup_selfillum(shader, albedo_output, basetexture_node)

        self._setup_envmap(shader, basetexture_node, bumpmap_node, uv_node, uv_out)

    def _apply_transform(self, transform, texture_node, uv_node, uv_out):
        """Feed ``texture_node`` from ``uv_out``, inserting a Mapping when needed.

        Reuses the shared UV chain for an identity transform so we do not emit a
        redundant UVMap/Mapping pair per texture.
        """
        if transform and transform != self._DEFAULT_TRANSFORM:
            self.handle_transform(transform, texture_node.inputs[0], uv_node=uv_node)
        else:
            self.connect_nodes(uv_out, texture_node.inputs[0])

    def _build_blended_albedo(self, shader, basetexture, basetexture2, uv_node, uv_out):
        """Two-way ``$basetexture``/``$basetexture2`` blend.

        Returns ``(basetexture_node, albedo_output, blend_output)``.
        """
        basetexture_node = self.create_texture_node(basetexture, '$basetexture')
        self._apply_transform(self.basetexturetransform, basetexture_node, uv_node, uv_out)
        basetexture2_node = self.create_texture_node(basetexture2, '$basetexture2')
        self._apply_transform(self.basetexturetransform, basetexture2_node, uv_node, uv_out)

        blend_output = self._build_blend_factor(uv_node, uv_out)

        color_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'basetexture blend')
        color_mix.blend_type = 'MIX'
        self.connect_nodes(blend_output, color_mix.inputs['Fac'])
        albedo_output = color_mix.outputs['Color']

        bs_socket = basetexture_node.outputs['Color']
        bs_socket2 = basetexture2_node.outputs['Color']

        if self.detail:
            if self.detail2:
                # Each layer gets its own detail texture before blending.
                self.handle_detail(color_mix.inputs['Color1'], bs_socket, uv_node=uv_node)
                self.handle_detail2(color_mix.inputs['Color2'], bs_socket2, uv_node=uv_node)
                self.connect_nodes(albedo_output, shader.inputs['Base Color'])
            else:
                self.connect_nodes(bs_socket, color_mix.inputs['Color1'])
                self.connect_nodes(bs_socket2, color_mix.inputs['Color2'])
                albedo_output, _ = self.handle_detail(shader.inputs['Base Color'], albedo_output,
                                                     uv_node=uv_node)
        else:
            self.connect_nodes(bs_socket, color_mix.inputs['Color1'])
            self.connect_nodes(bs_socket2, color_mix.inputs['Color2'])
            self.connect_nodes(albedo_output, shader.inputs['Base Color'])

        return basetexture_node, albedo_output, blend_output

    def _build_blend_factor(self, uv_node, uv_out):
        """Return a socket carrying the SDK's ``blendfactor``.

        ``smoothstep( saturate(modt.g - modt.r), saturate(modt.g + modt.r), vertexAlpha )``
        or, with ``$maskedblending``, just ``modt.g``.
        """
        vertex_color = self.create_node(Nodes.ShaderNodeVertexColor, 'vertex blend')
        blendmodulatetexture = self.blendmodulatetexture

        if blendmodulatetexture is None:
            # No modulation texture: the raw vertex alpha is the blend factor.
            return vertex_color.outputs['Alpha']

        modt = self.create_texture_node(blendmodulatetexture, '$blendmodulatetexture')
        self._apply_transform(self.blendmasktransform, modt, uv_node, uv_out)
        sep_rgb = self.create_node(Nodes.ShaderNodeSeparateColor, 'blendmodulate split')
        sep_rgb.mode = "RGB"
        self.connect_nodes(modt.outputs['Color'], sep_rgb.inputs[0])

        if self.maskedblending:
            # SDK: `blendfactor = modt.g` -- vertex alpha is ignored entirely.
            return sep_rgb.outputs[1]

        # minb = saturate(g - r), maxb = saturate(g + r)
        min_sub = self.create_node(Nodes.ShaderNodeMath, 'blend min')
        min_sub.operation = 'SUBTRACT'
        min_sub.use_clamp = True  # saturate()
        self.connect_nodes(sep_rgb.outputs[1], min_sub.inputs[0])
        self.connect_nodes(sep_rgb.outputs[0], min_sub.inputs[1])

        # smoothstep divides by (maxb - minb), so the edges must never be equal.
        # They collapse whenever the modulation texture's red channel is 0
        # (minb == maxb == g), which divides by zero and renders solid black.
        # saturate() cannot prevent that, so bias the lower edge down slightly.
        # Since r >= 0 implies saturate(g-r) <= saturate(g+r), this guarantees
        # minb < maxb for every possible texel.
        min_biased = self.create_node(Nodes.ShaderNodeMath, 'blend min bias')
        min_biased.operation = 'SUBTRACT'
        min_biased.inputs[1].default_value = 0.001
        self.connect_nodes(min_sub.outputs[0], min_biased.inputs[0])

        max_add = self.create_node(Nodes.ShaderNodeMath, 'blend max')
        max_add.operation = 'ADD'
        max_add.use_clamp = True  # saturate()
        self.connect_nodes(sep_rgb.outputs[1], max_add.inputs[0])
        self.connect_nodes(sep_rgb.outputs[0], max_add.inputs[1])

        smoothstep = self.create_node(Nodes.ShaderNodeMapRange, 'blendfactor smoothstep')
        smoothstep.interpolation_type = 'SMOOTHSTEP'
        # Map Range's From Min/From Max are the smoothstep edges.
        self.connect_nodes(vertex_color.outputs['Alpha'], smoothstep.inputs['Value'])
        self.connect_nodes(min_biased.outputs[0], smoothstep.inputs['From Min'])
        self.connect_nodes(max_add.outputs[0], smoothstep.inputs['From Max'])
        return smoothstep.outputs[0]

    def _setup_normals(self, shader, uv_node, uv_out, blend_output):
        """Blend the two bump maps with the same factor as the albedo."""
        bumpmap = self.bumpmap
        if not bumpmap:
            return None

        bumpmap_node = self.create_texture_node(bumpmap, '$bumpmap')
        self._apply_transform(self.bumptransform, bumpmap_node, uv_node, uv_out)
        normal_source = bumpmap_node.outputs['Color']

        # Only blend when there is a second map *and* a factor to blend with.
        bumpmap2 = self.bumpmap2
        if bumpmap2 and blend_output is not None:
            bumpmap2_node = self.create_texture_node(bumpmap2, '$bumpmap2')
            self._apply_transform(self.bumptransform2, bumpmap2_node, uv_node, uv_out)
            normal_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'bumpmap blend')
            normal_mix.blend_type = 'MIX'
            self.connect_nodes(blend_output, normal_mix.inputs['Fac'])
            self.connect_nodes(normal_source, normal_mix.inputs['Color1'])
            self.connect_nodes(bumpmap2_node.outputs['Color'], normal_mix.inputs['Color2'])
            normal_source = normal_mix.outputs['Color']

        normal_map = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_source, normal_map.inputs['Color'])
        self.connect_nodes(normal_map.outputs['Normal'], shader.inputs['Normal'])
        return bumpmap_node

    def _setup_alpha(self, shader, basetexture_node):
        """Wire base-texture alpha, but only when the material is actually blended.

        The pixel shader does compute ``alpha *= baseColor.a`` (guarded by
        ``!bBaseAlphaEnvmapMask && !bSelfIllum``), but that alone does not make the
        surface transparent: ``lightmappedgeneric_dx9_helper.cpp`` classifies the
        material as fully opaque unless it is blended or alpha-tested::

            bFullyOpaqueWithoutAlphaTest = (nBlendType != BT_BLENDADD) &&
                                           (nBlendType != BT_BLEND) && ...;
            bFullyOpaque = bFullyOpaqueWithoutAlphaTest && !bIsAlphaTested;
            ...
            pShaderShadow->EnableAlphaWrites( bFullyOpaque );

        For an opaque material the alpha output is used for depth/special purposes
        rather than blending, so wiring it into Alpha here would make solid world
        geometry see-through. Only connect it when ``$alphatest`` or
        ``$translucent`` asks for blending.

        The channel is also claimed by ``$basealphaenvmapmask`` (specular mask),
        ``$selfillum`` (illumination mask) and ``$detailblendmode`` 7
        (``TCOMBINE_MOD2X_SELECT_TWO_PATTERNS`` pattern selector); the SDK clears
        ``MATERIAL_VAR_ALPHATEST`` outright for the first two.
        """
        if self.basealphaenvmapmask or self.selfillum:
            return
        if self.detail is not None and self.detailmode == TCOMBINE_MOD2X_SELECT_TWO_PATTERNS:
            return

        if self.alphatest:
            self.set_blend_mode('HASHED', alpha_threshold=self.alphatestreference)
        elif self.translucent:
            self.set_blend_mode('BLEND')
            self.bpy_material.use_backface_culling = True
            self.bpy_material.show_transparent_back = False
        else:
            # Fully opaque: alpha is not a blend factor here.
            return

        self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Alpha'])

    def _setup_selfillum(self, shader, albedo_output, basetexture_node):
        """SDK: ``diffuse = lerp( diffuse, $selfillumtint * albedo, baseColor.a )``.

        Base-texture alpha is the self-illumination mask, so it becomes emission
        strength and the (optionally tinted) albedo becomes emission colour.
        """
        emission_color_input = 'Emission Color' if is_blender_4() else 'Emission'
        if emission_color_input not in shader.inputs:
            return

        tint = self.selfillumtint
        if tint is not None and tuple(tint[:3]) != (1.0, 1.0, 1.0):
            tint_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'selfillumtint')
            tint_mix.blend_type = 'MULTIPLY'
            tint_mix.inputs['Fac'].default_value = 1.0
            self.connect_nodes(albedo_output, tint_mix.inputs['Color1'])
            tint_mix.inputs['Color2'].default_value = tint
            self.connect_nodes(tint_mix.outputs['Color'], shader.inputs[emission_color_input])
        else:
            self.connect_nodes(albedo_output, shader.inputs[emission_color_input])

        if 'Emission Strength' in shader.inputs:
            mask = self.selfillummask
            if mask is not None:
                mask_node = self.create_texture_node(mask, '$selfillummask')
                self.connect_nodes(mask_node.outputs['Color'], shader.inputs['Emission Strength'])
            elif basetexture_node is not None:
                self.connect_nodes(basetexture_node.outputs['Alpha'], shader.inputs['Emission Strength'])

    def _setup_envmap(self, shader, basetexture_node, bumpmap_node, uv_node, uv_out):
        """Translate the cubemap specular term into Principled inputs.

        Blender has no cubemap slot on Principled -- environment reflection comes
        from the world -- so only the reflection *strength* and tint are mapped.
        """
        spec_input = 'Specular IOR Level' if is_blender_4() else 'Specular'

        if not self.envmap:
            # No cubemap: the SDK emits diffuse only.
            shader.inputs[spec_input].default_value = 0.0
            return

        # $basetexturenoenvmap zeroes the normal alpha for that layer
        # (`vNormal.a = 0.0f`), killing its reflection. With a single layer that
        # means no reflection at all.
        if self.basetexturenoenvmap and not self.basetexture2:
            shader.inputs[spec_input].default_value = 0.0
            return

        # $fresnelreflection scales how much reflection survives head-on
        # (1.0 == mirror). Fold it into the constant strength.
        strength = self.clamp_value(0.5 * self.fresnelreflection)

        # specularFactor: exactly one of the three mask sources, per the shader's
        # SKIP directives.
        mask_output = None
        envmapmask = self.envmapmask
        if envmapmask is not None:
            mask_node = self.create_texture_node(envmapmask, '$envmapmask')
            self._apply_transform(self.envmapmasktransform, mask_node, uv_node, uv_out)
            mask_output = mask_node.outputs['Color']
        elif self.normalmapalphaenvmapmask and bumpmap_node is not None:
            mask_output = bumpmap_node.outputs['Alpha']
        elif self.basealphaenvmapmask and basetexture_node is not None:
            # SDK: `specularFactor *= 1.0 - blendedAlpha` -- note the inversion.
            invert = self.create_node(Nodes.ShaderNodeInvert, 'basealphaenvmapmask')
            self.connect_nodes(basetexture_node.outputs['Alpha'], invert.inputs['Color'])
            mask_output = invert.outputs['Color']

        # $envmaptint / $envmapcontrast / $envmapsaturation all scale the
        # reflection's *brightness and colour*, not its sharpness:
        #
        #   specularLighting *= g_EnvmapTint;
        #   specularLighting  = lerp( spec, spec*spec, g_EnvmapContrast );
        #   greyScale         = dot( spec, (0.299, 0.587, 0.114) );
        #   specularLighting  = lerp( greyScale, spec, g_EnvmapSaturation );
        #
        # Since Principled has no cubemap input, collapse them into a single
        # brightness scale on the specular strength plus a tint colour. Working in
        # luminance keeps `lerp(c, c*c, contrast)` meaningful: squaring a value
        # below 1 *darkens* it, so high contrast means a dimmer, punchier
        # reflection -- not a glossier one.
        tint = self.envmaptint or [1.0, 1.0, 1.0, 1.0]
        luma = 0.299 * tint[0] + 0.587 * tint[1] + 0.114 * tint[2]

        contrast = self.clamp_value(self.envmapcontrast)
        if contrast:
            # lerp( luma, luma^2, contrast )
            luma = luma * (1.0 - contrast) + (luma * luma) * contrast

        strength *= self.clamp_value(luma)
        shader_tint = tint

        saturation = self.envmapsaturation
        if saturation != 1.0:
            # lerp( greyScale, colour, saturation ) -- desaturate the tint toward
            # its own luminance.
            grey = 0.299 * tint[0] + 0.587 * tint[1] + 0.114 * tint[2]
            shader_tint = [c * saturation + grey * (1.0 - saturation) for c in tint[:3]] + [1.0]

        if mask_output is not None:
            # specularFactor is a per-pixel multiplier on the reflection, so scale
            # the constant strength by the mask: Mix(Fac=mask, A=0, B=strength).
            mask_scale = self.create_node(Nodes.ShaderNodeMix, 'envmapmask')
            mask_scale.data_type = 'FLOAT'
            self.connect_nodes(mask_output, mask_scale.inputs['Factor'])
            mask_scale.inputs['A'].default_value = 0.0
            mask_scale.inputs['B'].default_value = self.clamp_value(strength)
            self.connect_nodes(mask_scale.outputs[0], shader.inputs[spec_input])
        else:
            shader.inputs[spec_input].default_value = self.clamp_value(strength)

        if 'Specular Tint' in shader.inputs and tuple(shader_tint[:3]) != (1.0, 1.0, 1.0):
            # Normalise so the tint carries hue only; brightness already went into
            # the strength above and would otherwise be applied twice.
            peak = max(shader_tint[:3])
            if peak > 0.0:
                shader.inputs['Specular Tint'].default_value = [c / peak for c in shader_tint[:3]] + [1.0]


class WorldVertexTransition(LightmapGeneric):
    """WorldVertexTransition -- two-texture blended world geometry (displacements).

    ``worldvertextransition.cpp`` sets up a ``LightmappedGeneric_DX9_Vars_t`` and
    calls ``DrawLightmappedGeneric_DX9``, so it is the same shader with a different
    name. The only functional difference is that it does not expose
    ``$alphatestreference`` (``info.m_nAlphaTestReference = -1``).
    """
    SHADER = 'worldvertextransition'
    SUPPORTS_ALPHATESTREFERENCE = False


class ReflectiveLightmapGeneric(LightmapGeneric):
    SHADER = 'lightmappedreflective'


class SDKLightmapGeneric(LightmapGeneric):
    SHADER = 'sdk_lightmappedgeneric'
