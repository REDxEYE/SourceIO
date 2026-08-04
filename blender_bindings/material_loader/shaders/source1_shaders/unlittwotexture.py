from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase


class UnlitTwoTexture(Source1ShaderBase):
    """UnlitTwoTexture -- two multiplied unlit textures.

    ``unlittwotexture_ps2x.fxc``::

        HALF4 baseColor  = tex2D( BaseTextureSampler,  i.baseTexCoord.xy );
        HALF4 baseColor2 = tex2D( BaseTextureSampler2, i.baseTexCoord2.xy );
        HALF4 result     = baseColor * baseColor2 * g_DiffuseModulation;
        float alpha      = 1.0f;
        return FinalOutput( float4( result.rgb, alpha ), ... );

    Both textures are loaded as sRGB colour (``LoadTexture( TEXTURE2,
    TEXTUREFLAGS_SRGB )``), and ``g_DiffuseModulation`` carries ``$color``.
    Note the shader hardcodes ``alpha = 1.0``: base alpha only reaches the
    framebuffer when ``$translucent``/``$additive`` puts the material in a blended
    pass.
    """
    SHADER: str = 'unlittwotexture'

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0, 0.3, 1.0))

    @property
    def texture2(self):
        # SDK loads this with TEXTUREFLAGS_SRGB -- it is a colour texture, so it
        # must NOT be flagged as non-colour data.
        return self._texture_property('$texture2', (1.0, 1.0, 1.0, 1.0))

    @property
    def color(self):
        return self._color_property('$color', [1, 1, 1])

    @property
    def color2(self):
        return self._color_property('$color2', [1, 1, 1])

    @property
    def additive(self):
        return self._bool_property('$additive')

    @property
    def translucent(self):
        return self._bool_property('$translucent')

    @property
    def nocull(self):
        return self._bool_property('$nocull')

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1),
                                               'rotate': (0, 0, 0), 'translate': (0, 0, 0)})

    @property
    def texture2transform(self):
        return self._vmt.get_transform_matrix('$texture2transform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1),
                                               'rotate': (0, 0, 0), 'translate': (0, 0, 0)})

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        self.do_arrange = True
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        self.bpy_material.use_backface_culling = not self.nocull

        shader = self.create_node(Nodes.ShaderNodeEmission, self.SHADER)
        shader.inputs['Strength'].default_value = 1.0

        basetexture_node = None
        color_output = None

        basetexture = self.basetexture
        if basetexture is not None:
            basetexture_node = self.create_texture_node(basetexture, '$basetexture')
            if self.basetexturetransform:
                self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
            color_output = basetexture_node.outputs['Color']

            texture2 = self.texture2
            if texture2 is not None:
                texture2_node = self.create_texture_node(texture2, '$texture2')
                if self.texture2transform:
                    self.handle_transform(self.texture2transform, texture2_node.inputs[0])
                # result = baseColor * baseColor2
                twotex = self.create_node(Nodes.ShaderNodeMixRGB, 'twotex_mult')
                twotex.blend_type = 'MULTIPLY'
                twotex.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_output, twotex.inputs['Color1'])
                self.connect_nodes(texture2_node.outputs['Color'], twotex.inputs['Color2'])
                color_output = twotex.outputs['Color']

            # ... * g_DiffuseModulation ($color)
            color = self.color or self.color2
            if color is not None and tuple(color[:3]) != (1.0, 1.0, 1.0):
                color_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'color_mult')
                color_mix.blend_type = 'MULTIPLY'
                color_mix.inputs['Fac'].default_value = 1.0
                self.connect_nodes(color_output, color_mix.inputs['Color1'])
                color_mix.inputs['Color2'].default_value = self.ensure_length(list(color[:3]), 4, 1.0)
                color_output = color_mix.outputs['Color']

            self.connect_nodes(color_output, shader.inputs['Color'])
        else:
            color = self.color or self.color2
            if color is not None:
                shader.inputs['Color'].default_value = self.ensure_length(list(color[:3]), 4, 1.0)

        surface_output = shader.outputs['Emission']

        if self.additive:
            self.set_blend_mode('BLEND')
            transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
            add_shader = self.create_node(Nodes.ShaderNodeAddShader, 'additive')
            self.connect_nodes(transparent.outputs['BSDF'], add_shader.inputs[0])
            self.connect_nodes(surface_output, add_shader.inputs[1])
            surface_output = add_shader.outputs['Shader']
        elif self.translucent and basetexture_node is not None:
            # Only meaningful with a base texture to take alpha from.
            self.set_blend_mode('BLEND')
            mix = self.create_node(Nodes.ShaderNodeMixShader, 'alpha mix')
            transparent = self.create_node(Nodes.ShaderNodeBsdfTransparent)
            self.connect_nodes(transparent.outputs['BSDF'], mix.inputs[1])
            self.connect_nodes(surface_output, mix.inputs[2])
            self.connect_nodes(basetexture_node.outputs['Alpha'], mix.inputs['Fac'])
            surface_output = mix.outputs['Shader']

        culling = self.create_node_group('BackfaceCulling')
        if culling.node_tree is not None and '$nocull' in culling.inputs:
            # One boolean property drives backface culling in both EEVEE and Cycles.
            driver = culling.inputs['$nocull'].driver_add('default_value')
            driver.driver.expression = '1-var'
            var = driver.driver.variables.new()
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'MATERIAL'
            var.targets[0].id = self.bpy_material
            var.targets[0].data_path = 'use_backface_culling'
            self.connect_nodes(surface_output, culling.inputs[0])
            surface_output = culling.outputs[0]

        self.connect_nodes(surface_output, material_output.inputs['Surface'])
