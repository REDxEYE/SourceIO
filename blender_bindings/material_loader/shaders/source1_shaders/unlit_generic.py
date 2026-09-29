from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from .detail import DetailSupportMixin


class UnlitGeneric(DetailSupportMixin, Source1ShaderBase):
    """UnlitGeneric -- emissive/unlit surfaces.

    In DX9 this is not a separate shader: ``unlitgeneric_dx9.cpp`` fills a
    ``VertexLitGeneric_DX9_Vars_t`` and calls
    ``DrawVertexLitGeneric_DX9( ..., bVertexLitGeneric=false, ... )``. Inside the
    helper that flag only disables diffuse lighting and enables vertex colour::

        bool hasDiffuseLighting = bVertexLitGeneric;                                  // false here
        bool bHasVertexColor = bVertexLitGeneric ? false : IS_FLAG_SET( MATERIAL_VAR_VERTEXCOLOR );
        bool bHasVertexAlpha = bVertexLitGeneric ? false : IS_FLAG_SET( MATERIAL_VAR_VERTEXALPHA );

    So the parameter surface is VertexLitGeneric's -- ``$detail``, ``$envmap``,
    ``$selfillum``... -- and it is modelled here with an Emission shader (no
    diffuse response) rather than a Principled BSDF.
    """
    SHADER: str = 'unlitgeneric'

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0, 0.3, 1.0))

    @property
    def color(self):
        return self._color_property('$color')

    @property
    def color2(self):
        return self._color_property('$color2')

    @property
    def translucent(self):
        return self._bool_property('$translucent')

    @property
    def alphatest(self):
        return self._bool_property('$alphatest')

    @property
    def alphatestreference(self):
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def additive(self):
        return self._bool_property('$additive')

    @property
    def nocull(self):
        return self._bool_property('$nocull')

    @property
    def vertexcolor(self):
        return self._bool_property('$vertexcolor')

    @property
    def vertexalpha(self):
        return self._bool_property('$vertexalpha')

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1),
                                               'rotate': (0, 0, 0), 'translate': (0, 0, 0)})

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeEmission, self.SHADER)
        shader.inputs['Strength'].default_value = 1.0

        if self.nocull:
            self.bpy_material.use_backface_culling = False

        blended = self.translucent or self.alphatest or self.additive
        mix_node = None
        if blended:
            self.set_blend_mode('BLEND' if (self.translucent or self.additive) else 'HASHED',
                                alpha_threshold=self.alphatestreference if self.alphatest else None)
            transparent_node = self.create_node(Nodes.ShaderNodeBsdfTransparent)
            if self.additive:
                # Additive surfaces add their emission on top of a transparent
                # base rather than replacing it.
                mix_node = self.create_node(Nodes.ShaderNodeAddShader, 'additive')
                self.connect_nodes(transparent_node.outputs['BSDF'], mix_node.inputs[0])
                self.connect_nodes(shader.outputs['Emission'], mix_node.inputs[1])
            else:
                mix_node = self.create_node(Nodes.ShaderNodeMixShader, 'alpha mix')
                self.connect_nodes(transparent_node.outputs['BSDF'], mix_node.inputs[1])
                self.connect_nodes(shader.outputs['Emission'], mix_node.inputs[2])
            self.connect_nodes(mix_node.outputs['Shader'], material_output.inputs['Surface'])
        else:
            self.connect_nodes(shader.outputs['Emission'], material_output.inputs['Surface'])

        basetexture = self.basetexture
        if basetexture is None:
            # $color still applies without a texture (e.g. flat coloured overlays).
            color = self.color or self.color2
            if color:
                shader.inputs['Color'].default_value = color
            return

        basetexture_node = self.create_texture_node(basetexture, '$basetexture')
        if self.basetexturetransform:
            _, uv_node = self.handle_transform(self.basetexturetransform, basetexture_node.inputs[0])
        else:
            uv_node = None

        color_output = basetexture_node.outputs['Color']

        if self.detail:
            color_output, _ = self.handle_detail(shader.inputs['Color'], color_output, uv_node=uv_node)

        color = self.color or self.color2
        if color:
            color_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'color')
            color_mix.blend_type = 'MULTIPLY'
            color_mix.inputs['Fac'].default_value = 1.0
            self.connect_nodes(color_output, color_mix.inputs['Color1'])
            color_mix.inputs['Color2'].default_value = color
            color_output = color_mix.outputs['Color']

        if self.vertexcolor:
            # Unlit passes enable vertex colour modulation (bHasVertexColor above).
            vertex_color = self.create_node(Nodes.ShaderNodeVertexColor, 'vertexcolor')
            vc_mix = self.create_node(Nodes.ShaderNodeMixRGB, 'vertexcolor mix')
            vc_mix.blend_type = 'MULTIPLY'
            vc_mix.inputs['Fac'].default_value = 1.0
            self.connect_nodes(color_output, vc_mix.inputs['Color1'])
            self.connect_nodes(vertex_color.outputs['Color'], vc_mix.inputs['Color2'])
            color_output = vc_mix.outputs['Color']

        self.connect_nodes(color_output, shader.inputs['Color'])

        # Alpha drives the transparent mix (additive uses the texture's own
        # brightness instead, so no factor is needed there).
        if mix_node is not None and not self.additive:
            alpha_output = basetexture_node.outputs['Alpha']
            if self.vertexalpha:
                vertex_color = self.get_node('vertexcolor') or self.create_node(
                    Nodes.ShaderNodeVertexColor, 'vertexcolor')
                alpha_mul = self.create_node(Nodes.ShaderNodeMath, 'vertexalpha')
                alpha_mul.operation = 'MULTIPLY'
                self.connect_nodes(alpha_output, alpha_mul.inputs[0])
                self.connect_nodes(vertex_color.outputs['Alpha'], alpha_mul.inputs[1])
                alpha_output = alpha_mul.outputs[0]
            self.connect_nodes(alpha_output, mix_node.inputs['Fac'])


class SDKUnlitGeneric(UnlitGeneric):
    """``sdk_unlitgeneric`` behaves like ``unlitgeneric``.

    Previously this subclassed :class:`Source1ShaderBase` directly without a
    ``create_nodes`` implementation, so every ``sdk_unlitgeneric`` material raised
    ``NotImplementedError`` from the abstract base and failed to import.
    """
    SHADER: str = 'sdk_unlitgeneric'
