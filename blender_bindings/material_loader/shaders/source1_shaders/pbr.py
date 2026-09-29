from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4, is_blender_4_3


class StrataPbr(Source1ShaderBase):
    SHADER = 'pbr'

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0, 0.3, 1.0))

    @property
    def bumpmap(self):
        return self._texture_property('$bumpmap', (0.5, 0.5, 1.0, 1.0), is_data=True, normal_map=True)

    @property
    def mraotexture(self):
        return self._texture_property('$mraotexture', (0.5, 0.5, 1.0, 1.0), is_data=True)

    @property
    def emissiontexture(self):
        return self._texture_property('$emissiontexture', (0.0, 0.0, 0.0, 1.0))

    @property
    def mraoscale(self):
        return self._color_property('$mraoscale', length=3)

    @property
    def envmaptint(self):
        return self._color_property('$envmaptint')

    @property
    def translucent(self):
        return self._bool_property('$translucent', 0)

    @property
    def alphatest(self):
        return self._bool_property('$alphatest', 0)

    @property
    def alphatestreference(self):
        return self._vmt.get_float('$alphatestreference', 0.5)

    @property
    def basetexturetransform(self):
        return self._vmt.get_transform_matrix('$basetexturetransform', self._DEFAULT_TRANSFORM)

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])

        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        uv_out = uv_node.outputs[0]
        albedo_out = None
        basetexture = self.basetexture
        if basetexture:
            basetexture_node = self.create_and_connect_texture_node(basetexture, shader.inputs["Base Color"],name='$basetexture')
            basetexture_node.location = [-800, 0]
            albedo_out = basetexture_node.outputs[0]
            if self.basetexturetransform:
                self._apply_transform(self.basetexturetransform, basetexture_node, uv_node, uv_out)

        bumpmap = self.bumpmap
        if bumpmap:
            bumpmap_node = self.create_node(Nodes.ShaderNodeTexImage, '$bumpmap')
            bumpmap_node.image = bumpmap

            if self.basetexturetransform:
                self._apply_transform(self.basetexturetransform, bumpmap_node, uv_node, uv_out)

            normalmap_node = self.create_node(Nodes.ShaderNodeNormalMap)

            self.connect_nodes(bumpmap_node.outputs['Color'], normalmap_node.inputs['Color'])
            self.connect_nodes(normalmap_node.outputs['Normal'], shader.inputs['Normal'])

        mrao = self.mraotexture
        if mrao:
            mrao_node = self.create_and_connect_texture_node(mrao, shader.inputs["Metallic"],name='$mraotexture')
            mrao_node.location = [-800, -400]

            if self.basetexturetransform:
                self._apply_transform(self.basetexturetransform, mrao_node, uv_node, uv_out)

            separate_color = self.create_node(Nodes.ShaderNodeSeparateColor)
            self.connect_nodes(mrao_node.outputs['Color'], separate_color.inputs['Color'])
            self.connect_nodes(separate_color.outputs['Red'], shader.inputs['Metallic'])
            self.connect_nodes(separate_color.outputs['Green'], shader.inputs['Roughness'])
            # self.connect_nodes(separate_color.outputs['B'], shader.inputs['Specular'])

        emissiontexture = self.emissiontexture

        if emissiontexture:
            emission_node = self.create_and_connect_texture_node(emissiontexture, shader.inputs["Emission Color"],name='$emissiontexture')
            emission_node.location = [-800, -800]

            if self.basetexturetransform:
                self._apply_transform(self.basetexturetransform, emission_node, uv_node, uv_out)

        object_info = self.create_node(Nodes.ShaderNodeObjectInfo)
        tint_mixer = self.create_node(Nodes.ShaderNodeMix)
        tint_mixer.data_type = 'RGBA'
        tint_mixer.blend_type = 'MULTIPLY'

        tint_mixer.location = [-600, -800]
        tint_mixer.inputs['Factor'].default_value = 1.0
        self.insert_node(albedo_out, tint_mixer.inputs['A'], tint_mixer.outputs[0])
        self.connect_nodes(object_info.outputs['Color'], tint_mixer.inputs['B'])
        self.connect_nodes(tint_mixer.outputs['Result'], shader.inputs['Base Color'])

    def _apply_transform(self, transform, texture_node, uv_node, uv_out):
        """Feed ``texture_node`` from ``uv_out``, inserting a Mapping when needed.

        Reuses the shared UV chain for an identity transform so we do not emit a
        redundant UVMap/Mapping pair per texture.
        """
        if transform and transform != self._DEFAULT_TRANSFORM:
            self.handle_transform(transform, texture_node.inputs[0], uv_node=uv_node)
        else:
            self.connect_nodes(uv_out, texture_node.inputs[0])
