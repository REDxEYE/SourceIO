from typing import Any

import bpy
from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4

#: Vertex-colour layer written by the BSP importer for displacements that carry a
#: LUMP_DISP_MULTIBLEND. ``m_vMultiBlend`` is ``(w1, w2, w3, w4)`` mapped straight
#: onto RGBA; CS:GO's pixel shader reads only ``.g``/``.b``/``.a`` for layers 2/3/4
#: and never samples ``.r``, because layer 1 is the base the lerp chain starts from.
MULTIBLEND_LAYER = 'multiblend'


class Lightmapped4WayBlend(Source1ShaderBase):
    """lightmapped_4wayblend -- CS:GO four-texture displacement blend.

    Absent from Source SDK 2013, but ``lightmapped_4wayblend_ps20b.fxc`` exists in
    public CS:GO source mirrors. Per layer, the vertex weight is *gained* by a
    luminance term and then smoothstepped, applied as a sequential lerp chain::

        lumN         = smoothstep($textureN_lumstart, $textureN_lumend, Luminance(baseColorN))
        lum          = lerp(1 - lumPrev, lumN, $lumblendfactorN)
        blendfactorN = smoothstep($textureN_blendstart, $textureN_blendend, w * (1 + lum))
        baseColor    = lerp(baseColor, baseColorN, blendfactorN)

    Note ``w * (1 + lum)`` -- a gain in ``[w, 2w]``, not a multiply. There is no
    ``$blendmodulatetexture``; the luminance system replaces it. The bundled
    ``4wayBlend`` node group implements these curves.
    """
    SHADER = 'lightmapped_4wayblend'

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0.0, 0.3, 1.0))

    @property
    def basetexture2(self):
        return self._texture_property('$basetexture2', (0.3, 0.3, 0.0, 1.0))

    @property
    def basetexture3(self):
        return self._texture_property('$basetexture3', (0.3, 0.3, 0.0, 1.0))

    @property
    def basetexture4(self):
        return self._texture_property('$basetexture4', (0.3, 0.3, 0.0, 1.0))

    @property
    def detail(self):
        return self._texture_property('$detail', (0.3, 0.3, 0.0, 1.0))

    @property
    def ssbump(self):
        return self._bool_property('$ssbump')

    def _uvscale(self, key: str):
        """Per-layer ``$textureN_uvscale``, broadcast to a 3-component vector.

        CS:GO samples layer 1 with raw coords and has no ``$texture1_uvscale``
        uniform (``common_4wayblend_fxc.h`` scales only layers 2-4), but the key
        appears in real VMTs and other tools honour it, so it is applied anyway.
        """
        vector, _ = self._vmt.get_vector(key, None)
        if not vector:
            return vector
        vector = list(vector)
        return self.ensure_length(vector, 3, vector[0])

    @property
    def texture1_uvscale(self):
        return self._uvscale('$texture1_uvscale')

    @property
    def texture2_uvscale(self):
        return self._uvscale('$texture2_uvscale')

    @property
    def texture3_uvscale(self):
        return self._uvscale('$texture3_uvscale')

    @property
    def texture4_uvscale(self):
        return self._uvscale('$texture4_uvscale')

    @property
    def detailscale(self):
        return self._uvscale('$detailscale')

    def _normal(self, *keys: str):
        """First present normal map among ``keys``, decoded as a normal/ssbump."""
        for key in keys:
            if self._vmt.get_string(key, None):
                return self._texture_property(key, (0.5, 0.5, 1.0, 1.0),
                                              normal_map=True, ssbump=self.ssbump)
        return None

    @property
    def bumpmap(self):
        return self._normal('$bumpmap')

    @property
    def bumpmap2(self):
        return self._normal('$bumpmap2', '$basenormalmap2')

    @property
    def bumpmap3(self):
        return self._normal('$basenormalmap3')

    @property
    def bumpmap4(self):
        return self._normal('$basenormalmap4')

    def _bind_blend_weights(self, group: bpy.types.ShaderNodeGroup):
        """Point the group's Color Attribute node at the real multiblend layer.

        The bundled node group ships with ``layer_name = 'Col'``, a name the BSP
        importer never creates, so every weight reads as the fallback value and the
        four layers blend incorrectly. Retarget the node tree's own attribute
        lookup to :data:`MULTIBLEND_LAYER`.

        The node tree is shared between all materials using this shader, so this is
        a one-time fixup rather than per-material state.
        """
        node_tree = group.node_tree
        if node_tree is None:
            self.logger.error('4wayBlend node group is missing from the asset library')
            return
        retargeted = 0
        for node in node_tree.nodes:
            if node.type == 'VERTEX_COLOR' and node.layer_name != MULTIBLEND_LAYER:
                node.layer_name = MULTIBLEND_LAYER
                retargeted += 1
        if retargeted:
            self.logger.info(f'Retargeted {retargeted} blend-weight lookup(s) to {MULTIBLEND_LAYER!r}')

    def _add_uv_scale(self, texture_node, scale, name: str):
        """Feed ``texture_node`` from UVs multiplied by a per-layer ``$textureN_uvscale``."""
        if not scale:
            return
        uv = self.create_node(Nodes.ShaderNodeUVMap, name='UV Map', location=[-760, -700])
        scaler = self.create_node(Nodes.ShaderNodeVectorMath, name=name, location=[-580, -700])
        scaler.operation = 'MULTIPLY'
        scaler.inputs[1].default_value = scale
        self.connect_nodes(uv.outputs[0], scaler.inputs[0])
        self.connect_nodes(scaler.outputs[0], texture_node.inputs[0])

    def create_nodes(self, material:bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        self.do_arrange = True
        
        vars = [
            '$texture1_lumstart',
            '$texture1_lumend',
            '$texture2_lumstart',
            '$texture2_lumend',
            '$texture2_blendstart',
            '$texture2_blendend',
            '$lumblendfactor2',
            '$texture3_lumstart',
            '$texture3_lumend',
            '$texture3_blendstart',
            '$texture3_blendend',
            '$texture3_bumpblendfactor',
            #'$texture4_blendmode',
            '$texture4_lumstart',
            '$texture4_lumend',
            '$texture4_blendstart',
            '$texture4_blendend',
            '$texture4_bumpblendfactor',
            '$lumblendfactor3',
            '$lumblendfactor4',
            '$detailblendfactor',
            '$detailblendfactor2',
            '$detailblendfactor3',
            '$detailblendfactor4'
        ]

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)
        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        shader.inputs['Specular IOR Level' if is_blender_4() else 'Specular'].default_value = 0.0
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        Fway: bpy.types.ShaderNodeGroup
        Fway = self.create_node_group('4wayBlend')
        self._bind_blend_weights(Fway)
        self.connect_nodes(Fway.outputs['Albedo'], shader.inputs['Base Color'])
        normalMap = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(Fway.outputs['Normal'], normalMap.inputs['Color'])
        self.connect_nodes(Fway.outputs['Normal Strength'], normalMap.inputs['Strength'])
        self.connect_nodes(normalMap.outputs['Normal'], shader.inputs['Normal'])

        bases = [None, None, None, None]
        normals = [None, None]

        # $basetexture .. $basetexture4 -> the group's four albedo inputs.
        for idx, (image, socket, scale_var) in enumerate((
                (self.basetexture,  '$basetexture',  '$texture1_uvscale'),
                (self.basetexture2, '$basetexture2', '$texture2_uvscale'),
                (self.basetexture3, '$basetexture3', '$texture3_uvscale'),
                (self.basetexture4, '$basetexture4', '$texture4_uvscale'),
        )):
            if image is None:
                continue
            node = self.create_texture_node(image, f'$basetexture{idx + 1}')
            bases[idx] = node
            self.connect_nodes(node.outputs['Color'], Fway.inputs[socket])
            self._add_uv_scale(node, getattr(self, f'texture{idx + 1}_uvscale'), scale_var)

        if self.detail:
            detail_node = self.create_texture_node(self.detail, '$detail')
            self.connect_nodes(detail_node.outputs['Color'], Fway.inputs['$detail'])
            self._add_uv_scale(detail_node, self.detailscale, '$detailscale')

        # Only two bump slots exist on the group; $basenormalmap3/4 are parsed but
        # have nowhere to go, so their blend factors act on these two.
        for idx, (image, socket) in enumerate(((self.bumpmap, '$bumpmap'),
                                               (self.bumpmap2, '$bumpmap2'))):
            if image is None:
                continue
            node = self.create_texture_node(image, f'$bumpmap{idx + 1}')
            normals[idx] = node
            self.connect_nodes(node.outputs['Color'], Fway.inputs[socket])
            self._add_uv_scale(node, getattr(self, f'texture{idx + 1}_uvscale'),
                               f'$texture{idx + 1}_uvscale')

        for var in vars:
            value = self._vmt.get_float(var, 0)
            Fway.inputs[var].default_value = value
