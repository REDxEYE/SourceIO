from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import ExtraMaterialParameters, Nodes
from SourceIO.library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from .lightmap_generic import LightmapGeneric

#: Vertex-colour layer written by the BSP importer for displacements that carry a
#: LUMP_DISP_MULTIBLEND. ``m_vMultiBlend`` is ``(w1, w2, w3, w4)`` mapped straight
#: onto RGBA; the pixel shader reads only ``.g``/``.b``/``.a`` for layers 2/3/4 and
#: never samples ``.r``, because layer 1 is the base the lerp chain starts from.
MULTIBLEND_LAYER = 'alphablend'

#: ``Luminance()`` from ``common_fxc.h``, applied with a Vector Math dot product.
#:
#: Deliberately *not* ``ShaderNodeRGBToBW``, which is the obvious one-node
#: substitute but uses the Rec.709 luma weights -- measured as
#: (0.212639, 0.715169, 0.072193) in Blender 5.2. On a saturated texel such as
#: (0.5, 0.25, 0.75) the two disagree by 0.041, and since 54 of Black Mesa's 89
#: materials ramp between ``$texture1_lumstart 0.15`` and ``lumend 0.0``, that is
#: over a quarter of the entire blend ramp -- enough to visibly move where layers
#: transition. Both cost one node, so there is nothing to trade for it.
LUMINANCE_WEIGHTS = (0.30, 0.59, 0.11)

#: Tangent-space flat normal, used when a layer joins the bump chain but the layer
#: below it has no normal map to lerp from.
FLAT_NORMAL = (0.5, 0.5, 1.0, 1.0)

#: Softness of Blender's box projection, in UV units. Source weights its three
#: seamless projections by the squared world normal, which is a fairly wide blend;
#: this is the closest equivalent knob.
SEAMLESS_PROJECTION_BLEND = 0.25

#: Name of the custom property stamped on every generated group, and the revision it
#: records. Bump this whenever a group's interface or internals change.
#:
#: A group is looked up by name, so a stale one left in the .blend by an older addon
#: version would otherwise be reused with the wrong maths -- exactly how the old
#: bundled ``4wayBlend`` asset went wrong. Version mismatch rebuilds it in place, so
#: there is only ever one canonical definition.
GROUP_VERSION_KEY = 'sourceio_group_version'
GROUP_VERSION = 1

LUMINANCE_GROUP = 'SourceIO Source1 4Way Luminance'
BLEND_FACTOR_GROUP = 'SourceIO Source1 4Way Blend Factor'


def _smoothstep_edges(start: float, end: float) -> tuple[float, float]:
    """Map Range edges for ``smoothstep(start, end, x)``.

    Inverted ranges (``start > end``) are passed through untouched -- they are the
    common case, see :class:`Lightmapped4WayBlend`. Only a *degenerate* range is
    adjusted: Blender returns 0 for it, whereas HLSL's ``smoothstep`` divides by zero
    and saturates into a step function, so nudge the end to reproduce HLSL.
    """
    return (start, start + 1e-6) if start == end else (start, end)


def _new_socket(tree, name: str, in_out: str, socket_type: str = 'NodeSocketFloat',
                default=None):
    """Add a group interface socket, across the 4.0 interface API change."""
    if hasattr(tree, 'interface'):  # Blender 4.0+
        socket = tree.interface.new_socket(name, in_out=in_out, socket_type=socket_type)
    else:
        collection = tree.inputs if in_out == 'INPUT' else tree.outputs
        socket = collection.new(socket_type, name)
    if default is not None:
        socket.default_value = default
    return socket


def _clear_interface(tree):
    if hasattr(tree, 'interface'):
        tree.interface.clear()
    else:
        tree.inputs.clear()
        tree.outputs.clear()


def _build_luminance_group(tree):
    """``smoothstep($textureN_lumstart, $textureN_lumend, Luminance(color))``."""
    _new_socket(tree, 'Color', 'INPUT', 'NodeSocketColor', (0.5, 0.5, 0.5, 1.0))
    _new_socket(tree, 'Lum Start', 'INPUT', 'NodeSocketFloat', 0.0)
    _new_socket(tree, 'Lum End', 'INPUT', 'NodeSocketFloat', 1.0)
    _new_socket(tree, 'Luminance', 'OUTPUT', 'NodeSocketFloat')

    group_in = tree.nodes.new('NodeGroupInput')
    group_in.location = (-420, 0)
    group_out = tree.nodes.new('NodeGroupOutput')
    group_out.location = (220, 0)

    dot = tree.nodes.new(Nodes.ShaderNodeVectorMath)
    dot.operation = 'DOT_PRODUCT'
    dot.label = 'Luminance()'
    dot.location = (-220, 40)
    dot.inputs[1].default_value = LUMINANCE_WEIGHTS

    step = tree.nodes.new(Nodes.ShaderNodeMapRange)
    step.interpolation_type = 'SMOOTHSTEP'
    step.label = 'smoothstep'
    step.location = (0, 0)

    tree.links.new(group_in.outputs['Color'], dot.inputs[0])
    tree.links.new(dot.outputs['Value'], step.inputs['Value'])
    tree.links.new(group_in.outputs['Lum Start'], step.inputs['From Min'])
    tree.links.new(group_in.outputs['Lum End'], step.inputs['From Max'])
    tree.links.new(step.outputs[0], group_out.inputs['Luminance'])


def _build_blend_factor_group(tree):
    """``smoothstep(blendstart, blendend, w * (1 + lerp(1 - lumPrev, lum, f)))``.

    ``Lum Prev`` left unconnected with ``Lum Blend Factor`` at 1.0 collapses to
    ``smoothstep(blendstart, blendend, w * (1 + lum))``, which is what a layer whose
    predecessor is absent needs.
    """
    _new_socket(tree, 'Weight', 'INPUT', 'NodeSocketFloat', 0.0)
    _new_socket(tree, 'Lum Prev', 'INPUT', 'NodeSocketFloat', 0.0)
    _new_socket(tree, 'Lum', 'INPUT', 'NodeSocketFloat', 0.0)
    _new_socket(tree, 'Lum Blend Factor', 'INPUT', 'NodeSocketFloat', 1.0)
    _new_socket(tree, 'Blend Start', 'INPUT', 'NodeSocketFloat', 0.0)
    _new_socket(tree, 'Blend End', 'INPUT', 'NodeSocketFloat', 1.0)
    _new_socket(tree, 'Blend Factor', 'OUTPUT', 'NodeSocketFloat')

    group_in = tree.nodes.new('NodeGroupInput')
    group_in.location = (-720, 0)
    group_out = tree.nodes.new('NodeGroupOutput')
    group_out.location = (320, 0)

    invert = tree.nodes.new(Nodes.ShaderNodeMath)
    invert.operation = 'SUBTRACT'
    invert.label = '1 - lumPrev'
    invert.location = (-520, -120)
    invert.inputs[0].default_value = 1.0

    mix = tree.nodes.new(Nodes.ShaderNodeMix)
    mix.data_type = 'FLOAT'
    mix.label = 'lerp -> lum'
    mix.location = (-340, -60)

    one_plus = tree.nodes.new(Nodes.ShaderNodeMath)
    one_plus.operation = 'ADD'
    one_plus.label = '1 + lum'
    one_plus.location = (-160, -60)
    one_plus.inputs[1].default_value = 1.0

    gain = tree.nodes.new(Nodes.ShaderNodeMath)
    gain.operation = 'MULTIPLY'
    gain.label = 'w * (1 + lum)'
    gain.location = (20, 20)

    step = tree.nodes.new(Nodes.ShaderNodeMapRange)
    step.interpolation_type = 'SMOOTHSTEP'
    step.label = 'smoothstep'
    step.location = (160, 0)

    tree.links.new(group_in.outputs['Lum Prev'], invert.inputs[1])
    tree.links.new(invert.outputs[0], mix.inputs['A'])
    tree.links.new(group_in.outputs['Lum'], mix.inputs['B'])
    tree.links.new(group_in.outputs['Lum Blend Factor'], mix.inputs['Factor'])
    tree.links.new(mix.outputs[0], one_plus.inputs[0])
    tree.links.new(group_in.outputs['Weight'], gain.inputs[0])
    tree.links.new(one_plus.outputs[0], gain.inputs[1])
    tree.links.new(gain.outputs[0], step.inputs['Value'])
    tree.links.new(group_in.outputs['Blend Start'], step.inputs['From Min'])
    tree.links.new(group_in.outputs['Blend End'], step.inputs['From Max'])
    tree.links.new(step.outputs[0], group_out.inputs['Blend Factor'])


#: Group name -> builder. Both are generated in Python rather than shipped in the
#: asset .blend, so they cannot drift out of sync with the code that drives them.
GROUP_BUILDERS = {
    LUMINANCE_GROUP: _build_luminance_group,
    BLEND_FACTOR_GROUP: _build_blend_factor_group,
}


class Lightmapped4WayBlend(LightmapGeneric):
    """Lightmapped_4WayBlend -- four-texture displacement blend.

    Absent from Source SDK 2013; used heavily by Black Mesa's Xen chapters and
    present in CS:GO as ``lightmapped_4wayblend_ps20b.fxc``. It is a superset of
    LightmappedGeneric -- shipped materials rely on ``$envmap``,
    ``$basealphaenvmapmask``, ``$ssbump`` and ``$seamless_scale`` -- so this derives
    from :class:`LightmapGeneric` and only replaces the albedo and normal blends.

    Per layer, the vertex weight is *gained* by a luminance term and then
    smoothstepped, applied as a sequential lerp chain::

        lumN         = smoothstep($textureN_lumstart, $textureN_lumend, Luminance(baseColorN))
        lum          = lerp(1 - lumPrev, lumN, $lumblendfactorN)
        blendfactorN = smoothstep($textureN_blendstart, $textureN_blendend, w * (1 + lum))
        baseColor    = lerp(baseColor, baseColorN, blendfactorN)
        normal       = lerp(normal, normalN, blendfactorN * $textureN_bumpblendfactor)

    Note ``w * (1 + lum)`` -- a gain in ``[w, 2w]``, not a multiply. There is no
    ``$blendmodulatetexture``; the luminance system replaces it.

    ``lumStart > lumEnd`` is not a mistake, it is the common case: 54 of Black Mesa's
    89 materials set ``$texture1_lumstart 0.15`` / ``$texture1_lumend 0.0`` to make
    *darker* texels blend more eagerly. Map Range divides by
    ``from_max - from_min`` and clamps afterwards, so an inverted range inverts the
    curve for free -- nothing here may "fix up" the edges into ascending order.

    Earlier versions of this shader delegated to a bundled ``4wayBlend`` node group.
    That group is gone: it only had two bump slots for four layers, its blend curves
    did not match the SDK, and it carried a hardcoded ``Col`` attribute name that had
    to be patched at import time. Everything is built from primitives instead.
    """
    SHADER = 'lightmapped_4wayblend'

    #: Per-layer VMT keys. Layer 1 has luminance parameters (they feed layer 2's
    #: gain) but no blend range, no bump blend factor and no vertex weight of its
    #: own -- it is the base of the chain.
    #:
    #: ``$bumpmap2``/``$basenormalmap2`` are interchangeable spellings for layer 2's
    #: normal map; Black Mesa uses ``$basenormalmap2`` in 83 of 89 materials and
    #: ``$bumpmap2`` in 2.
    LAYERS = (
        ('$basetexture', ('$bumpmap',), '$texture1_uvscale'),
        ('$basetexture2', ('$bumpmap2', '$basenormalmap2'), '$texture2_uvscale'),
        ('$basetexture3', ('$basenormalmap3', '$bumpmap3'), '$texture3_uvscale'),
        ('$basetexture4', ('$basenormalmap4', '$bumpmap4'), '$texture4_uvscale'),
    )

    # ------------------------------------------------------------------ textures

    def _layer_basetexture(self, layer: int):
        key = self.LAYERS[layer][0]
        default = (0.3, 0.0, 0.3, 1.0) if layer == 0 else (0.3, 0.3, 0.0, 1.0)
        return self._texture_property(key, default)

    def _layer_normal_key(self, layer: int) -> str | None:
        """Which of the layer's interchangeable normal-map keys the VMT actually set."""
        for key in self.LAYERS[layer][1]:
            if self._vmt.get_string(key, None):
                return key
        return None

    def _layer_normal(self, layer: int):
        """Normal map for ``layer``, decoded as a normal or ssbump."""
        key = self._layer_normal_key(layer)
        if key is None:
            return None
        return self._texture_property(key, FLAT_NORMAL,
                                      normal_map=True, ssbump=self.ssbump)

    @property
    def basetexture(self):
        return self._layer_basetexture(0)

    @property
    def basetexture2(self):
        return self._layer_basetexture(1)

    @property
    def basetexture3(self):
        return self._layer_basetexture(2)

    @property
    def basetexture4(self):
        return self._layer_basetexture(3)

    @property
    def bumpmap(self):
        return self._layer_normal(0)

    @property
    def bumpmap2(self):
        return self._layer_normal(1)

    # ------------------------------------------------------------------- colours

    @property
    def color2(self):
        """``$color2`` -- diffuse modulation, applied after the blend and detail.

        Set by 84 of Black Mesa's 89 materials and routinely above 1.0
        (``[3.2 3.2 3.2]``), so the multiply must not clamp.
        """
        return self._color_property('$color2', None)

    # -------------------------------------------------------------------- floats

    def _layer_float(self, layer: int, suffix: str, default: float) -> float:
        return self._vmt.get_float(f'$texture{layer + 1}_{suffix}', default)

    def _lum_range(self, layer: int) -> tuple[float, float]:
        """``($textureN_lumstart, $textureN_lumend)``, inverted ranges preserved."""
        return (self._layer_float(layer, 'lumstart', 0.0),
                self._layer_float(layer, 'lumend', 1.0))

    def _blend_range(self, layer: int) -> tuple[float, float]:
        return (self._layer_float(layer, 'blendstart', 0.0),
                self._layer_float(layer, 'blendend', 1.0))

    def _lumblendfactor(self, layer: int) -> float:
        return self._vmt.get_float(f'$lumblendfactor{layer + 1}', 1.0)

    def _bumpblendfactor(self, layer: int) -> float:
        return self._layer_float(layer, 'bumpblendfactor', 1.0)

    def _detailblendfactor(self, layer: int) -> float:
        suffix = '' if layer == 0 else str(layer + 1)
        return self._vmt.get_float(f'$detailblendfactor{suffix}', 1.0)

    @property
    def _needs_blended_alpha(self) -> bool:
        """Whether anything downstream actually reads the blended base alpha.

        ``$basealphaenvmapmask`` is the only consumer, and it is the *last* of the
        three cubemap mask sources the SDK considers, so an earlier one winning means
        the alpha chain would be built and never sampled. Guessing wrong in the
        conservative direction is harmless: :meth:`_setup_envmap` falls back to layer
        1's own alpha.
        """
        if not (self.basealphaenvmapmask and self.envmap):
            return False
        if self._vmt.get_string('$envmapmask', None) is not None:
            return False
        return not self.normalmapalphaenvmapmask

    def _layer_uvscale(self, layer: int):
        """Per-layer ``$textureN_uvscale``, broadcast to a 3-component vector.

        CS:GO's ``common_4wayblend_fxc.h`` scales only layers 2-4 and has no
        ``$texture1_uvscale`` uniform, but the key appears in 8 shipped Black Mesa
        materials, so it is honoured for layer 1 too.
        """
        vector, _ = self._vmt.get_vector(self.LAYERS[layer][2], None)
        if not vector:
            return None
        vector = list(vector)
        if len(vector) == 1:
            vector = [vector[0], vector[0]]
        # Only x/y matter for a 2D lookup; leave the third component neutral so a
        # box-projected (seamless) layer is not squashed along the third axis.
        return self.ensure_length(vector, 3, 1.0)

    # ------------------------------------------------------------- coordinates

    def _build_uv_chain(self):
        """``$seamless_scale`` -- world-space projection, not a UV multiply.

        ``lightmappedgeneric_vs20.fxc`` does not touch the UVs at all in this mode::

            o.SeamlessTexCoord.xyz = SEAMLESS_SCALE * worldPos;

        and the pixel shader sums three planar lookups (``.zy``, ``.xz``, ``.xy``)
        weighted by the squared world normal. That is box projection, which Blender's
        Image Texture node implements natively, so feeding it scaled world position
        costs one node rather than three lookups per layer.

        Getting this wrong is very visible: every shipped value is around ``0.002``
        (85 of 89 materials), so multiplying UVs by it -- as the inherited
        implementation does -- collapses each texture to a single flat colour.
        """
        uv_node = self.create_node(Nodes.ShaderNodeUVMap)
        seamless_scale = self.seamless_scale
        if not seamless_scale:
            return uv_node, uv_node.outputs[0]

        self._seamless = True
        geometry = self.create_node(Nodes.ShaderNodeNewGeometry, 'world position')
        scale_node = self.create_node(Nodes.ShaderNodeVectorMath, '$seamless_scale')
        scale_node.operation = 'MULTIPLY'
        # Geometry.Position is in Blender metres; the SDK works in Hammer units. Y is
        # negated for the same reason as in the water shader: the importer mirrors
        # that axis when converting Source coordinates.
        factor = seamless_scale / SOURCE1_HAMMER_UNIT_TO_METERS
        scale_node.inputs[1].default_value = (factor, -factor, factor)
        self.connect_nodes(geometry.outputs['Position'], scale_node.inputs[0])
        return uv_node, scale_node.outputs[0]

    def _layer_uv_socket(self, layer: int, uv_out):
        """Coordinates for one layer, ``$textureN_uvscale`` applied once.

        Order matches the SDK: the shared transform/seamless chain first, then the
        per-layer scale on top of it. Cached because a layer's base texture and its
        normal map share the same coordinates -- building a scaler per *texture*
        rather than per *layer* duplicated up to four identical nodes.
        """
        if layer in self._layer_uv_cache:
            return self._layer_uv_cache[layer]

        scale = self._layer_uvscale(layer)
        if scale is None:
            socket = uv_out
        else:
            scaler = self.create_node(Nodes.ShaderNodeVectorMath, self.LAYERS[layer][2])
            scaler.operation = 'MULTIPLY'
            scaler.inputs[1].default_value = scale
            self.connect_nodes(uv_out, scaler.inputs[0])
            socket = scaler.outputs[0]
        self._layer_uv_cache[layer] = socket
        return socket

    def _layer_texture_node(self, image, name: str, layer: int, uv_out):
        """Texture node for one layer, fed by that layer's shared coordinates."""
        node = self.create_texture_node(image, name)
        if self._seamless:
            node.projection = 'BOX'
            node.projection_blend = SEAMLESS_PROJECTION_BLEND
        self.connect_nodes(self._layer_uv_socket(layer, uv_out), node.inputs[0])
        return node

    # -------------------------------------------------------------- node groups

    @staticmethod
    def _ensure_node_group(name: str):
        """Return the generated group ``name``, building or refreshing it as needed."""
        tree = bpy.data.node_groups.get(name)
        if tree is not None and tree.get(GROUP_VERSION_KEY) == GROUP_VERSION:
            return tree
        if tree is None:
            tree = bpy.data.node_groups.new(name, 'ShaderNodeTree')
        else:
            # Older revision (or something else claiming the name): rebuild in place
            # so every material ends up on the current definition.
            tree.nodes.clear()
            _clear_interface(tree)
        GROUP_BUILDERS[name](tree)
        tree[GROUP_VERSION_KEY] = GROUP_VERSION
        return tree

    def _group(self, name: str, label: str):
        group = self.create_node(Nodes.ShaderNodeGroup, label)
        group.node_tree = self._ensure_node_group(name)
        group.width = 200
        return group

    # ------------------------------------------------------------- blend factors

    def _layer_luminance(self, color_socket, layer: int):
        """One ``Luminance`` group instance for ``layer``."""
        group = self._group(LUMINANCE_GROUP, f'$texture{layer + 1}_lum')
        start, end = _smoothstep_edges(*self._lum_range(layer))
        self.connect_nodes(color_socket, group.inputs['Color'])
        group.inputs['Lum Start'].default_value = start
        group.inputs['Lum End'].default_value = end
        return group.outputs['Luminance']

    def _weight_sockets(self):
        """Vertex weights for layers 2-4: ``multiblend`` ``.g``, ``.b``, ``.a``."""
        vertex_color = self.create_node(Nodes.ShaderNodeVertexColor, 'multiblend weights')
        vertex_color.layer_name = MULTIBLEND_LAYER
        split = self.create_node(Nodes.ShaderNodeSeparateColor, 'multiblend split')
        split.mode = 'RGB'
        self.connect_nodes(vertex_color.outputs['Color'], split.inputs[0])
        return (split.outputs[1], split.outputs[2], vertex_color.outputs['Alpha'])

    def _build_blend_factors(self, lum_sockets, weight_sockets, present):
        """``blendfactorN`` for layers 2-4, or None where the layer is absent.

        One :data:`BLEND_FACTOR_GROUP` instance per layer. ``lum_sockets[layer]`` is
        never None here -- a layer with no base texture is skipped outright, and the
        luminance is derived from that same texture -- so only the *previous* layer's
        term can be missing.
        """
        factors = [None, None, None]
        for layer in (1, 2, 3):
            if not present[layer]:
                continue
            group = self._group(BLEND_FACTOR_GROUP, f'$texture{layer + 1}_blend')
            self.connect_nodes(weight_sockets[layer - 1], group.inputs['Weight'])
            self.connect_nodes(lum_sockets[layer], group.inputs['Lum'])

            lum_prev = lum_sockets[layer - 1]
            if lum_prev is not None:
                self.connect_nodes(lum_prev, group.inputs['Lum Prev'])
                group.inputs['Lum Blend Factor'].default_value = self._lumblendfactor(layer)
            else:
                # No predecessor to blend against: lerp(_, lum, 1.0) == lum, which
                # drops the `1 - lumPrev` term exactly as the ungrouped chain did.
                group.inputs['Lum Blend Factor'].default_value = 1.0

            start, end = _smoothstep_edges(*self._blend_range(layer))
            group.inputs['Blend Start'].default_value = start
            group.inputs['Blend End'].default_value = end
            factors[layer - 1] = group.outputs['Blend Factor']
        return factors

    # ---------------------------------------------------------------- node tree

    def create_nodes(self, material: bpy.types.Material,
                     extra_parameters: dict[ExtraMaterialParameters, Any]):
        self._seamless = False
        #: blendfactor for layers 2-4, shared by the albedo, alpha and bump chains.
        self._blend_factors: list = [None, None, None]
        #: layer index -> coordinate socket, so a layer's base texture and normal map
        #: share one ``$textureN_uvscale`` node instead of building one each.
        self._layer_uv_cache: dict = {}
        self._blended_alpha_output = None
        self._warn_unsupported()
        result = super().create_nodes(material, extra_parameters)
        self._prune_unused_uv_node()
        return result

    def _prune_unused_uv_node(self):
        """Drop the shared UVMap node when nothing ended up using it.

        In seamless mode the layers are driven by world position, so the UVMap node
        is only there for ``$detail`` and ``$*transform``. Most materials are
        seamless and have no detail texture, which would otherwise leave a stray
        node in every one of them.
        """
        for node in list(self.bpy_material.node_tree.nodes):
            if node.type == 'UVMAP' and not any(out.links for out in node.outputs):
                self.bpy_material.node_tree.nodes.remove(node)

    def _warn_unsupported(self):
        """Report per-layer modes this translation does not model.

        ``$textureN_blendmode`` is 0 in every shipped material, so its non-zero
        behaviour is unknown rather than merely unimplemented -- worth a line in the
        log if one ever turns up.
        """
        for layer in range(4):
            mode = self._vmt.get_int(f'$texture{layer + 1}_blendmode', 0)
            if mode:
                self.logger.warn(f'$texture{layer + 1}_blendmode {mode} is not supported; '
                                 'blending layer normally')

    def _build_blended_albedo(self, shader, basetexture, basetexture2, uv_node, uv_out):
        """The four-layer albedo chain. Returns ``(layer1_node, albedo, blendfactor)``."""
        images = [basetexture, basetexture2, self.basetexture3, self.basetexture4]
        present = [image is not None for image in images]

        nodes = []
        for layer, image in enumerate(images):
            if image is None:
                nodes.append(None)
                continue
            nodes.append(self._layer_texture_node(image, self.LAYERS[layer][0],
                                                  layer, uv_out))

        # All four luminance terms come from the raw layer samples, not from the
        # partially blended result, because $textureN_lumstart/lumend describe that
        # one texture.
        lum_sockets = [None if node is None
                       else self._layer_luminance(node.outputs['Color'], layer)
                       for layer, node in enumerate(nodes)]

        self._blend_factors = self._build_blend_factors(lum_sockets, self._weight_sockets(),
                                                        present)

        # $basealphaenvmapmask masks the cubemap with `1 - blendedAlpha`, so when it is
        # in play the alpha has to follow the same chain as the colour.
        blend_alpha = self._needs_blended_alpha
        albedo_output = nodes[0].outputs['Color']
        alpha_output = nodes[0].outputs['Alpha']
        for layer in (1, 2, 3):
            factor = self._blend_factors[layer - 1]
            if factor is None:
                continue
            mix = self.create_node(Nodes.ShaderNodeMixRGB, f'$basetexture{layer + 1} blend')
            mix.blend_type = 'MIX'
            self.connect_nodes(factor, mix.inputs['Fac'])
            self.connect_nodes(albedo_output, mix.inputs['Color1'])
            self.connect_nodes(nodes[layer].outputs['Color'], mix.inputs['Color2'])
            albedo_output = mix.outputs['Color']

            if blend_alpha:
                alpha_mix = self.create_node(Nodes.ShaderNodeMix,
                                             f'$basetexture{layer + 1} alpha')
                alpha_mix.data_type = 'FLOAT'
                self.connect_nodes(factor, alpha_mix.inputs['Factor'])
                self.connect_nodes(alpha_output, alpha_mix.inputs['A'])
                self.connect_nodes(nodes[layer].outputs['Alpha'], alpha_mix.inputs['B'])
                alpha_output = alpha_mix.outputs[0]

        if blend_alpha:
            self._blended_alpha_output = alpha_output

        albedo_output = self._build_detail(albedo_output, uv_node)
        albedo_output = self._apply_color2(albedo_output)
        self.connect_nodes(albedo_output, shader.inputs['Base Color'])
        return nodes[0], albedo_output, self._blend_factors[0]

    def _mix_float(self, previous, value: float, factor_socket, name: str):
        """``lerp(previous, value, factor)`` where ``previous`` is a socket or float."""
        mix = self.create_node(Nodes.ShaderNodeMix, name)
        mix.data_type = 'FLOAT'
        self.connect_nodes(factor_socket, mix.inputs['Factor'])
        if isinstance(previous, float):
            mix.inputs['A'].default_value = previous
        else:
            self.connect_nodes(previous, mix.inputs['A'])
        mix.inputs['B'].default_value = value
        return mix.outputs[0]

    def _build_detail(self, albedo_output, uv_node):
        """``$detail`` with a per-layer blend factor.

        Each layer carries its own ``$detailblendfactorN``, blended along the same
        chain as the colour, so detail fades in and out with the layers. Applied as
        mod2x (``$detailblendmode`` 0, the default)::

            baseColor *= lerp( 1.0, 2.0 * detailColor, fBlendFactor )

        No shipped material sets ``$detailblendmode``, so the other TCOMBINE modes
        are not wired up here; one turning up is logged rather than guessed at.
        """
        detail_image = self.detail
        if detail_image is None:
            return albedo_output

        mode = self.detailmode
        if mode:
            self.logger.warn(f'$detailblendmode {mode} is not supported by '
                             f'{self.SHADER}; applying mod2x detail instead')

        factors = [self._detailblendfactor(layer) for layer in range(4)]
        factor = factors[0]
        for layer in (1, 2, 3):
            blend_factor = self._blend_factors[layer - 1]
            if blend_factor is None or factors[layer] == factor:
                continue
            factor = self._mix_float(factor, factors[layer], blend_factor,
                                     f'$detailblendfactor{layer + 1}')

        detail_node = self.create_texture_node(detail_image, '$detail')
        self._build_detail_uv(detail_node, self.detailscale,
                              self.detailtexturetransform, uv_node)

        # 2.0 * detailColor
        mod2x = self.create_node(Nodes.ShaderNodeMixRGB, 'detail mod2x')
        mod2x.blend_type = 'MULTIPLY'
        mod2x.inputs['Fac'].default_value = 1.0
        self.connect_nodes(detail_node.outputs['Color'], mod2x.inputs['Color1'])
        mod2x.inputs['Color2'].default_value = (2.0, 2.0, 2.0, 1.0)

        # lerp( 1.0, that, fBlendFactor )
        strength = self.create_node(Nodes.ShaderNodeMixRGB, '$detailblendfactor')
        strength.blend_type = 'MIX'
        if isinstance(factor, float):
            strength.inputs['Fac'].default_value = factor
        else:
            self.connect_nodes(factor, strength.inputs['Fac'])
        strength.inputs['Color1'].default_value = (1.0, 1.0, 1.0, 1.0)
        self.connect_nodes(mod2x.outputs['Color'], strength.inputs['Color2'])

        combine = self.create_node(Nodes.ShaderNodeMixRGB, 'DetailBlend')
        combine.blend_type = 'MULTIPLY'
        combine.inputs['Fac'].default_value = 1.0
        self.connect_nodes(albedo_output, combine.inputs['Color1'])
        self.connect_nodes(strength.outputs['Color'], combine.inputs['Color2'])
        return combine.outputs['Color']

    def _apply_color2(self, albedo_output):
        """``baseColor *= $color2`` -- diffuse modulation, values may exceed 1."""
        color2 = self.color2
        if color2 is None or tuple(color2[:3]) == (1.0, 1.0, 1.0):
            return albedo_output
        modulate = self.create_node(Nodes.ShaderNodeMixRGB, '$color2')
        modulate.blend_type = 'MULTIPLY'
        modulate.inputs['Fac'].default_value = 1.0
        self.connect_nodes(albedo_output, modulate.inputs['Color1'])
        modulate.inputs['Color2'].default_value = color2
        return modulate.outputs['Color']

    def _setup_normals(self, shader, uv_node, uv_out, blend_output):
        """The four-layer normal chain, scaled by ``$textureN_bumpblendfactor``.

        Lerping the encoded tangent-space RGB and decoding once is equivalent to
        lerping the unpacked vectors, since the mapping is affine.
        """
        # Resolve the key once: it decides whether the layer participates *and*
        # labels the node with the spelling the VMT actually used.
        keys = [self._layer_normal_key(layer) for layer in range(4)]
        if not any(keys):
            return None

        layer1_node = None
        normal_source = None
        for layer, key in enumerate(keys):
            if key is None:
                continue
            image = self._texture_property(key, FLAT_NORMAL,
                                           normal_map=True, ssbump=self.ssbump)
            node = self._layer_texture_node(image, key, layer, uv_out)
            if layer == 0:
                layer1_node = node
                normal_source = node.outputs['Color']
                continue

            factor = self._blend_factors[layer - 1]
            if factor is None:
                # No albedo for this layer, so there is no weight to blend its
                # normal with; the lowest present normal map wins outright.
                if normal_source is None:
                    normal_source = node.outputs['Color']
                continue

            mix = self.create_node(Nodes.ShaderNodeMixRGB, f'layer{layer + 1} normal blend')
            mix.blend_type = 'MIX'
            bump_factor = self._bumpblendfactor(layer)
            if bump_factor == 1.0:
                self.connect_nodes(factor, mix.inputs['Fac'])
            else:
                scaled = self.create_node(Nodes.ShaderNodeMath,
                                          f'$texture{layer + 1}_bumpblendfactor')
                scaled.operation = 'MULTIPLY'
                scaled.inputs[1].default_value = bump_factor
                self.connect_nodes(factor, scaled.inputs[0])
                self.connect_nodes(scaled.outputs[0], mix.inputs['Fac'])
            if normal_source is None:
                # Nothing below this layer supplies a normal; lerp up from flat.
                mix.inputs['Color1'].default_value = FLAT_NORMAL
            else:
                self.connect_nodes(normal_source, mix.inputs['Color1'])
            self.connect_nodes(node.outputs['Color'], mix.inputs['Color2'])
            normal_source = mix.outputs['Color']

        normal_map = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(normal_source, normal_map.inputs['Color'])
        self.connect_nodes(normal_map.outputs['Normal'], shader.inputs['Normal'])
        return layer1_node


class BMSLightmapped4WayBlend(Lightmapped4WayBlend):
    """Black Mesa's own build of the shader; same parameters, different name.

    Materials spell it both ways -- ``Lightmapped_4WayBlend`` in 85 of them and
    ``BMS_Lightmapped_4WayBlend`` in 3 -- and shader names are matched lowercased, so
    only the prefixed spelling needs its own entry.
    """
    SHADER = 'bms_lightmapped_4wayblend'
