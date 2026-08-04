import math
from typing import Any

import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes, ExtraMaterialParameters
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4
from SourceIO.blender_bindings.utils.texture_utils import setup_image_sequence_node
from SourceIO.library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS

#: Water's index of refraction. The SDK hardcodes the air/water Fresnel term
#: rather than exposing it, and 1.333 is the physical value Blender expects.
WATER_IOR = 1.333


class Water(Source1ShaderBase):
    """water -- reflective/refractive liquid surfaces.

    Grounded in ``water.cpp`` and ``water_ps2x_helper.h``. Above water, the pixel
    shader mixes a refracted and a reflected sample by a Fresnel term, then fogs
    the refracted half by depth::

        fNdotV   = saturate( dot( vEyeVect, vNormal ) );
        fFresnel = pow( 1.0 - fNdotV, 5 );                       // Schlick
        vRefractColor = lerp( vRefractColor, waterFogColor,
                              saturate( waterFogDepthValue - 0.05 ) );
        result   = lerp( vRefractColor, vReflectColor, fFresnel );

    Blender's Principled BSDF already derives Schlick Fresnel from ``IOR``, and
    EEVEE's screen-space refraction with a Transmission Weight of 1.0 reproduces
    the refract/reflect split, so those terms are configured rather than rebuilt
    out of nodes. ``$fogcolor`` becomes the underwater colour -- it is what the
    surface turns into with depth, not a surface tint.

    Animation comes from the material proxies, not from the shader:

    * ``AnimatedTexture`` plays a multi-frame ``$normalmap`` (HL2's
      ``dev/water_normal`` has 29 frames) at ``$animatedtextureframerate``.
    * ``TextureScroll`` slides ``$bumptransform`` at ``$texturescrollrate`` along
      ``$texturescrollangle``.
    * ``$scroll1``/``$scroll2`` are read by the shader itself, which samples the
      normal map three times at time-scaled offsets and averages them::

          vNormal = 0.33 * ( vNormal + vNormal1 + vNormal2 );

    * ``$flowmap`` (Portal 2 and CS:GO) pushes two counter-phased normal samples
      along a painted flow direction.

    Every time-varying term hangs off one driven Value node holding Source's
    ``curtime`` (``frame / fps``), with the arithmetic done in Math nodes. That
    keeps the SDK's formulae recognisable, keeps them correct at any scene frame
    rate, and keeps the driver itself to a bare multiply -- see :meth:`_time_output`
    for why that last part matters.

    Only the above-water side is translated. ``$bottommaterial`` names a separate
    material for the underwater view, and the two coincident faces z-fight if both
    are imported.
    """
    SHADER: str = 'water'

    # ------------------------------------------------------------------ textures

    @property
    def normalmap(self):
        return self._texture_property('$normalmap', (0.5, 0.5, 1.0, 1.0), normal_map=True)

    @property
    def basetexture(self):
        return self._texture_property('$basetexture', (0.3, 0.0, 0.3, 1.0))

    @property
    def flowmap(self):
        return self._texture_property('$flowmap', (0.5, 0.5, 0.0, 1.0), is_data=True)

    # -------------------------------------------------------------------- colours

    @property
    def fogcolor(self):
        return self._color_property('$fogcolor', [0.2, 0.3, 0.4])

    @property
    def reflecttint(self):
        return self._color_property('$reflecttint', [1, 1, 1])

    @property
    def refracttint(self):
        return self._color_property('$refracttint', [1, 1, 1])

    # --------------------------------------------------------------------- floats

    @property
    def fogstart(self):
        return self._vmt.get_float('$fogstart', 0.0)

    @property
    def fogend(self):
        return self._vmt.get_float('$fogend', 100.0)

    @property
    def reflectamount(self):
        return self._vmt.get_float('$reflectamount', 0.8)

    @property
    def refractamount(self):
        return self._vmt.get_float('$refractamount', 0.2)

    @property
    def bluramount(self):
        return self._vmt.get_float('$bluramount', 0.0)

    # Flow-map parameters, with the SDK's InitFloatParam defaults.
    @property
    def flow_worlduvscale(self):
        return self._vmt.get_float('$flow_worlduvscale', 1.0)

    @property
    def flow_normaluvscale(self):
        return self._vmt.get_float('$flow_normaluvscale', 1.0)

    @property
    def flow_timeintervalinseconds(self):
        return self._vmt.get_float('$flow_timeintervalinseconds', 0.4)

    @property
    def flow_uvscrolldistance(self):
        return self._vmt.get_float('$flow_uvscrolldistance', 0.2)

    @property
    def flow_bumpstrength(self):
        return self._vmt.get_float('$flow_bumpstrength', 1.0)

    # ---------------------------------------------------------------------- flags

    @property
    def abovewater(self):
        # The SDK warns and defaults to 1 when it is missing.
        return self._vmt.get_int('$abovewater', 1) != 0

    @property
    def fogenable(self):
        return self._vmt.get_int('$fogenable', 1) != 0

    # ----------------------------------------------------------------- transforms

    _DEFAULT_TRANSFORM = {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1),
                          'rotate': (0, 0, 0), 'translate': (0, 0, 0)}

    @property
    def bumptransform(self):
        return self._vmt.get_transform_matrix('$bumptransform', dict(self._DEFAULT_TRANSFORM))

    @property
    def scroll1(self):
        value, _ = self._vmt.get_vector('$scroll1', None)
        return value

    @property
    def scroll2(self):
        value, _ = self._vmt.get_vector('$scroll2', None)
        return value

    # -------------------------------------------------------------------- proxies

    def _proxy(self, name: str):
        """Return a named proxy's parameter block, or None."""
        proxies = self._vmt.get('proxies', None)
        if not proxies:
            return None
        for proxy_name, proxy_data in proxies.items():
            if proxy_name.lower() == name and hasattr(proxy_data, 'get'):
                return proxy_data
        return None

    @staticmethod
    def _proxy_float(proxy, key: str, default: float) -> float:
        try:
            return float(proxy.get(key, default))
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------ node tree

    def create_nodes(self, material: bpy.types.Material, extra_parameters: dict[ExtraMaterialParameters, Any]):
        # Water writes depth and is opaque to the rasterizer; see-through comes
        # from screen-space refraction, not from alpha blending.
        self.set_blend_mode('OPAQUE')
        self.bpy_material.use_screen_refraction = True
        self.bpy_material.use_backface_culling = True
        if hasattr(self.bpy_material, 'use_raytrace_refraction'):
            self.bpy_material.use_raytrace_refraction = True

        material_output = self.create_node(Nodes.ShaderNodeOutputMaterial)

        if self.use_bvlg_status and bpy.data.node_groups.get("Water") is not None:
            self._create_bvlg_nodes(material_output)
            return

        shader = self.create_node(Nodes.ShaderNodeBsdfPrincipled, self.SHADER)
        self.connect_nodes(shader.outputs['BSDF'], material_output.inputs['Surface'])
        self._setup_surface(shader)
        self._setup_normals(shader)

    def _create_bvlg_nodes(self, material_output):
        """Wire the bundled ``Water`` node group (a glossy/refraction mix)."""
        group_node = self.create_node(Nodes.ShaderNodeGroup, self.SHADER)
        group_node.node_tree = bpy.data.node_groups.get("Water")
        group_node.width = group_node.bl_width_max
        self.connect_nodes(group_node.outputs['BSDF'], material_output.inputs['Surface'])

        normalmap_node = self._create_normal_texture_node()
        if normalmap_node is not None:
            # The group takes a raw normal colour, so the flow/scroll chains feed
            # into it exactly as they would feed a Normal Map node.
            self.connect_nodes(self._animated_normal_source(normalmap_node),
                               group_node.inputs['$normalmap'])

        for socket_name, value in (('$reflecttint', self.reflecttint),
                                   ('$refracttint', self.refracttint)):
            if value is not None and socket_name in group_node.inputs:
                group_node.inputs[socket_name].default_value = value
        if '$bluramount' in group_node.inputs:
            group_node.inputs['$bluramount'].default_value = self.bluramount

    def _setup_surface(self, shader):
        """Translate the reflect/refract/fog terms into Principled inputs."""
        # Schlick Fresnel in the shader == Principled's own IOR-driven Fresnel.
        shader.inputs['IOR'].default_value = WATER_IOR

        # $fogcolor is what the water becomes with depth. The SDK lerps the
        # refracted sample all the way to it over `saturate(depth - 0.05)`, so it
        # is a replacement colour rather than a tint over the surface.
        fogcolor = self.fogcolor
        shader.inputs['Base Color'].default_value = fogcolor if fogcolor is not None else (1.0, 1.0, 1.0, 1.0)
        if self.fogenable and self.fogend > 0.0:
            # Depth-dependent absorption needs the importer's world scale, which a
            # material has no access to; record the authored distances instead.
            shader['water_fog_start'] = self.fogstart
            shader['water_fog_end'] = self.fogend
            shader['water_fog_end_meters'] = self.fogend * SOURCE1_HAMMER_UNIT_TO_METERS

        # $reflectamount/$refractamount scale how far the normal perturbs each
        # dependent lookup -- i.e. how choppy the surface reads. Blender has no
        # equivalent, so they drive roughness: heavier distortion blurs both the
        # reflection and the refraction. $bluramount wins when it is set, since
        # that is explicitly a blur request ($BLURREFRACT in the SDK).
        if self.bluramount:
            roughness = self.bluramount
        else:
            roughness = 0.02 + 0.1 * max(self.reflectamount, self.refractamount)
        shader.inputs['Roughness'].default_value = self.clamp_value(roughness)

        transmission_input = 'Transmission Weight' if is_blender_4() else 'Transmission'
        if transmission_input in shader.inputs:
            shader.inputs[transmission_input].default_value = 1.0

        reflecttint = self.reflecttint
        if (reflecttint is not None and tuple(reflecttint[:3]) != (1.0, 1.0, 1.0)
                and 'Specular Tint' in shader.inputs):
            shader.inputs['Specular Tint'].default_value = reflecttint

    def _create_normal_texture_node(self):
        """Create the ``$normalmap`` node, animated when the VTF has frames."""
        animated = self._proxy('animatedtexture')
        frame_rate = 0.0
        image = None
        frame_count = 1

        if animated is not None:
            # AnimatedTexture drives whichever var it names; $normalmap is the only
            # one with a texture node here, so ignore any other target.
            target = str(animated.get('animatedtexturevar', '$normalmap')).lower()
            if target == '$normalmap':
                frame_rate = self._proxy_float(animated, 'animatedtextureframerate', 30.0)
                image, frame_count = self.load_animated_texture('$normalmap', (0.5, 0.5, 1.0, 1.0),
                                                                normal_map=True)
        if image is None:
            image = self.normalmap
            frame_count = 1
        if image is None:
            return None

        normalmap_node = self.create_texture_node(image, '$normalmap')
        if frame_count > 1:
            setup_image_sequence_node(normalmap_node, frame_count, frame_rate)
            self.logger.info(f'Animating $normalmap over {frame_count} frames at {frame_rate} fps')
        self._setup_bump_uv(normalmap_node)
        return normalmap_node

    def _animated_normal_source(self, normalmap_node):
        """Socket carrying the final normal colour, animated per the VMT's family."""
        if self.flowmap is not None:
            return self._build_flow_normal(normalmap_node)
        scroll1 = self.scroll1
        if scroll1 and abs(scroll1[0]) > 0.0:
            # MULTITEXTURE: the shader averages three time-offset samples.
            return self._build_multitexture_normal(normalmap_node)
        return normalmap_node.outputs['Color']

    def _setup_normals(self, shader):
        normalmap_node = self._create_normal_texture_node()
        if normalmap_node is None:
            return

        normal_map = self.create_node(Nodes.ShaderNodeNormalMap)
        self.connect_nodes(self._animated_normal_source(normalmap_node), normal_map.inputs['Color'])
        self.connect_nodes(normal_map.outputs['Normal'], shader.inputs['Normal'])
        normal_map.inputs['Strength'].default_value = self.clamp_value(
            self.flow_bumpstrength if self.flowmap is not None else 1.0, 0.0, 10.0)

    def _build_multitexture_normal(self, normalmap_node):
        """``$scroll1``/``$scroll2``: average three independently scrolled samples.

        ``water_ps2x_helper.h`` samples one normal map at three sets of coordinates
        and averages them, which is what keeps a scrolling texture from reading as
        a single sliding sheet::

            vNormal = 0.33 * ( vNormal + vNormal1 + vNormal2 );

        The extra two coordinate sets come from the vertex shader, rotated 45
        degrees and scaled differently from the first (``Water_vs20.fxc``)::

            o.vExtraBumpTexCoord.x = (u + v) * 0.1  + TexOffsets.x;
            o.vExtraBumpTexCoord.y = (v - u) * 0.1  + TexOffsets.y;
            o.vExtraBumpTexCoord.z = v * 0.45 + TexOffsets.z;
            o.vExtraBumpTexCoord.w = u * 0.45 + TexOffsets.w;

        where ``TexOffsets`` is ``curtime * $scroll1`` / ``curtime * $scroll2``.
        """
        image = normalmap_node.image
        scroll1 = self.scroll1 or (0.0, 0.0)
        scroll2 = self.scroll2 or (0.0, 0.0)

        def scrolled_layer(name: str, rotation: float, scale: tuple[float, float], scroll):
            texture = self.create_texture_node(image, name)
            mapping = self.create_node(Nodes.ShaderNodeMapping, f'{name} mapping')
            mapping.inputs['Rotation'].default_value = (0.0, 0.0, math.radians(rotation))
            mapping.inputs['Scale'].default_value = (scale[0], scale[1], 1.0)
            uv_node = self.create_node(Nodes.ShaderNodeUVMap)
            self.connect_nodes(uv_node.outputs[0], mapping.inputs['Vector'])
            self.connect_nodes(mapping.outputs['Vector'], texture.inputs[0])
            self._scroll_uv(mapping, scroll[0], scroll[1] if len(scroll) > 1 else scroll[0])
            return texture

        # Layer 2's (u+v, v-u) is a 45-degree rotation at 0.1 scale.
        layer2 = scrolled_layer('$normalmap scroll1', 45.0, (0.1, 0.1), scroll1)
        # Layer 3's (v, u) swaps the axes -- a reflection, not a rotation. Blender's
        # Mapping node applies Scale before Rotation, so scaling y negative and then
        # rotating 90 degrees yields (v, u): (u, -v) -> (v, u).
        layer3 = scrolled_layer('$normalmap scroll2', 90.0, (0.45, -0.45), scroll2)

        # 0.33 * (n0 + n1 + n2)
        add1 = self.create_node(Nodes.ShaderNodeVectorMath, 'scroll sum 1')
        add1.operation = 'ADD'
        self.connect_nodes(normalmap_node.outputs['Color'], add1.inputs[0])
        self.connect_nodes(layer2.outputs['Color'], add1.inputs[1])

        add2 = self.create_node(Nodes.ShaderNodeVectorMath, 'scroll sum 2')
        add2.operation = 'ADD'
        self.connect_nodes(add1.outputs[0], add2.inputs[0])
        self.connect_nodes(layer3.outputs['Color'], add2.inputs[1])

        average = self.create_node(Nodes.ShaderNodeVectorMath, 'scroll average')
        average.operation = 'SCALE'
        average.inputs['Scale'].default_value = 0.33
        self.connect_nodes(add2.outputs[0], average.inputs[0])
        return average.outputs[0]

    def _build_flow_normal(self, normalmap_node):
        """``$flowmap``: two counter-phased normal samples pushed along the flow.

        Absent from Source SDK 2013 but present in CS:GO's ``water_ps2x.fxc``,
        which Portal 2's water uses for every liquid. Per pixel::

            vFlowVectorTs      = ( vFlowTexel.rg * 2.0 ) - 1.0;
            flTimeInIntervals  = ( g_flTime / ( $flow_timeintervalinseconds * 2 ) ) + flNoise;
            flScrollTime1      = frac( flTimeInIntervals );
            flScrollTime2      = frac( flTimeInIntervals + 0.5 );   // half an interval out of phase
            vFlowUv1 = vUv + flOffset1 + flScrollTime1 * ( $flow_uvscrolldistance * vFlowVectorTs );
            vFlowUv2 = vUv + flOffset2 + flScrollTime2 * ( $flow_uvscrolldistance * vFlowVectorTs );
            flWeight2 = abs( 2 * frac( flTimeInIntervals ) - 1 );
            vNormal.xy = lerp( vNormal1.xy, vNormal2.xy, flWeight2 );

        Each sample slides along the flow vector for one interval, then is
        cross-faded out while the other takes over -- so the water flows along the
        painted direction without any layer stretching indefinitely. The
        cross-fade weight is triangular and in phase with the scroll, so both are
        keyframed over one full interval and cycled.

        Note the two samplers take *different* coordinates, which is easy to get
        wrong: the flow map is looked up with the surface's own texture UVs, while
        the normal layers are looked up in **world space**. Feeding the normal
        layers' heavily-scaled UVs to the flow map instead tiles it wrongly (visible
        seams) and shrinks the normal UVs so far that the scroll offset dwarfs them.

        The per-pixel ``$flow_noise_texture`` offset (which decorrelates the phase
        spatially so the whole surface does not pulse in unison) is dropped: it
        would need the scroll offset to vary per pixel, and Blender's Mapping node
        applies a single uniform translation.
        """
        image = normalmap_node.image
        flowmap = self.flowmap
        # water.cpp passes the reciprocals of both scales to the shader, so a larger
        # authored value means a smaller multiplier.
        world_uv_scale = 1.0 / self.flow_worlduvscale if self.flow_worlduvscale else 1.0
        normal_uv_scale = 1.0 / self.flow_normaluvscale if self.flow_normaluvscale else 1.0
        scroll_distance = self.flow_uvscrolldistance
        interval = self.flow_timeintervalinseconds

        # vWorldUv = i.vBumpTexCoord.xy * flWorldUvScale -- the flow map is sampled
        # with the surface's texture coordinates, not the world-space ones below.
        flowmap_node = self.create_texture_node(flowmap, '$flowmap')
        flow_uv = self.create_node(Nodes.ShaderNodeUVMap, 'flowmap UV')
        flow_uv_scaled = self.create_node(Nodes.ShaderNodeVectorMath, 'flowmap uv scale')
        flow_uv_scaled.operation = 'SCALE'
        flow_uv_scaled.inputs['Scale'].default_value = world_uv_scale
        self.connect_nodes(flow_uv.outputs[0], flow_uv_scaled.inputs[0])
        self.connect_nodes(flow_uv_scaled.outputs[0], flowmap_node.inputs[0])

        # vFlowVectorTs = flowmap.rg * 2 - 1, i.e. a signed 2D direction.
        flow_vector = self.create_node(Nodes.ShaderNodeVectorMath, 'flow vector')
        flow_vector.operation = 'MULTIPLY_ADD'
        flow_vector.inputs[1].default_value = (2.0, 2.0, 2.0)
        flow_vector.inputs[2].default_value = (-1.0, -1.0, -1.0)
        self.connect_nodes(flowmap_node.outputs['Color'], flow_vector.inputs[0])

        # scroll_offset = vFlowVectorTs * $flow_uvscrolldistance * scrollTime, with
        # scrollTime ramping 0..1 across the interval. The ramp is the keyframed
        # part; the direction comes from the texture.
        layers = []
        for index, phase in enumerate((0.0, 0.5)):
            texture = self.create_texture_node(image, f'$normalmap flow{index + 1}')

            step = self.create_node(Nodes.ShaderNodeVectorMath, f'flow{index + 1} step')
            step.operation = 'SCALE'
            self.connect_nodes(flow_vector.outputs[0], step.inputs[0])
            self.connect_nodes(self._flow_scroll_output(scroll_distance, interval, phase),
                               step.inputs['Scale'])

            scaled_uv = self.create_node(Nodes.ShaderNodeVectorMath, f'flow{index + 1} uv scale')
            scaled_uv.operation = 'SCALE'
            scaled_uv.inputs['Scale'].default_value = normal_uv_scale
            self.connect_nodes(self._flow_world_uv_output(), scaled_uv.inputs[0])

            offset_uv = self.create_node(Nodes.ShaderNodeVectorMath, f'flow{index + 1} uv offset')
            offset_uv.operation = 'ADD'
            self.connect_nodes(scaled_uv.outputs[0], offset_uv.inputs[0])
            self.connect_nodes(step.outputs[0], offset_uv.inputs[1])
            self.connect_nodes(offset_uv.outputs[0], texture.inputs[0])
            layers.append(texture)

        # lerp( n1, n2, flWeight2 ) with flWeight2 = abs(2*frac(t) - 1): a triangle
        # wave that is 1 when layer 1 has just restarted and 0 mid-interval.
        blend = self.create_node(Nodes.ShaderNodeMixRGB, 'flow crossfade')
        blend.blend_type = 'MIX'
        self.connect_nodes(layers[0].outputs['Color'], blend.inputs['Color1'])
        self.connect_nodes(layers[1].outputs['Color'], blend.inputs['Color2'])
        self.connect_nodes(self._flow_weight_output(interval), blend.inputs['Fac'])
        return blend.outputs['Color']

    def _flow_world_uv_output(self):
        """``float2( i.worldPos.x, -i.worldPos.y )`` -- the normal layers' coordinates.

        The flow shader looks its normal map up in world space, not in the surface's
        UVs, which is what lets a single tiling normal map cover arbitrarily large
        water without seams.

        Two conversions are needed. The Y axis is negated, matching the SDK. And the
        authored ``$flow_normaluvscale`` is calibrated for Hammer units, while
        Blender geometry is in metres, so world position is converted back to Hammer
        units before scaling -- otherwise the normal map tiles ~52x too large.
        """
        existing = self.get_node('flow world uv')
        if existing is not None:
            return existing.outputs[0]

        geometry = self.create_node(Nodes.ShaderNodeNewGeometry, 'flow position')
        split = self.create_node(Nodes.ShaderNodeSeparateXYZ, 'flow world split')
        self.connect_nodes(geometry.outputs['Position'], split.inputs[0])

        negate_y = self.create_node(Nodes.ShaderNodeMath, 'flow world -y')
        negate_y.operation = 'MULTIPLY'
        negate_y.inputs[1].default_value = -1.0
        self.connect_nodes(split.outputs['Y'], negate_y.inputs[0])

        combine = self.create_node(Nodes.ShaderNodeCombineXYZ, 'flow world xy')
        self.connect_nodes(split.outputs['X'], combine.inputs['X'])
        self.connect_nodes(negate_y.outputs[0], combine.inputs['Y'])

        to_hammer = self.create_node(Nodes.ShaderNodeVectorMath, 'flow world uv')
        to_hammer.operation = 'SCALE'
        to_hammer.inputs['Scale'].default_value = 1.0 / SOURCE1_HAMMER_UNIT_TO_METERS
        self.connect_nodes(combine.outputs[0], to_hammer.inputs[0])
        return to_hammer.outputs[0]

    def _flow_interval_output(self, interval: float):
        """The SDK's ``flTimeInIntervals`` -- time in units of one flow interval.

        ``flTimeInIntervals = g_flTime / ( $flow_timeintervalinseconds * 2 )``. The
        per-pixel ``flNoise`` term is omitted; see :meth:`_build_flow_normal`.
        """
        existing = self.get_node('flow intervals')
        if existing is not None:
            return existing.outputs[0]
        period = (interval or 0.4) * 2.0
        return self._math_node('DIVIDE', 'flow intervals', period).outputs[0]

    def _flow_scroll_output(self, scroll_distance: float, interval: float, phase: float):
        """``flScrollTimeN * $flow_uvscrolldistance`` -- one saw ramp per interval.

        SDK: ``flScrollTime1 = frac( t )``, ``flScrollTime2 = frac( t + 0.5 )``, so
        each layer ramps 0..1 over an interval, half an interval out of phase. The
        snap back to 0 is invisible because the cross-fade weight is 0 for that
        layer at exactly that instant.
        """
        intervals = self._flow_interval_output(interval)
        suffix = '2' if phase else '1'
        if phase:
            shifted = self._math_node('ADD', f'flow phase {suffix}', phase, socket=intervals)
            intervals = shifted.outputs[0]
        ramp = self._math_node('FRACT', f'flow ramp {suffix}', socket=intervals)
        scaled = self._math_node('MULTIPLY', f'flow scroll {suffix}', scroll_distance,
                                 socket=ramp.outputs[0])
        return scaled.outputs[0]

    def _flow_weight_output(self, interval: float):
        """``flWeight2 = abs( 2 * frac( flTimeInIntervals ) - 1 )``.

        A triangle wave: 1 when layer 1 has just restarted (so layer 2 is showing
        fully), 0 half an interval later when layer 2 restarts.
        """
        ramp = self._math_node('FRACT', 'flow weight fract',
                               socket=self._flow_interval_output(interval))
        doubled = self._math_node('MULTIPLY_ADD', 'flow weight scale', 2.0, socket=ramp.outputs[0])
        doubled.inputs[2].default_value = -1.0
        return self._math_node('ABSOLUTE', 'flow weight', socket=doubled.outputs[0]).outputs[0]

    def _setup_bump_uv(self, texture_node):
        """Feed ``$normalmap`` from ``$bumptransform``, scrolled if a proxy asks.

        ``TextureScroll`` animates ``$bumptransform``'s translation at
        ``texturescrollrate`` per second along ``texturescrollangle``; 72 of the
        shipped HL2/Portal 2 water materials do exactly that.
        """
        transform = self.bumptransform
        scroll = self._proxy('texturescroll')
        if scroll is None and (not transform or transform == self._DEFAULT_TRANSFORM):
            return

        mapping, _ = self.handle_transform(transform or dict(self._DEFAULT_TRANSFORM),
                                           texture_node.inputs[0])
        if scroll is None:
            return
        if str(scroll.get('texturescrollvar', '$bumptransform')).lower() != '$bumptransform':
            return

        rate = self._proxy_float(scroll, 'texturescrollrate', 0.0)
        if rate == 0.0:
            return
        angle = math.radians(self._proxy_float(scroll, 'texturescrollangle', 0.0))
        self._scroll_uv(mapping, rate * math.cos(angle), rate * math.sin(angle))
        self.logger.info(f'Scrolling $bumptransform at {rate}/s along {math.degrees(angle):.0f} deg')

    def _time_output(self):
        """Socket carrying Source's ``curtime`` in seconds, shared per material.

        A single Value node is driven by the bare expression ``frame * <1/fps>`` and
        every time-varying term is then built from Math nodes fed by it. The driver
        deliberately contains nothing but a multiply: materials are constructed
        before they are assigned to an object, and a driver evaluated on an unused
        datablock runs in Blender's restricted namespace, which rejects *any*
        function name (``abs``, ``min``, ...) and latches ``bpy.app.autoexec_fail``
        for the rest of the session -- leaving such drivers stuck at 0.0 even after
        the material is assigned. Bare arithmetic is always allowed.
        """
        existing = self.get_node('curtime')
        if existing is not None:
            return existing.outputs[0]

        render = bpy.context.scene.render
        fps = render.fps or 24
        fps_base = getattr(render, 'fps_base', 1.0) or 1.0

        time_node = self.create_node(Nodes.ShaderNodeValue, 'curtime')
        fcurve = time_node.outputs[0].driver_add('default_value')
        fcurve.driver.type = 'SCRIPTED'
        fcurve.driver.expression = f'frame * {fps_base / fps:.10g}'
        return time_node.outputs[0]

    def _math_node(self, operation: str, name: str, value=None, socket=None):
        """Math node computing ``time <operation> value``, fed from ``socket``."""
        node = self.create_node(Nodes.ShaderNodeMath, name)
        node.operation = operation
        self.connect_nodes(socket if socket is not None else self._time_output(), node.inputs[0])
        if value is not None:
            node.inputs[1].default_value = value
        return node

    def _scroll_uv(self, mapping_node, speed_x: float, speed_y: float):
        """Slide a Mapping node's Location at ``speed_*`` UV units per second.

        Matches Source's ``TexOffsets = curtime * $scroll`` and ``TextureScroll``'s
        rate, both of which are per second.
        """
        if speed_x == 0.0 and speed_y == 0.0:
            return
        offset = self.create_node(Nodes.ShaderNodeCombineXYZ, 'scroll offset')
        for index, speed in (('X', speed_x), ('Y', speed_y)):
            if speed == 0.0:
                continue
            scaled = self._math_node('MULTIPLY', f'scroll speed {index}', speed)
            self.connect_nodes(scaled.outputs[0], offset.inputs[index])
        self.connect_nodes(offset.outputs[0], mapping_node.inputs['Location'])
