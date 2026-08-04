import bpy

from SourceIO.blender_bindings.material_loader.shader_base import Nodes
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('MaterialLoader')

#: ``$detailblendmode`` values (``TCOMBINE_*`` in ``common_ps_fxc.h``) that are
#: backed by a ``$DetailBlendModeN`` node group in the bundled asset .blend.
NODE_GROUP_DETAIL_MODES = (0, 1, 2, 5)

#: TCOMBINE_MOD2X_SELECT_TWO_PATTERNS -- built from primitives instead, since the
#: asset .blend ships no node group for it.
TCOMBINE_MOD2X_SELECT_TWO_PATTERNS = 7


class DetailSupportMixin(Source1ShaderBase):

    @property
    def detail(self):
        texture_path = self._vmt.get_string('$detail', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 0.5))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def detail2(self):
        texture_path = self._vmt.get_string('$detail2', None)
        if texture_path is not None:
            image = self.load_texture_or_default(texture_path, (1.0, 1.0, 1.0, 0.5))
            image.colorspace_settings.is_data = True
            image.colorspace_settings.name = 'Non-Color'
            return image
        return None

    @property
    def detailfactor(self):
        return self._vmt.get_float('$detailblendfactor', 1)
    
    @property
    def detailfactor2(self):
        return self._vmt.get_float('$detailblendfactor2', 1)

    @property
    def detailmode(self):
        return self._vmt.get_int('$detailblendmode', 0)

    @property
    def detailscale(self):
        value, value_type = self._vmt.get_vector('$detailscale', [4, 4])
        if value is None:
            return None
        divider = 255 if value_type is int else 1
        value = list(map(lambda a: a / divider, value))
        if len(value) == 1:
            value = [value[0], value[0]]
        value += (1,)
        return self.ensure_length(value, 3, 1.0)
    
    @property
    def detailscale2(self):
        value, value_type = self._vmt.get_vector('$detailscale2', [4, 4])
        if value is None:
            return None
        divider = 255 if value_type is int else 1
        value = list(map(lambda a: a / divider, value))
        if len(value) == 1:
            value = [value[0], value[0]]
        value += (1,)
        return self.ensure_length(value, 3, 1.0)

    @property
    def detailtint(self):
        color_value, value_type = self._vmt.get_vector('$detailtint', [1, 1, 1])

        divider = 255 if value_type is int else 1
        color_value = list(map(lambda a: a / divider, color_value))
        if len(color_value) == 1:
            color_value = [color_value[0], color_value[0], color_value[0]]

        return self.ensure_length(color_value, 4, 1.0)

    @property
    def detailtexturetransform(self):
        return self._vmt.get_transform_matrix('$detailtexturetransform',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
                                               'translate': (0, 0, 0)})
    
    @property
    def detailtexturetransform2(self):
        return self._vmt.get_transform_matrix('$basetexturetransform2',
                                              {'center': (0.5, 0.5, 0), 'scale': (1.0, 1.0, 1), 'rotate': (0, 0, 0),
                                               'translate': (0, 0, 0)})

    def _build_detail_uv(self, detail_node, detail_scale, transform, uv_node):
        """Feed ``detail_node`` from a $detailscale-multiplied UV chain."""
        scale = self.create_node(Nodes.ShaderNodeVectorMath)
        scale.location = [-1250, -150]
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = detail_scale
        if transform and uv_node is not None:
            self.handle_transform(transform, scale.inputs[0], uv_node=uv_node)
        else:
            uv = uv_node if uv_node is not None else self.create_node(Nodes.ShaderNodeUVMap)
            uv.location = [-1400, -150]
            self.connect_nodes(uv.outputs[0], scale.inputs[0])
        self.connect_nodes(scale.outputs[0], detail_node.inputs[0])

    def _handle_detail_mod2x_select(self, next_socket, albedo_socket, detail_image,
                                    blend_factor, detail_scale, transform, uv_node):
        """``$detailblendmode`` 7 -- TCOMBINE_MOD2X_SELECT_TWO_PATTERNS.

        The asset .blend has no node group for this mode, so build it from
        primitives. SDK (``TextureCombine`` in ``common_ps_fxc.h``)::

            float3 dc = lerp( detailColor.r, detailColor.a, baseColor.a );
            baseColor.rgb *= lerp( float3(1,1,1), 2.0*dc, fBlendFactor );

        Base alpha selects between the detail texture's red and alpha channels,
        and the result is applied as a mod2x multiply.
        """
        basetexture_node = albedo_socket.node
        if 'Alpha' not in basetexture_node.outputs:
            logger.error('Detail mode 7 needs a base texture with an alpha channel')
            return albedo_socket, None

        detail = self.create_texture_node(detail_image, '$detail')
        detail.location = [-1100, -130]
        self._build_detail_uv(detail, detail_scale, transform, uv_node)

        split = self.create_node(Nodes.ShaderNodeSeparateColor, 'detail split')
        split.mode = 'RGB'
        self.connect_nodes(detail.outputs['Color'], split.inputs[0])

        # dc = lerp( detail.r, detail.a, baseColor.a )
        select = self.create_node(Nodes.ShaderNodeMix, 'detail pattern select')
        select.data_type = 'FLOAT'
        self.connect_nodes(basetexture_node.outputs['Alpha'], select.inputs['Factor'])
        self.connect_nodes(split.outputs[0], select.inputs['A'])          # red
        self.connect_nodes(detail.outputs['Alpha'], select.inputs['B'])   # alpha
        select.location = [-800, -130]

        # 2.0 * dc
        mod2x = self.create_node(Nodes.ShaderNodeMath, 'detail mod2x')
        mod2x.operation = 'MULTIPLY'
        mod2x.inputs[1].default_value = 2.0
        self.connect_nodes(select.outputs[0], mod2x.inputs[0])

        # lerp( 1.0, 2*dc, fBlendFactor )
        blend = self.create_node(Nodes.ShaderNodeMix, 'detail blendfactor')
        blend.data_type = 'FLOAT'
        blend.inputs['Factor'].default_value = blend_factor
        blend.inputs['A'].default_value = 1.0
        self.connect_nodes(mod2x.outputs[0], blend.inputs['B'])

        # baseColor.rgb *= that
        multiply = self.create_node(Nodes.ShaderNodeMixRGB, 'DetailBlend')
        multiply.blend_type = 'MULTIPLY'
        multiply.inputs['Fac'].default_value = 1.0
        self.connect_nodes(albedo_socket, multiply.inputs['Color1'])
        self.connect_nodes(blend.outputs[0], multiply.inputs['Color2'])
        multiply.location = [-500, -60]

        self.connect_nodes(multiply.outputs['Color'], next_socket)
        return multiply.outputs['Color'], detail

    def handle_detail(self, next_socket: bpy.types.NodeSocket, albedo_socket: bpy.types.NodeSocket, *, uv_node=None):
        if self.detailmode == TCOMBINE_MOD2X_SELECT_TWO_PATTERNS:
            return self._handle_detail_mod2x_select(
                next_socket, albedo_socket, self.detail, self.detailfactor,
                self.detailscale, self.detailtexturetransform, uv_node)
        if self.detailmode not in NODE_GROUP_DETAIL_MODES:
            logger.error(f'Failed to load detail: unhandled Detail mode, got' + str(self.detailmode))
            return albedo_socket, None
        detailblend = self.create_node_group('$DetailBlendMode' + str(self.detailmode), [-500, -60], name='DetailBlend')
        detailblend.width = 210
        detailblend.inputs['$detailblendfactor [float]'].default_value = self.detailfactor
        if self.detailmode == 5:
            # Mode 5 (TCOMBINE_RGB_ADDITIVE_SELFILLUM) splices itself between the
            # shader and whatever consumes it, so it needs an existing outgoing
            # BSDF link to steal. Materials that have not wired the output yet (e.g.
            # Portal 2's paint/bridge_paint_*) would raise IndexError here.
            bsdf_output = next_socket.node.outputs.get('BSDF', None)
            if bsdf_output is None or not bsdf_output.links:
                logger.error('Detail mode 5 needs a connected BSDF output; skipping detail')
                return albedo_socket, None
            self.connect_nodes(detailblend.outputs['BSDF'], bsdf_output.links[0].to_socket)
            self.connect_nodes(bsdf_output, detailblend.inputs[0])
        else:
            self.connect_nodes(albedo_socket, detailblend.inputs['$basetexture [texture]'])
            self.connect_nodes(detailblend.outputs['$basetexture [texture]'], next_socket)
        detail = self.create_and_connect_texture_node(self.detail,
                                                      detailblend.inputs['$detail [texture]'],
                                                      detailblend.inputs.get('$detail alpha [texture alpha]', None),
                                                      name='$detail')
        if self.detailmode == 4:
            self.connect_nodes(albedo_socket.node.outputs['Alpha'],
                               detailblend.intputs['$basetexture alpha [texture alpha]'])
        detail.location = [-1100, -130]
        self._build_detail_uv(detail, self.detailscale, self.detailtexturetransform, uv_node)
        return detailblend.outputs.get('$basetexture [texture]', albedo_socket), detail

    def handle_detail2(self, next_socket: bpy.types.NodeSocket, albedo_socket: bpy.types.NodeSocket, *, uv_node=None):
        if self.detailmode == TCOMBINE_MOD2X_SELECT_TWO_PATTERNS:
            return self._handle_detail_mod2x_select(
                next_socket, albedo_socket, self.detail2, self.detailfactor2,
                self.detailscale2, self.detailtexturetransform2, uv_node)
        if self.detailmode not in NODE_GROUP_DETAIL_MODES:
            logger.error(f'Failed to load detail: unhandled Detail mode, got' + str(self.detailmode))
            return albedo_socket, None
        detailblend = self.create_node_group('$DetailBlendMode' + str(self.detailmode), [-500, -60], name='DetailBlend')
        detailblend.width = 210
        detailblend.inputs['$detailblendfactor [float]'].default_value = self.detailfactor2
        if self.detailmode == 5:
            # Mode 5 (TCOMBINE_RGB_ADDITIVE_SELFILLUM) splices itself between the
            # shader and whatever consumes it, so it needs an existing outgoing
            # BSDF link to steal. Materials that have not wired the output yet (e.g.
            # Portal 2's paint/bridge_paint_*) would raise IndexError here.
            bsdf_output = next_socket.node.outputs.get('BSDF', None)
            if bsdf_output is None or not bsdf_output.links:
                logger.error('Detail mode 5 needs a connected BSDF output; skipping detail')
                return albedo_socket, None
            self.connect_nodes(detailblend.outputs['BSDF'], bsdf_output.links[0].to_socket)
            self.connect_nodes(bsdf_output, detailblend.inputs[0])
        else:
            self.connect_nodes(albedo_socket, detailblend.inputs['$basetexture [texture]'])
            self.connect_nodes(detailblend.outputs['$basetexture [texture]'], next_socket)
        detail = self.create_and_connect_texture_node(self.detail2,
                                                      detailblend.inputs['$detail [texture]'],
                                                      detailblend.inputs.get('$detail alpha [texture alpha]', None),
                                                      name='$detail')
        if self.detailmode == 4:
            self.connect_nodes(albedo_socket.node.outputs['Alpha'],
                               detailblend.intputs['$basetexture alpha [texture alpha]'])
        detail.location = [-1100, -130]
        self._build_detail_uv(detail, self.detailscale2, self.detailtexturetransform2, uv_node)
        return detailblend.outputs.get('$basetexture [texture]', albedo_socket), detail
