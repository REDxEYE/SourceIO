import bpy

from .....logger import SLoggingManager
from ..source1_shader_base import Source1ShaderBase

log_manager = SLoggingManager()
logger = log_manager.get_logger('MaterialLoader')


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
    def detailfactor(self):
        return self._vmt.get_float('$detailblendfactor', 1)

    @property
    def detailmode(self):
        return self._vmt.get_int('$detailblendmode', -1)

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

    def handle_detail(self, next_socket: bpy.types.NodeSocket, albedo_socket: bpy.types.NodeSocket, *, uv_node=None):
        if self.detailmode not in [0, 1, 2, 5]:
            logger.error(f'Failed to load detail: unhandled Detail mode, got' + str(self.detailmode))
            return albedo_socket, None
        detailblend = self.create_node_group('$DetailBlendMode' + str(self.detailmode), [-500, -60], name='DetailBlend')
        detailblend.width = 210
        detailblend.inputs['$detailblendfactor [float]'].default_value = self.detailfactor
        if self.detailmode == 5:
            self.connect_nodes(detailblend.outputs['BSDF'], next_socket.node.outputs['BSDF'].links[0].to_socket)
            self.connect_nodes(next_socket.node.outputs['BSDF'], detailblend.inputs[0])
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
        scale = self.create_node("ShaderNodeVectorMath")
        scale.location = [-1250, -150]
        scale.operation = "MULTIPLY"
        scale.inputs[1].default_value = self.detailscale
        if self.detailtexturetransform and uv_node is not None:
            self.handle_transform(self.detailtexturetransform, scale.inputs[0], uv_node=uv_node)
        else:
            if uv_node is not None:
                uv = uv_node
            else:
                uv = self.create_node("ShaderNodeUVMap")
            uv.location = [-1400, -150]
            self.connect_nodes(uv.outputs[0], scale.inputs[0])

        self.connect_nodes(scale.outputs[0], detail.inputs[0])
        return detailblend.outputs.get('$basetexture [texture]', albedo_socket), detail
