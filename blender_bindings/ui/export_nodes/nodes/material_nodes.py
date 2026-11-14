"""Source 1 prop VMT/VTF graph nodes. All effects are returned as recipes."""

import bpy
from bpy.types import Node

from SourceIO.blender_bindings.source1.model_export.material_assets import (
    ExternalTextureSourceRecipe, ImageSourceRecipe, KVPair, MaterialParameters,
    VTFRecipe, parse_kv_entries, render_material, render_vtf)
from .base_node import SourceIOModelTreeNode


_FORMAT_ITEMS = [(name, name, '') for name in (
    'DXT1', 'DXT3', 'DXT5', 'RGBA8888', 'RGB888', 'BGR888',
    'RGB565', 'BGR565', 'ABGR8888', 'ARGB8888', 'BGRA4444',
    'BGRA5551', 'BGRX5551', 'BGRX8888', 'I8', 'IA88', 'A8')]
_FILTER_ITEMS = [(name, name, '') for name in (
    'POINT', 'BOX', 'TRIANGLE', 'QUADRATIC', 'CUBIC', 'CATROM',
    'MITCHELL', 'GAUSSIAN', 'SINC', 'BESSEL', 'HANNING',
    'HAMMING', 'BLACKMAN', 'KAISER')]


def _linked_recipe(socket, resolve):
    if not socket.is_linked:
        return None
    result = resolve(socket)
    if result is None:
        raise ValueError(f'Linked input {socket.name!r} did not resolve')
    return result


class SourceIOImageSourceNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOImageSourceNode'
    bl_label = 'Image Source'

    image: bpy.props.PointerProperty(name='Image', type=bpy.types.Image)

    def init(self, context):
        self.outputs.new('SourceIOTextureSocket', 'Texture')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'image')

    def recipe(self, resolve=None) -> ImageSourceRecipe:
        if self.image is None:
            raise ValueError('Image Source requires a Blender Image')
        return ImageSourceRecipe(self.image)


class SourceIOExternalTextureNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOExternalTextureNode'
    bl_label = 'External Texture Source'

    source_path: bpy.props.StringProperty(name='Image / VTF file',
                                          subtype='FILE_PATH')

    def init(self, context):
        self.outputs.new('SourceIOTextureSocket', 'Texture')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'source_path')

    def recipe(self, resolve=None) -> ExternalTextureSourceRecipe:
        if not self.source_path.strip():
            raise ValueError('External Texture Source requires an image or '
                             'VTF file path')
        return ExternalTextureSourceRecipe(self.source_path)


class SourceIOVTFOutputNode(Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOVTFOutputNode'
    bl_label = 'VTF Output'

    vtf_name: bpy.props.StringProperty(name='VTF path under materials',
                                       default='models/props/texture.vtf')
    img_format: bpy.props.EnumProperty(name='Format', items=_FORMAT_ITEMS,
                                       default='DXT5')
    generate_mipmaps: bpy.props.BoolProperty(name='Generate mipmaps',
                                             default=True)
    mip_filter: bpy.props.EnumProperty(name='Mipmap filter',
                                       items=_FILTER_ITEMS, default='CATROM')
    is_normal: bpy.props.BoolProperty(name='Normal map (linear)',
                                      default=False)
    linear_space: bpy.props.BoolProperty(
        name='Linear space', default=True,
        description='Write the image values unchanged without the SRGB '
                    'flag, as vtex does. Off converts colours from linear '
                    'to sRGB and flags the VTF SRGB, which looks washed out '
                    'for ordinary 8-bit images')

    def init(self, context):
        self.width = 220
        self.inputs.new('SourceIOTextureSocket', 'Texture')
        self.outputs.new('SourceIOTextureVtfSocket', 'VTF')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'vtf_name')
        layout.prop(self, 'img_format')
        layout.prop(self, 'generate_mipmaps')
        if self.generate_mipmaps:
            layout.prop(self, 'mip_filter')
        layout.prop(self, 'is_normal')
        if not self.is_normal:
            layout.prop(self, 'linear_space')

    def recipe(self, resolve) -> VTFRecipe:
        source = _linked_recipe(self.inputs['Texture'], resolve)
        if source is None:
            raise ValueError('VTF Output requires a linked Image, external '
                             'file, or pixel array')
        return render_vtf(source, self.vtf_name, self.img_format,
                          self.mip_filter, self.generate_mipmaps,
                          self.is_normal, self.linear_space)


def _number(value):
    return format(value, '.9g')


def _vector(values):
    return '[{}]'.format(' '.join(_number(value) for value in values))


def _transform(scale, rotation, offset):
    return ('center .5 .5 scale {} rotate {} translate {}'
            .format(' '.join(_number(value) for value in scale),
                    _number(rotation),
                    ' '.join(_number(value) for value in offset)))


def _update_socket_visibility(node, context):
    node._sync_socket_visibility()


class _VMTNodeMixin:
    """Shared Blender controls and recipe collection for Source 1 shaders.

    Blender registers annotated RNA properties on concrete Node classes. Their
    annotations are copied to each concrete class below, not registered here.
    """
    mat: bpy.props.PointerProperty(name='Blender material',
                                   type=bpy.types.Material)
    vmt_path: bpy.props.StringProperty(name='VMT path under materials',
                                       default='models/props/material.vmt')
    model: bpy.props.BoolProperty(name='$model', default=True)
    translucent: bpy.props.BoolProperty(name='$translucent')
    alphatest: bpy.props.BoolProperty(name='$alphatest')
    alphatest_reference: bpy.props.FloatProperty(name='$alphatestreference',
                                                 default=0.5, min=0, max=1)
    nocull: bpy.props.BoolProperty(name='$nocull')
    selfillum: bpy.props.BoolProperty(name='$selfillum')
    additive: bpy.props.BoolProperty(name='$additive')
    vertex_color: bpy.props.BoolProperty(name='$vertexcolor')
    vertex_alpha: bpy.props.BoolProperty(name='$vertexalpha')
    alpha: bpy.props.FloatProperty(name='$alpha', default=1.0, min=0, max=1)
    halflambert: bpy.props.BoolProperty(name='$halflambert')
    rimlight: bpy.props.BoolProperty(name='$rimlight')
    rimlight_boost: bpy.props.FloatProperty(name='$rimlightboost',
                                            default=1.0, min=0)
    rimlight_exponent: bpy.props.FloatProperty(name='$rimlightexponent',
                                               default=4.0, min=0)
    phong_fresnel_enabled: bpy.props.BoolProperty(name='$phongfresnelranges')
    phong_fresnel_ranges: bpy.props.FloatVectorProperty(
        name='Phong Fresnel ranges', size=3, default=(0.0, 0.5, 1.0), min=0)
    phong: bpy.props.BoolProperty(name='$phong')
    phong_boost: bpy.props.FloatProperty(name='$phongboost', default=1.0,
                                         min=0)
    phong_exponent: bpy.props.FloatProperty(name='$phongexponent',
                                            default=5.0, min=0)
    phong_tint: bpy.props.FloatVectorProperty(
        name='$phongtint', size=3, default=(1, 1, 1), min=0, subtype='COLOR')
    phong_albedo_tint: bpy.props.BoolProperty(
        name='$phongalbedotint',
        description='Tint phong by the base texture, masked by exponent '
                    'texture G')
    normal_alpha_envmap_mask: bpy.props.BoolProperty(
        name='$normalmapalphaenvmapmask',
        description='Use the normal map alpha as the envmap mask')
    envmap: bpy.props.StringProperty(name='$envmap')
    envmap_tint: bpy.props.FloatVectorProperty(
        name='$envmaptint', size=3, default=(1, 1, 1), min=0,
        subtype='COLOR')
    envmap_contrast: bpy.props.FloatProperty(name='$envmapcontrast',
                                             default=0.0)
    envmap_saturation: bpy.props.FloatProperty(name='$envmapsaturation',
                                               default=1.0, min=0)
    color: bpy.props.FloatVectorProperty(name='$color2', size=3,
                                         default=(1, 1, 1), min=0, max=1,
                                         subtype='COLOR')
    detail_blend_factor: bpy.props.FloatProperty(name='$detailblendfactor',
                                                 default=1.0)
    detail_blend_mode: bpy.props.IntProperty(name='$detailblendmode',
                                             default=0, min=0)
    detail_scale: bpy.props.FloatProperty(name='$detailscale', default=1.0)
    eyeball_radius: bpy.props.FloatProperty(name='$eyeballradius',
                                            default=0.0, min=0)
    dilation: bpy.props.FloatProperty(name='$dilation', default=0.0, min=0)
    parallax_strength: bpy.props.FloatProperty(name='$parallaxstrength',
                                               default=0.0)
    pupil: bpy.props.FloatProperty(name='$pupil', default=0.0, min=0)
    refract_amount: bpy.props.FloatProperty(name='$refractamount',
                                            default=0.0)
    refract_tint: bpy.props.FloatVectorProperty(
        name='$refracttint', size=3, default=(1, 1, 1), min=0,
        subtype='COLOR')
    blur_amount: bpy.props.FloatProperty(name='$bluramount', default=0.0,
                                         min=0)
    raw_parameters: bpy.props.PointerProperty(
        name='Ordered raw VMT KV entries', type=bpy.types.Text)
    proxy_parameters: bpy.props.PointerProperty(name='Proxy KV entries',
                                                type=bpy.types.Text)

    show_textures: bpy.props.BoolProperty(
        name='Textures', default=True, update=_update_socket_visibility)
    show_surface: bpy.props.BoolProperty(
        name='Surface / Alpha', default=True,
        update=_update_socket_visibility)
    show_normal: bpy.props.BoolProperty(
        name='Normal', default=True, update=_update_socket_visibility)
    show_phong: bpy.props.BoolProperty(
        name='Phong', default=False, update=_update_socket_visibility)
    show_uv: bpy.props.BoolProperty(
        name='UV transforms', default=False,
        update=_update_socket_visibility)
    show_environment: bpy.props.BoolProperty(
        name='Environment / Mask', default=False,
        update=_update_socket_visibility)
    show_specific: bpy.props.BoolProperty(
        name='Shader-specific', default=True,
        update=_update_socket_visibility)
    show_raw: bpy.props.BoolProperty(
        name='Raw / Proxies', default=False,
        update=_update_socket_visibility)

    base_uv_enabled: bpy.props.BoolProperty(name='$basetexturetransform')
    base_uv_scale: bpy.props.FloatVectorProperty(name='Base UV scale',
                                                 size=2, default=(1, 1))
    base_uv_rotation: bpy.props.FloatProperty(name='Base UV rotation',
                                              default=0)
    base_uv_offset: bpy.props.FloatVectorProperty(name='Base UV translation',
                                                  size=2)
    normal_uv_enabled: bpy.props.BoolProperty(name='$bumptransform')
    normal_uv_scale: bpy.props.FloatVectorProperty(name='Normal UV scale',
                                                   size=2, default=(1, 1))
    normal_uv_rotation: bpy.props.FloatProperty(name='Normal UV rotation',
                                                default=0)
    normal_uv_offset: bpy.props.FloatVectorProperty(
        name='Normal UV translation', size=2)
    detail_uv_enabled: bpy.props.BoolProperty(name='$detailtexturetransform')
    detail_uv_scale: bpy.props.FloatVectorProperty(name='Detail UV scale',
                                                   size=2, default=(1, 1))
    detail_uv_rotation: bpy.props.FloatProperty(name='Detail UV rotation',
                                                default=0)
    detail_uv_offset: bpy.props.FloatVectorProperty(
        name='Detail UV translation', size=2)

    # (category, visible socket label, VMT texture parameter)
    _textures = ()
    _groups = ()
    _uv_slots = ()
    _scalar_keys = ()
    _vector_keys = ()
    _shader = ''
    _editable_shader = False

    def init(self, context):
        self.width = 250
        for _, label, _ in self._textures:
            self.inputs.new('SourceIOTextureVtfSocket', label)
        self.outputs.new('SourceIOMaterialSocket', 'Material')
        self._sync_socket_visibility()

    def _sync_socket_visibility(self):
        for group, label, _ in self._textures:
            socket = self.inputs.get(label)
            if socket is not None:
                hidden = (not getattr(self, 'show_' + group)
                          and not socket.is_linked)
                if socket.hide != hidden:
                    socket.hide = hidden

    def _has_detail(self):
        return any(key == '$detail' for _, _, key in self._textures)

    def update(self):
        self._sync_socket_visibility()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mat')
        layout.prop(self, 'vmt_path')
        layout.operator('sourceio.sync_vmt',
                        icon='FILE_REFRESH').material_node = self.name
        if self._editable_shader:
            layout.prop(self, 'shader_name')
        for group in self._groups:
            expanded = getattr(self, 'show_' + group)
            box = layout.box()
            row = box.row()
            row.prop(self, 'show_' + group, emboss=False,
                     icon='TRIA_DOWN' if expanded else 'TRIA_RIGHT')
            if not expanded:
                continue
            for socket_group, label, _ in self._textures:
                if socket_group == group:
                    state = ('linked' if self.inputs[label].is_linked
                             else 'unlinked')
                    box.label(text=f'{label}: {state}')
            if group == 'surface':
                for name in ('model', 'translucent', 'alphatest'):
                    box.prop(self, name)
                if self.alphatest:
                    box.prop(self, 'alphatest_reference')
                for name in ('nocull', 'selfillum', 'color'):
                    box.prop(self, name)
                if self._shader == 'UnlitGeneric':
                    for name in ('additive', 'vertex_color', 'vertex_alpha',
                                 'alpha'):
                        box.prop(self, name)
                if self._shader == 'VertexLitGeneric':
                    box.prop(self, 'halflambert')
            elif group == 'textures':
                if self._has_detail():
                    box.prop(self, 'detail_blend_factor')
                    box.prop(self, 'detail_scale')
                    box.prop(self, 'detail_blend_mode')
            elif group == 'phong':
                box.prop(self, 'phong')
                if self.phong:
                    for name in ('phong_boost', 'phong_exponent',
                                 'phong_tint'):
                        box.prop(self, name)
                    if self._shader == 'VertexLitGeneric':
                        if self.inputs['Phong exponent texture'].is_linked:
                            box.label(text='$phongexponent omitted: '
                                           'exponent texture linked')
                            box.prop(self, 'phong_albedo_tint')
                        box.prop(self, 'phong_fresnel_enabled')
                        if self.phong_fresnel_enabled:
                            box.prop(self, 'phong_fresnel_ranges')
                        box.prop(self, 'rimlight')
                        if self.rimlight:
                            box.prop(self, 'rimlight_boost')
                            box.prop(self, 'rimlight_exponent')
            elif group == 'environment':
                for name in ('envmap', 'envmap_tint', 'envmap_contrast',
                             'envmap_saturation'):
                    box.prop(self, name)
                if self._shader == 'VertexLitGeneric':
                    box.prop(self, 'normal_alpha_envmap_mask')
            elif group == 'uv':
                for slot in self._uv_slots:
                    box.prop(self, slot + '_uv_enabled')
                    if getattr(self, slot + '_uv_enabled'):
                        for suffix in ('scale', 'rotation', 'offset'):
                            box.prop(self, slot + '_uv_' + suffix)
            elif group == 'specific':
                for name, _ in self._scalar_keys + self._vector_keys:
                    box.prop(self, name)
            elif group == 'raw':
                box.prop(self, 'raw_parameters')
                box.prop(self, 'proxy_parameters')

    def recipe(self, resolve):
        textures = []
        for _, label, key in self._textures:
            texture = _linked_recipe(self.inputs[label], resolve)
            if texture is not None:
                if not isinstance(texture, VTFRecipe):
                    raise ValueError(
                        f'{label} must resolve to a VTF Output recipe')
                textures.append((key, texture))
        raw = (parse_kv_entries(self.raw_parameters.as_string(),
                                'VMT parameters')
               if self.raw_parameters else ())
        proxies = (parse_kv_entries(self.proxy_parameters.as_string(),
                                    'VMT proxies')
                   if self.proxy_parameters else ())
        if any(not isinstance(pair.value, tuple) for pair in proxies):
            raise ValueError(
                'Each Proxies entry must be a named KeyValues block')
        groups = self._groups
        params = MaterialParameters(
            model=self.model, translucent=self.translucent,
            alphatest=self.alphatest,
            alphatest_reference=self.alphatest_reference,
            nocull=self.nocull, selfillum=self.selfillum,
            phong=self.phong if 'phong' in groups else False,
            phong_boost=self.phong_boost, phong_exponent=self.phong_exponent,
            envmap=self.envmap if 'environment' in groups else '',
            color=tuple(self.color))
        typed = []
        if self._has_detail():
            if self.detail_blend_factor != 1.0:
                typed.append(KVPair('$detailblendfactor',
                                    _number(self.detail_blend_factor)))
            if self.detail_scale != 1.0:
                typed.append(KVPair('$detailscale',
                                    _number(self.detail_scale)))
            if self.detail_blend_mode:
                typed.append(KVPair('$detailblendmode',
                                    str(self.detail_blend_mode)))
        if ('phong' in groups and self.phong
                and tuple(self.phong_tint) != (1, 1, 1)):
            typed.append(KVPair('$phongtint', _vector(self.phong_tint)))
        if self._shader == 'VertexLitGeneric':
            typed.extend(self._vertex_lit_pairs())
        if self._shader == 'UnlitGeneric':
            for enabled, key in ((self.additive, '$additive'),
                                 (self.vertex_color, '$vertexcolor'),
                                 (self.vertex_alpha, '$vertexalpha')):
                if enabled:
                    typed.append(KVPair(key, '1'))
            if self.alpha != 1.0:
                typed.append(KVPair('$alpha', _number(self.alpha)))
        if 'environment' in groups and self.envmap:
            if tuple(self.envmap_tint) != (1, 1, 1):
                typed.append(KVPair('$envmaptint',
                                    _vector(self.envmap_tint)))
            if self.envmap_contrast != 0.0:
                typed.append(KVPair('$envmapcontrast',
                                    _number(self.envmap_contrast)))
            if self.envmap_saturation != 1.0:
                typed.append(KVPair('$envmapsaturation',
                                    _number(self.envmap_saturation)))
        for slot in self._uv_slots:
            if getattr(self, slot + '_uv_enabled'):
                key = {'base': '$basetexturetransform',
                       'normal': '$bumptransform',
                       'detail': '$detailtexturetransform'}[slot]
                typed.append(KVPair(key, _transform(
                    getattr(self, slot + '_uv_scale'),
                    getattr(self, slot + '_uv_rotation'),
                    getattr(self, slot + '_uv_offset'))))
        for name, key in self._scalar_keys:
            value = getattr(self, name)
            if value != 0.0:
                typed.append(KVPair(key, _number(value)))
        for name, key in self._vector_keys:
            value = getattr(self, name)
            if tuple(value) != (1, 1, 1):
                typed.append(KVPair(key, _vector(value)))
        shader = (self.shader_name.strip() if self._editable_shader
                  else self._shader)
        return render_material(self.mat, self.vmt_path, shader, params,
                               textures=tuple(textures),
                               typed_pairs=tuple(typed),
                               raw_pairs=raw, proxies=proxies)

    def _vertex_lit_pairs(self):
        """VertexLitGeneric-only typed parameters."""
        pairs = []
        if self.halflambert:
            pairs.append(KVPair('$halflambert', '1'))
        if self.phong and self.phong_fresnel_enabled:
            pairs.append(KVPair('$phongfresnelranges',
                                _vector(self.phong_fresnel_ranges)))
        if (self.phong and self.phong_albedo_tint
                and self.inputs['Phong exponent texture'].is_linked):
            pairs.append(KVPair('$phongalbedotint', '1'))
        if self.envmap and self.normal_alpha_envmap_mask:
            if not self.inputs['Normal texture'].is_linked:
                raise ValueError(f'{self.name}: $normalmapalphaenvmapmask '
                                 f'needs a linked Normal texture')
            if self.inputs['Environment mask'].is_linked:
                raise ValueError(f'{self.name}: use either the Environment '
                                 f'mask or $normalmapalphaenvmapmask')
            pairs.append(KVPair('$normalmapalphaenvmapmask', '1'))
        if self.phong and self.rimlight:
            pairs.extend((
                KVPair('$rimlight', '1'),
                KVPair('$rimlightboost', _number(self.rimlight_boost)),
                KVPair('$rimlightexponent', _number(self.rimlight_exponent)),
            ))
        return pairs


_SURFACE = ('textures', 'surface', 'normal', 'phong', 'uv', 'environment',
            'raw')
_BASE_DETAIL = (('textures', 'Base texture', '$basetexture'),
                ('textures', 'Detail texture', '$detail'))
_NORMAL = (('normal', 'Normal texture', '$bumpmap'),)
_ENV_MASK = (('environment', 'Environment mask', '$envmapmask'),)
_SELFILLUM_MASK = (('surface', 'Self-illumination mask', '$selfillummask'),)


class SourceIOVertexLitGenericVMTNode(_VMTNodeMixin, Node,
                                      SourceIOModelTreeNode):
    bl_idname = 'SourceIOVertexLitGenericVMTNode'
    bl_label = 'VMT VertexLitGeneric'
    _shader = 'VertexLitGeneric'
    _textures = (_BASE_DETAIL + _NORMAL
                 + (('phong', 'Phong exponent texture',
                     '$phongexponenttexture'),
                    ('phong', 'Ambient occlusion texture',
                     '$ambientocclusiontexture'))
                 + _ENV_MASK + _SELFILLUM_MASK)
    _groups = _SURFACE
    _uv_slots = ('base', 'normal', 'detail')


class SourceIOUnlitGenericVMTNode(_VMTNodeMixin, Node,
                                  SourceIOModelTreeNode):
    bl_idname = 'SourceIOUnlitGenericVMTNode'
    bl_label = 'VMT UnlitGeneric'
    _shader = 'UnlitGeneric'
    _textures = _BASE_DETAIL + _ENV_MASK + _SELFILLUM_MASK
    _groups = ('textures', 'surface', 'uv', 'environment', 'raw')
    _uv_slots = ('base', 'detail')


class SourceIOEyeRefractVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOEyeRefractVMTNode'
    bl_label = 'VMT EyeRefract'
    _shader = 'EyeRefract'
    _textures = (('textures', 'Iris texture', '$iris'),
                 ('textures', 'Cornea texture', '$corneatexture'),
                 ('textures', 'Ambient occlusion texture',
                  '$ambientoccltexture'))
    _groups = ('textures', 'surface', 'specific', 'raw')
    _scalar_keys = (('eyeball_radius', '$eyeballradius'),
                    ('dilation', '$dilation'),
                    ('parallax_strength', '$parallaxstrength'))


class SourceIOEyesVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOEyesVMTNode'
    bl_label = 'VMT Eyes'
    _shader = 'Eyes'
    _textures = (('textures', 'Base texture', '$basetexture'),
                 ('textures', 'Iris texture', '$iris'))
    _groups = ('textures', 'surface', 'specific', 'raw')
    _scalar_keys = (('eyeball_radius', '$eyeballradius'), ('pupil', '$pupil'))


class SourceIORefractVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIORefractVMTNode'
    bl_label = 'VMT Refract'
    _shader = 'Refract'
    _textures = (('textures', 'Base texture', '$basetexture'),
                 ('normal', 'Normal texture', '$normalmap'),
                 ('normal', 'DuDv texture', '$dudvmap'))
    _groups = ('textures', 'surface', 'normal', 'uv', 'specific', 'raw')
    _uv_slots = ('base',)
    _scalar_keys = (('refract_amount', '$refractamount'),
                    ('blur_amount', '$bluramount'))
    _vector_keys = (('refract_tint', '$refracttint'),)


class SourceIOTeethVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOTeethVMTNode'
    bl_label = 'VMT Teeth'
    _shader = 'Teeth'
    _textures = _BASE_DETAIL + _NORMAL
    _groups = ('textures', 'surface', 'normal', 'phong', 'uv', 'raw')
    _uv_slots = ('base', 'normal', 'detail')


class SourceIOHairVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOHairVMTNode'
    bl_label = 'VMT Hair'
    shader_name: bpy.props.StringProperty(name='Shader identifier',
                                          default='Hair')
    _editable_shader = True
    _textures = _BASE_DETAIL + _NORMAL
    _groups = ('textures', 'surface', 'normal', 'uv', 'raw')
    _uv_slots = ('base', 'normal', 'detail')


class SourceIOSkinVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOSkinVMTNode'
    bl_label = 'VMT Skin'
    shader_name: bpy.props.StringProperty(name='Shader identifier',
                                          default='Skin')
    _editable_shader = True
    _textures = _BASE_DETAIL + _NORMAL
    _groups = ('textures', 'surface', 'normal', 'uv', 'raw')
    _uv_slots = ('base', 'normal', 'detail')


class SourceIOGenericVMTNode(_VMTNodeMixin, Node, SourceIOModelTreeNode):
    bl_idname = 'SourceIOGenericVMTNode'
    bl_label = 'VMT Generic shader'
    shader_name: bpy.props.StringProperty(name='Shader identifier',
                                          default='VertexLitGeneric')
    _editable_shader = True
    _textures = _BASE_DETAIL + _NORMAL + _ENV_MASK
    _groups = _SURFACE
    _uv_slots = ('base', 'normal', 'detail')


VMT_NODE_TYPES = (SourceIOVertexLitGenericVMTNode, SourceIOUnlitGenericVMTNode,
                  SourceIOEyeRefractVMTNode, SourceIOEyesVMTNode,
                  SourceIORefractVMTNode, SourceIOTeethVMTNode,
                  SourceIOHairVMTNode, SourceIOSkinVMTNode,
                  SourceIOGenericVMTNode)
# Blender collects bpy.props annotations from registered concrete classes, not
# from an unregistered Python mixin.
for _node_type in VMT_NODE_TYPES:
    _node_type.__annotations__ = {**_VMTNodeMixin.__annotations__,
                                  **_node_type.__annotations__}

classes = (SourceIOImageSourceNode, SourceIOExternalTextureNode,
           SourceIOVTFOutputNode) + VMT_NODE_TYPES
