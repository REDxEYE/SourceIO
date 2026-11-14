"""Socket types for the Source 1 model-definition node tree."""

import bpy
from bpy.types import NodeSocket


class SourceIOObjectSocket(NodeSocket):
    bl_idname = 'SourceIOObjectSocket'
    bl_label = 'Object socket'
    type = 'OBJECT'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.2, 0.2, 0.2, 1.0


class SourceIOBodygroupSocket(NodeSocket):
    bl_idname = 'SourceIOBodygroupSocket'
    bl_label = 'Bodygroup socket'
    type = 'BODYGROUP'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.5, 0.2, 0.2, 1.0


class SourceIOMaterialSocket(NodeSocket):
    bl_idname = 'SourceIOMaterialSocket'
    bl_label = 'Material socket'
    type = 'MATERIAL'
    material: bpy.props.PointerProperty(name='Material', type=bpy.types.Material)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, 'material', text=text)

    def draw_color(self, context, node):
        return 1.0, 0.2, 1.0, 1.0


class SourceIOSkinGroupSocket(NodeSocket):
    bl_idname = 'SourceIOSkinGroupSocket'
    bl_label = 'Skin group socket'
    type = 'SKINGROUP'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 1, 0.8, 0.2, 1.0


class SourceIOTextureSocket(NodeSocket):
    bl_idname = 'SourceIOTextureSocket'
    bl_label = 'Texture socket'
    type = 'TEXTURE'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 1.0, 0.2, 1.0, 1.0


class SourceIOTextureChannelSocket(NodeSocket):
    bl_idname = 'SourceIOTextureChannelSocket'
    bl_label = 'Texture Channel socket'
    type = 'TEXTURE_CHANNEL'
    default_value: bpy.props.FloatProperty(
        name='Default Value',
        description='Default channel value when no texture is connected',
        default=0.0,
    )

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, 'default_value', text=text)

    def draw_color(self, context, node):
        return 0.5, 0.5, 0.5, 1.0


class SourceIOTextureVtfSocket(NodeSocket):
    bl_idname = 'SourceIOTextureVtfSocket'
    bl_label = 'VTF Texture socket'
    type = 'TEXTURE_VTF'

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.3, 0.2, 1.0, 1.0
