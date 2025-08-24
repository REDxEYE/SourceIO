import bpy
from bpy.types import NodeSocket


class SourceIOTextureSocket(NodeSocket):
    bl_idname = 'SourceIOTextureSocket'
    bl_lable = 'Texture socket'
    type = "TEXTURE"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 1.0, 0.2, 1.0, 1.0


class SourceIOTextureChannelSocket(NodeSocket):
    bl_idname = 'SourceIOTextureChannelSocket'
    bl_lable = 'Texture Channel socket'
    type = "TEXTURE_CHANNEL"
    default_value: bpy.props.FloatProperty(
        name="Default Value",
        description="Default channel value when no texture is connected",
        default=0.0
    )

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "default_value", text=text)

    def draw_color(self, context, node):
        return 0.5, 0.5, 0.5, 1.0

class SourceIOTextureVtfSocket(NodeSocket):
    bl_idname = 'SourceIOTextureVtfSocket'
    bl_lable = 'VTF Texture socket'
    type = "TEXTURE_VTF"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    def draw_color(self, context, node):
        return 0.3, 0.2, 1.0, 1.0
