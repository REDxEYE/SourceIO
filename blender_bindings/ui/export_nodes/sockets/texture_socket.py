import bpy
from bpy.types import NodeSocket


class SourceIOTextureSocket(NodeSocket):
    bl_idname = 'SourceIOTextureSocket'
    bl_lable = 'Texture socket'
    type = "TEXTURE"
    texture: bpy.props.PointerProperty(name="Texture", type=bpy.types.Image)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "texture", text=text)

    def draw_color(self, context, node):
        return 1.0, 0.2, 1.0, 1.0
