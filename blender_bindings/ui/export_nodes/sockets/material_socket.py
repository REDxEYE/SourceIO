import bpy
from bpy.types import NodeSocket


class SourceIOMaterialSocket(NodeSocket):
    bl_idname = 'SourceIOMaterialSocket'
    bl_lable = 'Material socket'
    type = "MATERIAL"
    material: bpy.props.PointerProperty(name="Material", type=bpy.types.Material)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "material", text=text)

    def draw_color(self, context, node):
        return 1.0, 0.2, 1.0, 1.0
