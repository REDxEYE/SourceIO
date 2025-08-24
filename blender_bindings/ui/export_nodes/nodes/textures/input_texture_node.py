import bpy
import numpy as np
from bpy.types import Node

from SourceIO.blender_bindings.ui.export_nodes.nodes.base_node import SourceIOTextureTreeNode


class SourceIOTextureInputNode(SourceIOTextureTreeNode):
    bl_idname = 'SourceIOTextureInputNode'
    bl_label = "Texture"
    texture: bpy.props.PointerProperty(type=bpy.types.Image, name="Texture")

    _cached_result: np.ndarray | None = None

    def draw_buttons(self, context, layout):
        self.update()
        layout.prop(self, 'texture')

    def init(self, context):
        self.outputs.new('SourceIOTextureSocket', "texture")

    def process(self, inputs: dict) -> dict|None:
        width, height = self.texture.size
        data = np.zeros((height, width, 4), dtype=np.float32)
        self.texture.pixels.foreach_get(data.ravel())
        return {"texture": data}