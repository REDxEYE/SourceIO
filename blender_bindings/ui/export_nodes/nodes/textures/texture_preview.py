import bpy
import numpy as np

from SourceIO.blender_bindings.ui.export_nodes.build_graph import (
    GraphResolver)
from SourceIO.blender_bindings.ui.export_nodes.nodes.base_node import (
    SourceIOTextureTreeNode)


class SourceIO_OP_EvaluatePreview(bpy.types.Operator):
    bl_idname = "sourceio.evaluate_texture_preview"
    bl_label = "Evaluate texture preview"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return bool(space and space.type == 'NODE_EDITOR'
                    and space.edit_tree
                    and space.edit_tree.bl_idname
                    == 'sourceio.model_definition')

    def execute(self, context: bpy.types.Context):
        tree = context.space_data.edit_tree
        active = context.active_node
        if active and active.bl_idname == "SourceIOTexturePreviewNode":
            previews = [active]
        else:
            previews = [node for node in tree.nodes
                        if node.bl_idname == "SourceIOTexturePreviewNode"]
        resolver = GraphResolver()
        try:
            for node in previews:
                inputs = {}
                for socket in node.inputs:
                    if not socket.is_linked:
                        continue
                    inputs[socket.name] = resolver.pixels_for(resolver(socket))
                node.process(inputs)
        except (ValueError, OSError, RuntimeError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {"FINISHED"}


class SourceIOTexturePreviewNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTexturePreviewNode"
    bl_label = "Texture Preview"
    bl_icon = "TEXTURE"

    display: bpy.props.EnumProperty(name="Show", items=(
        ("RGB", "RGB", "Color channels at full opacity, like VTF viewers and "
                       "the engine sample them"),
        ("ALPHA", "Alpha", "Alpha channel as grayscale"),
        ("RGBA", "RGBA", "Color with alpha as transparency"),
    ), default="RGB")

    preview_image: bpy.props.PointerProperty(type=bpy.types.Image,
                                             options={'HIDDEN'})

    def copy(self, node):
        self.preview_image = None

    def init(self, context):
        self.inputs.new("SourceIOTextureSocket", "texture")
        self.inputs.new("SourceIOTextureChannelSocket", "channel")

    def draw_buttons(self, context, layout):
        layout.label(text="Preview the texture connected to the input.")
        layout.prop(self, "display", expand=True)
        layout.operator(SourceIO_OP_EvaluatePreview.bl_idname,
                        text="Update Preview")

    def process(self, inputs: dict) -> None:
        """Write the input into this node's preview Image."""
        texture = inputs.get("texture")
        if (isinstance(texture, np.ndarray) and texture.ndim == 3
                and texture.shape[-1] in (3, 4)):
            if texture.shape[-1] == 4:
                alpha = texture[..., 3]
            else:
                alpha = np.ones(texture.shape[:2], np.float32)
            opaque = np.ones_like(alpha)
            if self.display == "ALPHA":
                img_data = np.stack((alpha, alpha, alpha, opaque), axis=-1)
            elif self.display == "RGB":
                img_data = np.concatenate(
                    (texture[..., :3], opaque[..., None]), axis=-1)
            else:
                img_data = np.concatenate(
                    (texture[..., :3], alpha[..., None]), axis=-1)
            # Exported VTFs clamp to 0..1; show the same values.
            img_data = np.clip(img_data, 0.0, 1.0).astype(np.float32)
        else:
            channel = inputs.get("channel")
            if (not isinstance(channel, np.ndarray)
                    or channel.ndim not in (1, 2)):
                return None
            if channel.ndim == 1:
                channel = channel[np.newaxis, :]
            img_data = np.stack(
                (channel, channel, channel, np.ones_like(channel)), axis=-1)

        image = self.preview_image
        height, width = img_data.shape[:2]
        if image is None:
            name = f"SOURCE_NODES__Texture Preview {self.id_data.name} {self.name}"
            image = bpy.data.images.new(name, width=width, height=height,
                                        alpha=True)
            self.preview_image = image
        elif tuple(image.size) != (width, height):
            image.scale(width, height)
        image.pixels.foreach_set(img_data.ravel())
        image.update()
        return None
