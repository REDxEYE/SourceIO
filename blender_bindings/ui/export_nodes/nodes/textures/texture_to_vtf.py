import bpy
import numpy as np
from bpy.types import Node

from SourceIO.blender_bindings.operators.source1_operators import get_formats, get_filters
from SourceIO.blender_bindings.ui.export_nodes.nodes.base_node import SourceIOTextureTreeNode
from SourceIO.library.utils.pylib.vtf import ImageFormat, MipFilter
from SourceIO.library.utils.pylib.vtf import VTFFile


class SourceIOTextureToVTFNode(SourceIOTextureTreeNode):
    bl_idname = "SourceIOTextureToVTFNode"
    bl_label = "Texture to VTF"
    vtf_name: bpy.props.StringProperty(name="VTF name", default="texture.vtf")

    img_format: bpy.props.EnumProperty(name="Texture format", description="Texture format", items=get_formats(),
                                       default=ImageFormat.RGBA8888.name)

    generate_mipmaps: bpy.props.BoolProperty(
        name="Generate Mipmaps",
        description="Generate mipmaps for the texture",
        default=True,
    )

    mip_filter: bpy.props.EnumProperty(name="Mipmap filter", description="Mipmap filter", items=get_filters(),
                                       default=MipFilter.CATROM.name)

    fmt_remap = {a.name: a for a in list(ImageFormat)}
    filter_remap = {a.name: a for a in list(MipFilter)}

    def init(self, context):
        self.inputs.new("SourceIOTextureSocket", "texture")
        self.outputs.new("SourceIOTextureVtfSocket", "vtf_texture")

    def draw_buttons(self, context, layout):
        layout.prop(self, "vtf_name")
        layout.prop(self, "img_format")
        layout.prop(self, "generate_mipmaps")
        if self.generate_mipmaps:
            layout.prop(self, "mip_filter")

    def process(self, inputs: dict) -> dict | None:
        if "texture" not in inputs:
            return None

        img_data: np.ndarray = inputs["texture"]
        if img_data is None:
            return None

        fmt = self.fmt_remap[self.img_format]
        filter = self.filter_remap[self.mip_filter] if self.generate_mipmaps else MipFilter.GAUSSIAN

        if fmt in [ImageFormat.DXT1, ImageFormat.DXT3, ImageFormat.DXT5,
                   ImageFormat.RGBA8888, ImageFormat.RGB888, ImageFormat.BGR888,
                   ImageFormat.RGB565, ImageFormat.BGR565, ImageFormat.ABGR8888, ImageFormat.ARGB8888,
                   ImageFormat.RGB888_BLUESCREEN, ImageFormat.BGR888_BLUESCREEN, ImageFormat.BGRA4444,
                   ImageFormat.BGRA5551, ImageFormat.BGRX5551, ImageFormat.BGRX8888,
                   ]:
            src_format = ImageFormat.RGBA8888
            img_data_raw = (img_data * 255).astype(np.uint8).tobytes()
        else:
            src_format = ImageFormat.RGBA16161616F
            img_data_raw = img_data.astype(np.float16).tobytes()

        vtf = VTFFile()
        height, width, channels = img_data.shape

        vtf.create_from_data(img_data_raw, width, height, 1, 1, 1,
                             src_format, fmt, filter,
                             generate_mipmaps=self.generate_mipmaps,
                             generate_thumbnail=True,
                             resize_to_pow2=2
                             )
        data = vtf.to_bytes()
        return {"vtf_texture": (self.vtf_name, data)}
