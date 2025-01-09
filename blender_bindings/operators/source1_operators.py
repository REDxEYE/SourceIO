import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       StringProperty)

from .import_settings_base import ModelOptions, Source1BSPSettings
from .operator_helper import ImportOperatorHelper
from SourceIO.blender_bindings.material_loader.material_loader import Source1MaterialLoader
from SourceIO.blender_bindings.material_loader.shaders.source1_shader_base import Source1ShaderBase
from SourceIO.blender_bindings.models import import_model
from SourceIO.blender_bindings.models.common import put_into_collections
from SourceIO.blender_bindings.shared.exceptions import RequiredFileNotFound
from SourceIO.blender_bindings.source1.bsp.import_bsp import import_bsp
from SourceIO.blender_bindings.source1.vtf import import_texture, load_skybox_texture
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material, is_blender_4_1
from SourceIO.blender_bindings.utils.resource_utils import serialize_mounted_content, deserialize_mounted_content
from SourceIO.library.shared.app_id import SteamAppId
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import FileBuffer
from SourceIO.library.utils.path_utilities import path_stem
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

logger = SourceLogMan().get_logger("SourceIO::Operators")


# noinspection PyPep8Naming
class SOURCEIO_OT_MDLImport(ImportOperatorHelper, ModelOptions):
    """Load Source Engine MDL models"""
    bl_idname = "sourceio.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)

    filter_glob: StringProperty(default="*.mdl;*.md3", options={'HIDDEN'})

    def execute(self, context):
        
        directory = self.get_directory()

        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)

        for file in self.files:
            mdl_path = directory / file.name
            with FileBuffer(mdl_path) as f:
                try:
                    model_container = import_model(mdl_path, f, content_manager, self, None)
                except RequiredFileNotFound as e:
                    self.report({"ERROR"}, e.message)
                    return {'CANCELLED'}

            put_into_collections(model_container, mdl_path.stem, bodygroup_grouping=self.bodygroup_grouping)

            # if self.write_qc:
            #     from ... import bl_info
            #     from ...library.source1.qc.qc import generate_qc
            #     qc_file = bpy.data.texts.new('{}.qc'.format(TinyPath(file.name).stem))
            #     generate_qc(model_container.mdl, qc_file, ".".join(map(str, bl_info['version'])))
        return {'FINISHED'}


def get_items():
    return ([(str(-999), "Auto", "")] + [(str(e.value), e.name, "") for e in SteamAppId])


# noinspection PyPep8Naming
class SOURCEIO_OT_BSPImport(ImportOperatorHelper, Source1BSPSettings):
    """Load Source Engine BSP models"""
    bl_idname = "sourceio.bsp"
    bl_label = "Import Source BSP file"
    bl_options = {'UNDO'}

    # import_decal: BoolProperty(name="Import decals", default=False, subtype='UNSIGNED')
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})
    steam_app_id: bpy.props.EnumProperty(
        name="Override steamapp id",
        description="Override steamapp id",
        items=get_items()
    )

    def execute(self, context):
        content_manager = ContentManager()
        filepath = TinyPath(self.filepath)
        if self.discover_resources:
            content_manager.scan_for_content(filepath)
        else:
            deserialize_mounted_content(content_manager)
        with FileBuffer(filepath) as f:
            import_bsp(filepath, f, content_manager, self,
                       SteamAppId(int(self.steam_app_id)) if self.steam_app_id != "-999" else None)

        if self.discover_resources:
            serialize_mounted_content(content_manager)

        return {'FINISHED'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_DMXImporter(bpy.types.Operator):
    """Load Source Engine DMX scene"""
    bl_idname = "sourceio.dmx"
    bl_label = "[!!!WIP!!!] Import Source Session file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    project_dir: StringProperty(default='', name='SFM project folder (usermod)')
    filter_glob: StringProperty(default="*.dmx", options={'HIDDEN'})

    def execute(self, context):
        directory = self.get_directory()
        for file in self.files:
            load_session(directory / file.name, 1)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_VTFImport(ImportOperatorHelper):
    """Load Source Engine VTF texture"""
    bl_idname = "sourceio.vtf"
    bl_label = "Import VTF"
    bl_options = {'UNDO'}

    filter_glob: StringProperty(default="*.vtf", options={'HIDDEN'})
    need_popup = False

    def execute(self, context):
        directory = self.get_directory()

        for file in self.files:
            image = import_texture(TinyPath(file.name), (directory / file.name).open('rb'), True)
            if is_blender_4_1():
                if (context.region and context.region.type == 'WINDOW'
                        and context.area and context.area.ui_type == 'ShaderNodeTree'
                        and context.object and context.object.type == 'MESH'
                        and context.material):
                    node_tree = context.material.node_tree
                    image_node = node_tree.nodes.new(type="ShaderNodeTexImage")
                    image_node.image = image
                    image_node.location = context.space_data.cursor_location
                    for node in context.material.node_tree.nodes:
                        node.select = False
                    image_node.select = True
                if (context.region and context.region.type == 'WINDOW'
                        and context.area and context.area.ui_type in ["IMAGE_EDITOR", "UV"]):
                    context.space_data.image = image

        return {'FINISHED'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_SkyboxImport(ImportOperatorHelper):
    """Load Source Engine Skybox texture"""
    bl_idname = "sourceio.vtf_skybox"
    bl_label = "Import Skybox"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})

    resolution: EnumProperty(
        name="Skybox texture resolution",
        description="Resolution of final skybox texture",
        items=(
            ('1024', "1024x512", "256mb free ram required"),
            ('2048', "2048x1024", "512mb free ram required"),
            ('4096', "4096x2048", "1Gb free ram required"),
            ('8192', "8192x4096", "2Gb free ram required"),
            ('16384', "16384x8192", "8Gb free ram required")),
        default='2048',
    )

    def execute(self, context):
        directory = self.get_directory()
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)
        for file in self.files:
            skybox_name = path_stem(file.name)
            load_skybox_texture(skybox_name[:-2], content_manager, int(self.resolution))
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_VMTImport(ImportOperatorHelper):
    """Load Source Engine VMT material"""
    bl_idname = "sourceio.vmt"
    bl_label = "Import VMT"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
    override: BoolProperty(default=False, name='Override existing?')
    use_bvlg: BoolProperty(name="Use BlenderVertexLitGeneric shader", default=True, subtype='UNSIGNED')

    def execute(self, context):
        directory = self.get_directory()
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)

        for file in self.files:
            Source1ShaderBase.use_bvlg(self.use_bvlg)
            file_path = TinyPath(file.name)
            mat = get_or_create_material(file_path.stem, file_path.as_posix())
            loader = Source1MaterialLoader(content_manager, (directory / file.name).open('rb'), file_path.stem)
            bpy_material = bpy.data.materials.get(loader.material_name, dict())
            if bpy_material.get('source_loaded'):
                if self.override:
                    del bpy_material['source_loaded']
                else:
                    self.report({'INFO'}, '{} material already exists')
            loader.create_material(mat)
        content_manager.clean()
        return {'FINISHED'}

    # # noinspection PyUnresolvedReferences,PyPep8Naming
    # class SOURCEIO_OT_VTFExport(bpy.types.Operator):
    #     """Export VTF texture"""
    #     bl_idname = "sourceio.export_vtf"
    #     bl_label = "Export VTF"
    #
    #     filename_ext = ".vtf"
    #
    #     filter_glob: StringProperty(default="*.vtf", options={'HIDDEN'})
    #
    #     filepath: StringProperty(
    #         subtype='FILE_PATH',
    #     )
    #
    #     filename: StringProperty(
    #         name="File Name",
    #         description="Name used by the exported file",
    #         maxlen=255,
    #         subtype='FILE_NAME',
    #     )
    #
    #     img_format: EnumProperty(
    #         name="VTF Type Preset",
    #         description="Choose a preset. It will affect the result's format and flags.",
    #         items=(
    #             ('RGBA8888', "RGBA8888 Simple", "RGBA8888 format, format-specific Eight Bit Alpha flag only"),
    #             ('RGBA8888Normal', "RGBA8888 Normal Map",
    #              "RGB8888 format, format-specific Eight Bit Alpha and Normal Map flags"),
    #             ('RGB888', "RGB888 Simple", "RGB888 format, no alpha"),
    #             ('RGB888Normal', "RGB888 Normal Map", "RGB888 format, no alpha and Normal map flag"),
    #             ('DXT1', "DXT1 Simple", "DXT1 format, no flags"),
    #             ('DXT5', "DXT5 Simple", "DXT5 format, format-specific Eight Bit Alpha flag only"),
    #             ('DXT1Normal', "DXT1 Normal Map", "DXT1 format, Normal Map flag only"),
    #             ('DXT5Normal', "DXT5 Normal Map", "DXT5 format, format-specific Eight Bit Alpha and Normal Map flags")),
    #         default='RGBA8888',
    #     )
    #     filter_mode: EnumProperty(
    #         name='VTF mipmap filter',
    #         items=(
    #             ('0', 'Point Filter', 'Point Filter'),
    #             ('1', 'Box Filter', 'Box Filter'),
    #             ('2', 'Triangle Filter', 'Triangle Filter'),
    #             ('3', 'Quadratic Filter', 'Quadratic Filter'),
    #             ('4', 'Cubic Filter', 'Cubic Filter'),
    #             ('5', 'Catrom Filter', 'Catrom Filter'),
    #             ('6', 'Mitchell Filter', 'Mitchell Filter'),
    #             ('7', 'Gaussian Filter', 'Gaussian Filter'),
    #             ('8', 'SinC Filter', 'SinC Filter'),
    #             ('9', 'Bessel Filter', 'Bessel Filter'),
    #             ('10', 'Hanning Filter', 'Hanning Filter'),
    #             ('11', 'Hamming Filter', 'Hamming Filter'),
    #             ('12', 'Blackman Filter', 'Blackman Filter'),
    #             ('13', 'Kaiser Filter', 'Kaiser Filter'),
    #             ('14', 'Count Filter', 'Count Filter'),
    #         ),
    #         default='0'
    #     )
    #
    #     def execute(self, context):
    #         sima = context.space_data
    #         ima = sima.image
    #         if ima is None:
    #             self.report({"ERROR_INVALID_INPUT"}, "No Image provided")
    #         else:
    #             logger.info(context)
    #             export_texture(ima, self.filepath, self.img_format, self.filter_mode)
    #         return {'FINISHED'}
    #
    #     def invoke(self, context, event):
    #         if not self.filepath:
    #             blend_filepath = context.blend_data.filepath
    #             if not blend_filepath:
    #                 blend_filepath = "untitled"
    #             else:
    #                 blend_filepath = os.path.splitext(blend_filepath)[0]
    #                 self.filepath = os.path.join(
    #                     os.path.dirname(blend_filepath),
    #                     self.filename + self.filename_ext)
    #         else:
    #             self.filepath = os.path.join(
    #                 os.path.dirname(
    #                     self.filepath),
    #                 self.filename +
    #                 self.filename_ext)
    #
    #         context.window_manager.fileselect_add(self)
    #         return {'RUNNING_MODAL'}
    #
    #
    # def export(self, context):
    #     cur_img = context.space_data.image
    #     if cur_img is None:
    #         self.layout.operator(SOURCEIO_OT_VTFExport.bl_idname, text='Export to VTF')
    #     else:
    #         self.layout.operator(SOURCEIO_OT_VTFExport.bl_idname, text='Export to VTF').filename = \
    #             os.path.splitext(cur_img.name)[0]
