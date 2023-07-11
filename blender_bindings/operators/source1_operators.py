from pathlib import Path

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, StringProperty)

from .import_settings_base import MDLSettings, Source1BSPSettings
from ..source1.mdl import put_into_collections
from ..source1.mdl.v49.import_mdl import import_animations
from ..utils.resource_utils import serialize_mounted_content, deserialize_mounted_content
from ...library.shared.content_providers.content_manager import ContentManager
from ..source1.vtf import import_texture, load_skybox_texture
# from ..source1.vtf.export_vtf import export_texture

from ...logger import SLoggingManager
from ..material_loader.material_loader import Source1MaterialLoader
from ..material_loader.shaders.source1_shader_base import Source1ShaderBase
from ..source1.bsp.import_bsp import BSP
from ..source1.dmx.load_sfm_session import load_session
from ..source1.mdl.model_loader import import_model_from_full_path

logger = SLoggingManager().get_logger("SourceIO::Operators")


# noinspection PyPep8Naming
class SOURCEIO_OT_MDLImport(bpy.types.Operator, MDLSettings):
    """Load Source Engine MDL models"""
    bl_idname = "sourceio.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

    def execute(self, context):
        from ..source1.mdl.v49.import_mdl import import_materials

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.resolve()
        else:
            directory = Path(self.filepath).resolve()
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)

        for file in self.files:
            mdl_path = directory / file.name
            model_container = import_model_from_full_path(mdl_path, self.scale, self.create_flex_drivers,
                                                          unique_material_names=self.unique_materials_names,
                                                          bodygroup_grouping=self.bodygroup_grouping,
                                                          load_physics=self.import_physics,
                                                          load_refpose=self.load_refpose)
            put_into_collections(model_container, Path(model_container.mdl.header.name).stem,
                                 bodygroup_grouping=self.bodygroup_grouping)
            if self.import_textures:
                try:
                    import_materials(model_container.mdl, unique_material_names=self.unique_materials_names,
                                     use_bvlg=self.use_bvlg)
                except Exception as t_ex:
                    logger.error(f'Failed to import materials, caused by {t_ex}')
                    import traceback
                    traceback.print_exc()
            if self.import_animations and model_container.armature:
                import_animations(content_manager, model_container.mdl, model_container.armature, self.scale)
            if self.write_qc:
                from ... import bl_info
                from ...library.source1.qc.qc import generate_qc
                qc_file = bpy.data.texts.new('{}.qc'.format(Path(file.name).stem))
                generate_qc(model_container.mdl, qc_file, ".".join(map(str, bl_info['version'])))
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SOURCEIO_OT_RigImport(bpy.types.Operator):
    """Load SFM ik-rig script"""
    bl_idname = "sourceio.rig"
    bl_label = "Import SFM ik-rig script"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.py", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).absolute()
        else:
            raise Exception("Expected file")
        from ..source1.fake_sfm import load_script
        load_script(directory)

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_BSPImport(bpy.types.Operator, Source1BSPSettings):
    """Load Source Engine BSP models"""
    bl_idname = "sourceio.bsp"
    bl_label = "Import Source BSP file"
    bl_options = {'UNDO'}

    # import_decal: BoolProperty(name="Import decals", default=False, subtype='UNSIGNED')
    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    def execute(self, context):
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(self.filepath)
        else:
            deserialize_mounted_content(content_manager)

        bsp_map = BSP(self.filepath, content_manager, self)

        if self.discover_resources:
            serialize_mounted_content(content_manager)

        bsp_map.load_disp()
        bsp_map.load_entities()
        bsp_map.load_static_props()
        if self.import_cubemaps:
            bsp_map.load_cubemap()
        # if self.import_decal:
        #     bsp_map.load_overlays()
        if self.import_textures:
            bsp_map.load_materials(self.use_bvlg)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


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
        directory = Path(self.filepath).parent.absolute()
        for file in self.files:
            load_session(directory / file.name, 1)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_VTFImport(bpy.types.Operator):
    """Load Source Engine VTF texture"""
    bl_idname = "sourceio.vtf"
    bl_label = "Import VTF"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    filter_glob: StringProperty(default="*.vtf", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            import_texture(Path(file.name), (directory / file.name).open('rb'), True)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_SkyboxImport(bpy.types.Operator):
    """Load Source Engine Skybox texture"""
    bl_idname = "sourceio.vtf_skybox"
    bl_label = "Import Skybox"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
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
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(directory)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)
        for file in self.files:
            skybox_name = Path(file.name).stem
            load_skybox_texture(skybox_name[:-2], int(self.resolution))
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class SOURCEIO_OT_VMTImport(bpy.types.Operator):
    """Load Source Engine VMT material"""
    bl_idname = "sourceio.vmt"
    bl_label = "Import VMT"
    bl_options = {'UNDO'}

    discover_resources: BoolProperty(name="Mount discovered content", default=True)
    filepath: StringProperty(
        subtype='FILE_PATH',
    )
    files: CollectionProperty(type=bpy.types.PropertyGroup)
    filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
    override: BoolProperty(default=False, name='Override existing?')
    use_bvlg: BoolProperty(name="Use BlenderVertexLitGeneric shader", default=True, subtype='UNSIGNED')

    def execute(self, context):
        content_manager = ContentManager()
        if self.discover_resources:
            content_manager.scan_for_content(self.filepath)
            serialize_mounted_content(content_manager)
        else:
            deserialize_mounted_content(content_manager)
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            Source1ShaderBase.use_bvlg(self.use_bvlg)
            mat = Source1MaterialLoader((directory / file.name).open('rb'), Path(file.name).stem)
            bpy_material = bpy.data.materials.get(mat.material_name, dict())
            if bpy_material.get('source_loaded'):
                if self.override:
                    del bpy_material['source_loaded']
                else:
                    self.report({'INFO'}, '{} material already exists')
            mat.create_material()
        content_manager.clean()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

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
