import os
from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from ..material_loader.material_loader import Source1MaterialLoader
from ..source1.bsp.import_bsp import BSP, BPSPropCache
from ..source1.dmx.load_sfm_session import load_session
from ..source1.mdl.model_loader import import_model_from_full_path

from ...library.source1.vtf import is_vtflib_supported
from ...library.shared.content_providers.content_manager import ContentManager
from ...library.utils.math_utilities import SOURCE1_HAMMER_UNIT_TO_METERS
from ...logger import SLoggingManager

logger = SLoggingManager().get_logger("SourceIO::Operators")


# noinspection PyPep8Naming
class SOURCEIO_OT_MDLImport(bpy.types.Operator):
    """Load Source Engine MDL models"""
    bl_idname = "sourceio.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    write_qc: BoolProperty(name="Write QC", default=True, subtype='UNSIGNED')
    import_animations: BoolProperty(name="Load animations", default=False, subtype='UNSIGNED')
    unique_materials_names: BoolProperty(name="Unique material names", default=False, subtype='UNSIGNED')

    create_flex_drivers: BoolProperty(name="Create drivers for flexes", default=False, subtype='UNSIGNED')
    bodygroup_grouping: BoolProperty(name="Group meshes by bodygroup", default=True, subtype='UNSIGNED')
    import_textures: BoolProperty(name="Import materials", default=True, subtype='UNSIGNED')
    use_bvlg: BoolProperty(name="Use BlenderVertexLitGeneric shader", default=True, subtype='UNSIGNED')
    scale: FloatProperty(name="World scale", default=SOURCE1_HAMMER_UNIT_TO_METERS, precision=6)
    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

    def execute(self, context):
        from ..source1.mdl.v49.import_mdl import put_into_collections, import_materials, import_animations

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        content_manager = ContentManager()
        content_manager.scan_for_content(directory)

        bpy.context.scene['content_manager_data'] = content_manager.serialize()

        for file in self.files:
            mdl_path = directory / file.name
            model_container = import_model_from_full_path(mdl_path, self.scale, self.create_flex_drivers,
                                                          unique_material_names=self.unique_materials_names)
            put_into_collections(model_container, mdl_path.stem, bodygroup_grouping=self.bodygroup_grouping)

            if self.import_textures and is_vtflib_supported():
                try:

                    import_materials(model_container.mdl, unique_material_names=self.unique_materials_names,
                                     use_bvlg=self.use_bvlg)
                except Exception as t_ex:
                    logger.error(f'Failed to import materials, caused by {t_ex}')
                    import traceback
                    traceback.print_exc()
            if self.import_animations and model_container.armature:
                logger.info('Loading animations')
                import_animations(model_container.mdl, model_container.armature, self.scale)
            if self.write_qc:
                from ...library.source1.qc.qc import generate_qc
                from ... import bl_info
                qc_file = bpy.data.texts.new('{}.qc'.format(Path(file.name).stem))
                generate_qc(model_container.mdl, qc_file, ".".join(map(str, bl_info['version'])))
        content_manager.flush_cache()
        content_manager.clean()
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
class SOURCEIO_OT_BSPImport(bpy.types.Operator):
    """Load Source Engine BSP models"""
    bl_idname = "sourceio.bsp"
    bl_label = "Import Source BSP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    scale: FloatProperty(name="World scale", default=SOURCE1_HAMMER_UNIT_TO_METERS, precision=6)
    import_textures: BoolProperty(name="Import materials", default=True, subtype='UNSIGNED')
    import_cubemaps: BoolProperty(name="Import cubemaps", default=False, subtype='UNSIGNED')
    # import_decal: BoolProperty(name="Import decals", default=False, subtype='UNSIGNED')
    use_bvlg: BoolProperty(name="Use BlenderVertexLitGeneric shader", default=True, subtype='UNSIGNED')

    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    def execute(self, context):
        content_manager = ContentManager()
        content_manager.scan_for_content(self.filepath)

        bsp_map = BSP(self.filepath, scale=self.scale)
        bpy.context.scene['content_manager_data'] = content_manager.serialize()

        BPSPropCache().purge()

        bsp_map.load_disp()
        bsp_map.load_entities()
        bsp_map.load_static_props()
        if self.import_cubemaps:
            bsp_map.load_cubemap()
        # if self.import_decal:
        #     bsp_map.load_overlays()
        if self.import_textures:
            bsp_map.load_materials(self.use_bvlg)
        content_manager.flush_cache()
        content_manager.clean()
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
        content_manager = ContentManager()
        directory = Path(self.filepath).parent.absolute()
        for file in self.files:
            load_session(directory / file.name, 1)

        content_manager.flush_cache()
        content_manager.clean()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


if is_vtflib_supported():
    from ..source1.vtf.export_vtf import export_texture
    from ..source1.vtf import import_texture, load_skybox_texture


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
                import_texture(file.name, (directory / file.name).open('rb'), True)
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
            ContentManager().scan_for_content(directory)
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

        filepath: StringProperty(
            subtype='FILE_PATH',
        )
        files: CollectionProperty(type=bpy.types.PropertyGroup)
        filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
        override: BoolProperty(default=False, name='Override existing?')

        def execute(self, context):
            content_manager = ContentManager()
            if Path(self.filepath).is_file():
                directory = Path(self.filepath).parent.absolute()
            else:
                directory = Path(self.filepath).absolute()
            for file in self.files:
                mat = Source1MaterialLoader((directory / file.name).open('rb'), Path(file.name).stem)
                if mat.create_material() == 'EXISTS' and not self.override:
                    self.report({'INFO'}, '{} material already exists')
            content_manager.flush_cache()
            content_manager.clean()
            return {'FINISHED'}

        def invoke(self, context, event):
            wm = context.window_manager
            wm.fileselect_add(self)
            return {'RUNNING_MODAL'}


    # noinspection PyUnresolvedReferences,PyPep8Naming
    class SOURCEIO_OT_VTFExport(bpy.types.Operator):
        """Export VTF texture"""
        bl_idname = "sourceio.export_vtf"
        bl_label = "Export VTF"

        filename_ext = ".vtf"

        filter_glob: StringProperty(default="*.vtf", options={'HIDDEN'})

        filepath: StringProperty(
            subtype='FILE_PATH',
        )

        filename: StringProperty(
            name="File Name",
            description="Name used by the exported file",
            maxlen=255,
            subtype='FILE_NAME',
        )

        img_format: EnumProperty(
            name="VTF Type Preset",
            description="Choose a preset. It will affect the result's format and flags.",
            items=(
                ('RGBA8888', "RGBA8888 Simple", "RGBA8888 format, format-specific Eight Bit Alpha flag only"),
                ('RGBA8888Normal', "RGBA8888 Normal Map",
                 "RGB8888 format, format-specific Eight Bit Alpha and Normal Map flags"),
                ('RGB888', "RGB888 Simple", "RGB888 format, no alpha"),
                ('RGB888Normal', "RGB888 Normal Map", "RGB888 format, no alpha and Normal map flag"),
                ('DXT1', "DXT1 Simple", "DXT1 format, no flags"),
                ('DXT5', "DXT5 Simple", "DXT5 format, format-specific Eight Bit Alpha flag only"),
                ('DXT1Normal', "DXT1 Normal Map", "DXT1 format, Normal Map flag only"),
                ('DXT5Normal', "DXT5 Normal Map", "DXT5 format, format-specific Eight Bit Alpha and Normal Map flags")),
            default='RGBA8888',
        )
        filter_mode: EnumProperty(
            name='VTF mipmap filter',
            items=(
                ('0', 'Point Filter', 'Point Filter'),
                ('1', 'Box Filter', 'Box Filter'),
                ('2', 'Triangle Filter', 'Triangle Filter'),
                ('3', 'Quadratic Filter', 'Quadratic Filter'),
                ('4', 'Cubic Filter', 'Cubic Filter'),
                ('5', 'Catrom Filter', 'Catrom Filter'),
                ('6', 'Mitchell Filter', 'Mitchell Filter'),
                ('7', 'Gaussian Filter', 'Gaussian Filter'),
                ('8', 'SinC Filter', 'SinC Filter'),
                ('9', 'Bessel Filter', 'Bessel Filter'),
                ('10', 'Hanning Filter', 'Hanning Filter'),
                ('11', 'Hamming Filter', 'Hamming Filter'),
                ('12', 'Blackman Filter', 'Blackman Filter'),
                ('13', 'Kaiser Filter', 'Kaiser Filter'),
                ('14', 'Count Filter', 'Count Filter'),
            ),
            default='0'
        )

        def execute(self, context):
            sima = context.space_data
            ima = sima.image
            if ima is None:
                self.report({"ERROR_INVALID_INPUT"}, "No Image provided")
            else:
                logger.info(context)
                export_texture(ima, self.filepath, self.img_format, self.filter_mode)
            return {'FINISHED'}

        def invoke(self, context, event):
            if not self.filepath:
                blend_filepath = context.blend_data.filepath
                if not blend_filepath:
                    blend_filepath = "untitled"
                else:
                    blend_filepath = os.path.splitext(blend_filepath)[0]
                    self.filepath = os.path.join(
                        os.path.dirname(blend_filepath),
                        self.filename + self.filename_ext)
            else:
                self.filepath = os.path.join(
                    os.path.dirname(
                        self.filepath),
                    self.filename +
                    self.filename_ext)

            context.window_manager.fileselect_add(self)
            return {'RUNNING_MODAL'}


    def export(self, context):
        cur_img = context.space_data.image
        if cur_img is None:
            self.layout.operator(SOURCEIO_OT_VTFExport.bl_idname, text='Export to VTF')
        else:
            self.layout.operator(SOURCEIO_OT_VTFExport.bl_idname, text='Export to VTF').filename = \
                os.path.splitext(cur_img.name)[0]
