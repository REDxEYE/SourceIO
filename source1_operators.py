import os
from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

from .bpy_utilities.material_loader.material_loader import Source1MaterialLoader
from .source1.bsp.import_bsp import BSP
from .source1.dmx.dmx import Session
from .source1.vtf.export_vtf import export_texture
from .source1.vtf.import_vtf import import_texture
from .source_shared.content_manager import ContentManager
from .utilities.math_utilities import HAMMER_UNIT_TO_METERS
from .utilities.path_utilities import backwalk_file_resolver


# noinspection PyUnresolvedReferences,PyPep8Naming
class MDLImport_OT_operator(bpy.types.Operator):
    """Load Source Engine MDL models"""
    bl_idname = "source_io.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    write_qc: BoolProperty(name="Write QC", default=True, subtype='UNSIGNED')

    create_flex_drivers: BoolProperty(name="Create drivers for flexes", default=False, subtype='UNSIGNED')
    import_textures: BoolProperty(name="Import materials", default=True, subtype='UNSIGNED')
    scale: FloatProperty(name="World scale", default=HAMMER_UNIT_TO_METERS, precision=6)
    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        content_manager = ContentManager()
        content_manager.scan_for_content(directory)

        bpy.context.scene['content_manager_data'] = content_manager.serialize()

        from .source1.mdl.import_model import import_model, import_materials
        for file in self.files:
            mdl_path = directory / file.name
            vvd_file = backwalk_file_resolver(directory, mdl_path.stem + '.vvd')
            vtx_file = backwalk_file_resolver(directory, mdl_path.stem + '.dx90.vtx')

            model_container = import_model(mdl_path.open('rb'), vvd_file.open('rb'), vtx_file.open('rb'), self.scale,
                                           self.create_flex_drivers)

            if self.import_textures:
                try:
                    import_materials(model_container.mdl)
                except Exception as t_ex:
                    print(f'Failed to import materials, caused by {t_ex}')
                    import traceback
                    traceback.print_exc()
            if self.write_qc:
                from .source1.qc.qc import generate_qc
                from . import bl_info
                qc_file = bpy.data.texts.new('{}.qc'.format(Path(file.name).stem))
                generate_qc(model_container.mdl, qc_file, ".".join(map(str, bl_info['version'])))
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class BSPImport_OT_operator(bpy.types.Operator):
    """Load Source Engine BSP models"""
    bl_idname = "source_io.bsp"
    bl_label = "Import Source BSP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    scale: FloatProperty(name="World scale", default=HAMMER_UNIT_TO_METERS, precision=6)
    import_textures: BoolProperty(name="Import materials", default=False, subtype='UNSIGNED')

    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    def execute(self, context):
        content_manager = ContentManager()
        content_manager.scan_for_content(self.filepath)

        bsp_map = BSP(self.filepath, scale=self.scale)
        bpy.context.scene['content_manager_data'] = content_manager.serialize()

        bsp_map.load_map_mesh()
        bsp_map.load_disp()
        bsp_map.load_entities()
        bsp_map.load_static_props()
        # bsp_map.load_detail_props()
        if self.import_textures:
            bsp_map.load_materials()

        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class DMXImporter_OT_operator(bpy.types.Operator):
    """Load Source Engine DMX scene"""
    bl_idname = "source_io.dmx"
    bl_label = "Import Source Session file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    project_dir: StringProperty(default='', name='SFM project folder (usermod)')
    filter_glob: StringProperty(default="*.dmx", options={'HIDDEN'})

    def execute(self, context):
        directory = Path(self.filepath).parent.absolute()
        preferences = context.preferences
        addon_prefs = preferences.addons['SourceIO'].preferences
        print(addon_prefs)
        sfm_path = self.project_dir if self.project_dir else addon_prefs.sfm_path
        for file in self.files:
            importer = Session(str(directory / file.name), sfm_path)
            importer.parse()
            importer.load_scene()
            # importer.load_lights()
            # importer.create_cameras()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class VTFImport_OT_operator(bpy.types.Operator):
    """Load Source Engine VTF texture"""
    bl_idname = "import_texture.vtf"
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
class VMTImport_OT_operator(bpy.types.Operator):
    """Load Source Engine VMT material"""
    bl_idname = "import_texture.vmt"
    bl_label = "Import VMT"
    bl_options = {'UNDO'}

    filepath: StringProperty(
        subtype='FILE_PATH',
    )
    files: CollectionProperty(type=bpy.types.PropertyGroup)
    filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
    override: BoolProperty(default=False, name='Override existing?')

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            mat = Source1MaterialLoader((directory / file.name).open('rb'), Path(file.name).stem)
            if mat.create_material() == 'EXISTS' and not self.override:
                self.report({'INFO'}, '{} material already exists')
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences,PyPep8Naming
class VTFExport_OT_operator(bpy.types.Operator):
    """Export VTF texture"""
    bl_idname = "export_texture.vtf"
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
            print(context)
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
        self.layout.operator(VTFExport_OT_operator.bl_idname, text='Export to VTF')
    else:
        self.layout.operator(VTFExport_OT_operator.bl_idname, text='Export to VTF').filename = \
            os.path.splitext(cur_img.name)[0]
