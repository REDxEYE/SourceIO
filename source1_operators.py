import os
from pathlib import Path

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

# from .source1.mdl import mdl2model, qc_generator
from .source1.vtf.blender_material import BlenderMaterial
from .source1.vtf.export_vtf import export_texture
from .source1.vtf.import_vtf import import_texture
from .source1.dmx.dmx import Session
from .source1.bsp.import_bsp import BSP
from .source1.vtf.vmt import VMT

from .utilities.path_utilities import backwalk_file_resolver
from .utilities.path_utilities import case_insensitive_file_resolution


# noinspection PyUnresolvedReferences
class MDLImporter_OT_operator(bpy.types.Operator):
    """Load Source Engine MDL models"""
    bl_idname = "source_io.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    # organize_bodygroups: BoolProperty(name="Organize bodygroups", default=True, subtype='UNSIGNED')

    write_qc: BoolProperty(name="Write QC", default=True, subtype='UNSIGNED')

    load_phy: BoolProperty(name="Import physics", default=False, subtype='UNSIGNED')
    # import_textures: BoolProperty(name="Import textures", default=False, subtype='UNSIGNED')

    filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

    def execute(self, context):

        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
            # if self.wip_mode:
        from .source1.new_model_import import import_model
        from .source1.new_qc.qc import generate_qc
        from . import bl_info
        for file in self.files:
            mdl_path = directory / file.name
            vvd = backwalk_file_resolver(Path(mdl_path).parent, mdl_path.with_suffix('.vvd'))
            vtx = backwalk_file_resolver(mdl_path.parent, Path(mdl_path.stem + '.dx90.vtx'))
            if self.load_phy:
                phy = backwalk_file_resolver(Path(mdl_path).parent, mdl_path.with_suffix('.phy'))
            else:
                phy = None
            mdl, vvd, vtx = import_model(directory / file.name, vvd, vtx, phy)
            if self.write_qc:
                qc_file = bpy.data.texts.new('{}.qc'.format(Path(file.name).stem))
                text = generate_qc(mdl, ".".join(map(str, bl_info['version'])))
                qc_file.write(text)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# noinspection PyUnresolvedReferences
class BSPImporter_OT_operator(bpy.types.Operator):
    """Load Source Engine BSP models"""
    bl_idname = "source_io.bsp"
    bl_label = "Import Source BSP file"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype="FILE_PATH")
    # files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    filter_glob: StringProperty(default="*.bsp", options={'HIDDEN'})

    def execute(self, context):
        model = BSP(self.filepath)
        model.load_map_mesh()
        model.load_lights()
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


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


class VTFImporter_OT_operator(bpy.types.Operator):
    """Load Source Engine VTF texture"""
    bl_idname = "import_texture.vtf"
    bl_label = "Import VTF"
    bl_options = {'UNDO'}

    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    load_alpha: BoolProperty(default=True, name='Load alpha into separate image')
    only_alpha: BoolProperty(default=False, name='Only load alpha')

    filter_glob: StringProperty(default="*.vtf", options={'HIDDEN'})

    def execute(self, context):
        if Path(self.filepath).is_file():
            directory = Path(self.filepath).parent.absolute()
        else:
            directory = Path(self.filepath).absolute()
        for file in self.files:
            import_texture(str(directory / file.name), self.load_alpha, self.only_alpha)
        return {'FINISHED'}

    def invoke(self, context, event):
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}


# class VMTImporter_OT_operator(bpy.types.Operator):
#     """Load Source Engine VMT material"""
#     bl_idname = "import_texture.vmt"
#     bl_label = "Import VMT"
#     bl_options = {'UNDO'}
#
#     filepath: StringProperty(
#         subtype='FILE_PATH',
#     )
#     files: CollectionProperty(type=bpy.types.PropertyGroup)
#     load_alpha: BoolProperty(default=True, name='Load alpha into separate image')
#
#     filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
#     game: StringProperty(name="PATH TO GAME", subtype='FILE_PATH', default="")
#     override: BoolProperty(default=False, name='Override existing?')
#
#     def execute(self, context):
#         if Path(self.filepath).is_file():
#             directory = Path(self.filepath).parent.absolute()
#         else:
#             directory = Path(self.filepath).absolute()
#         for file in self.files:
#             vmt = VMT(str(directory / file.name), self.game)
#             mat = BlenderMaterial(vmt)
#             mat.load_textures(self.load_alpha)
#             if mat.create_material(
#                     self.override) == 'EXISTS' and not self.override:
#                 self.report({'INFO'}, '{} material already exists')
#         return {'FINISHED'}
#
#     def invoke(self, context, event):
#         wm = context.window_manager
#         wm.fileselect_add(self)
#         return {'RUNNING_MODAL'}

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
             "RGBA8888 format, format-specific Eight Bit Alpha and Normal Map flags"),
            ('RGB888', "RGBA888 Simple", "RGB888 format, no alpha"),
            ('RGB888Normal', "RGB888 Normal Map", "RGB888 format, no alpha and Normal map flag"),
            ('DXT1', "DXT1 Simple", "DXT1 format, no flags"),
            ('DXT5', "DXT5 Simple", "DXT5 format, format-specific Eight Bit Alpha flag only"),
            ('DXT1Normal', "DXT1 Normal Map",
             "DXT1 format, Normal Map flag only"),
            ('DXT5Normal', "DXT5 Normal Map",
             "DXT5 format, format-specific Eight Bit Alpha and Normal Map flags")),
        default='RGBA8888',
    )

    def execute(self, context):
        sima = context.space_data
        ima = sima.image
        if ima is None:
            self.report({"ERROR_INVALID_INPUT"}, "No Image provided")
        else:
            print(context)
            export_texture(ima, self.filepath, self.img_format)
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
