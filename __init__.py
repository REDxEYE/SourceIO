import os
from pathlib import Path

no_bpy = False
try:
    import bpy
    from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty
except ImportError:
    no_bpy = True
    print('No BPY')

if not no_bpy:
    from .mdl import mdl2model
    from .vtf.blender_material import BlenderMaterial
    from .vtf.export_vtf import export_texture
    from .vtf.import_vtf import import_texture
from .mdl import qc_generator
from .vtf.vmt import VMT

bl_info = {
    "name": "Source Engine model(.mdl, .vvd, .vtx)",
    "author": "RED_EYE",
    "version": (3, 4, 1),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine MDL (.mdl, .vvd, .vtx) ",
    "description": "Addon allows to import Source Engine models",
    # 'warning': 'May crash blender',
    # "wiki_url": "http://www.barneyparker.com/blender-json-import-export-plugin",
    # "tracker_url": "http://www.barneyparker.com/blender-json-import-export-plugin",
    "category": "Import-Export"
}

if not no_bpy:
    # noinspection PyUnresolvedReferences
    class MDLImporter_OT_operator(bpy.types.Operator):
        """Load Source Engine MDL models"""
        bl_idname = "source_io.mdl"
        bl_label = "Import Source MDL file"
        bl_options = {'UNDO'}

        filepath: StringProperty(subtype="FILE_PATH")
        files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
        normal_bones: BoolProperty(name="Normalize bones", default=False, subtype='UNSIGNED')
        join_clamped: BoolProperty(name="Join clamped meshes", default=False, subtype='UNSIGNED')
        organize_bodygroups: BoolProperty(name="Organize bodygroups", default=True, subtype='UNSIGNED')
        write_qc: BoolProperty(name="Write QC file", default=True, subtype='UNSIGNED')
        filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

        def execute(self, context):

            directory = Path(self.filepath).parent.absolute()
            for file in self.files:
                importer = mdl2model.Source2Blender(str(directory / file.name),
                                                    normal_bones=self.normal_bones,
                                                    join_clamped=self.join_clamped,
                                                    )
                importer.sort_bodygroups = self.organize_bodygroups
                importer.load()
                if self.write_qc:
                    qc = qc_generator.QC(importer.model)
                    qc_file = bpy.data.texts.new(
                        '{}.qc'.format(Path(file.name).stem))
                    qc.write_header(qc_file)
                    qc.write_models(qc_file)
                    qc.write_skins(qc_file)
                    qc.write_misc(qc_file)
                    qc.write_sequences(qc_file)
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
            directory = Path(self.filepath).parent.absolute()
            for file in self.files:
                import_texture(str(directory / file.name),
                               self.load_alpha, self.only_alpha)
            return {'FINISHED'}

        def invoke(self, context, event):
            wm = context.window_manager
            wm.fileselect_add(self)
            return {'RUNNING_MODAL'}


    class VMTImporter_OT_operator(bpy.types.Operator):
        """Load Source Engine VMT material"""
        bl_idname = "import_texture.vmt"
        bl_label = "Import VMT"
        bl_options = {'UNDO'}

        filepath: StringProperty(
            subtype='FILE_PATH',
        )

        filter_glob: StringProperty(default="*.vmt", options={'HIDDEN'})
        game: StringProperty(name="PATH TO GAME", subtype='FILE_PATH', default="")
        override: BoolProperty(default=False, name='Override existing?')

        def execute(self, context):
            vmt = VMT(self.filepath, self.game)
            mat = BlenderMaterial(vmt)
            mat.load_textures()
            if mat.create_material(
                    self.override) == 'EXISTS' and not self.override:
                self.report({'INFO'}, '{} material already exists')
            return {'FINISHED'}

        def invoke(self, context, event):
            wm = context.window_manager
            wm.fileselect_add(self)
            return {'RUNNING_MODAL'}


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

        imgFormat: EnumProperty(
            name="VTF Type Preset",
            description="Choose a preset. It will affect the result's format and flags.",
            items=(('RGBA8888Simple', "RGBA8888 Simple", "RGBA8888 format, format-specific Eight Bit Alpha flag only"),
                   ('RGBA8888Normal', "RGBA8888 Normal Map",
                    "RGBA8888 format, format-specific Eight Bit Alpha and Normal Map flags"),
                   ('DXT1Simple', "DXT1 Simple", "DXT1 format, no flags"),
                   ('DXT5Simple', "DXT5 Simple",
                    "DXT5 format, format-specific Eight Bit Alpha flag only"),
                   ('DXT1Normal', "DXT1 Normal Map",
                    "DXT1 format, Normal Map flag only"),
                   ('DXT5Normal', "DXT5 Normal Map",
                    "DXT5 format, format-specific Eight Bit Alpha and Normal Map flags")),
            default='RGBA8888Simple',
        )

        def execute(self, context):
            sima = context.space_data
            ima = sima.image
            if ima is None:
                self.report({"ERROR_INVALID_INPUT"}, "No Image provided")
            else:
                print(context)
                export_texture(ima, self.filepath, self.imgFormat)
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


    def menu_import(self, context):
        self.layout.operator(MDLImporter_OT_operator.bl_idname, text="Source model (.mdl)")
        self.layout.operator(VTFImporter_OT_operator.bl_idname, text="Source texture (.vtf)")
        self.layout.operator(VMTImporter_OT_operator.bl_idname, text="Source material (.vmt)")

if not no_bpy:
    classes = (MDLImporter_OT_operator, VMTImporter_OT_operator, VTFExport_OT_operator, VTFImporter_OT_operator)
    try:
        register_, unregister_ = bpy.utils.register_classes_factory(classes)
    except:
        register_ = lambda: 0
        unregister_ = lambda: 0

if not no_bpy:
    def register():
        register_()
        bpy.types.TOPBAR_MT_file_import.append(menu_import)
        bpy.types.IMAGE_MT_image.append(export)


    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)
        bpy.types.IMAGE_MT_image.remove(export)
        unregister_()

if __name__ == "__main__":
    if not no_bpy:
        register()
