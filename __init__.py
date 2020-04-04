import os
from pathlib import Path

NO_BPY = int(os.environ.get('NO_BPY', '0'))

bl_info = {
    "name": "Source1/Source2 Engine model(.mdl, .vvd, .vtx)",
    "author": "RED_EYE",
    "version": (3, 7, 1),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine MDL (.mdl, .vmdl_c) ",
    "description": "Addon allows to import Source Engine models",
    "category": "Import-Export"
}

if not NO_BPY:

    import bpy
    from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty

    from .source1.mdl import mdl2model, qc_generator
    from .source1.vtf.blender_material import BlenderMaterial
    from .source1.vtf.export_vtf import export_texture
    from .source1.vtf.import_vtf import import_texture
    from .source1.dmx.dmx import Session

    from .source2.vmdl import Vmdl
    from .source2.vtex import Vtex

    from .source1.vtf.vmt import VMT


    class SourceIOPreferences(bpy.types.AddonPreferences):
        bl_idname = __package__

        sfm_path: StringProperty(default='', name='SFM path')

        def draw(self, context):
            layout = self.layout
            layout.label(text='Enter SFM install path:')
            row = layout.row()
            row.prop(self, 'sfm_path')


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

        import_textures: BoolProperty(name="Import textures", default=False, subtype='UNSIGNED')

        filter_glob: StringProperty(default="*.mdl", options={'HIDDEN'})

        def execute(self, context):

            if Path(self.filepath).is_file():
                directory = Path(self.filepath).parent.absolute()
            else:
                directory = Path(self.filepath).absolute()
            for file in self.files:
                importer = mdl2model.Source2Blender(str(directory / file.name),
                                                    normal_bones=self.normal_bones,
                                                    join_clamped=self.join_clamped,
                                                    import_textures=self.import_textures,
                                                    context=context
                                                    )
                importer.sort_bodygroups = self.organize_bodygroups
                importer.load(dont_build_mesh=False)
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


    # noinspection PyUnresolvedReferences
    class VMDLImporter_OT_operator(bpy.types.Operator):
        """Load Source Engine MDL models"""
        bl_idname = "source_io.vmdl"
        bl_label = "Import Source VMDL file"
        bl_options = {'UNDO'}

        filepath: StringProperty(subtype="FILE_PATH")
        files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

        filter_glob: StringProperty(default="*.vmdl_c", options={'HIDDEN'})

        def execute(self, context):

            if Path(self.filepath).is_file():
                directory = Path(self.filepath).parent.absolute()
            else:
                directory = Path(self.filepath).absolute()
            for file in self.files:
                Vmdl(str(directory / file.name), True)
            return {'FINISHED'}

        def invoke(self, context, event):
            wm = context.window_manager
            wm.fileselect_add(self)
            return {'RUNNING_MODAL'}


    class DMXImporter_OT_operator(bpy.types.Operator):
        """Load Source Engine MDL models"""
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


    class VTEXImporter_OT_operator(bpy.types.Operator):
        """Load Source Engine VTF texture"""
        bl_idname = "import_texture.vtex"
        bl_label = "Import VTEX"
        bl_options = {'UNDO'}

        filepath: StringProperty(subtype='FILE_PATH', )
        files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
        filter_glob: StringProperty(default="*.vtex_c", options={'HIDDEN'})

        def execute(self, context):
            if Path(self.filepath).is_file():
                directory = Path(self.filepath).parent.absolute()
            else:
                directory = Path(self.filepath).absolute()
            for file in self.files:
                Vtex(str(directory / file.name)).load()
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
                ('DXT5', "DXT5 Simple","DXT5 format, format-specific Eight Bit Alpha flag only"),
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


    def menu_import(self, context):
        self.layout.operator(MDLImporter_OT_operator.bl_idname, text="Source model (.mdl)")
        self.layout.operator(VTFImporter_OT_operator.bl_idname, text="Source texture (.vtf)")
        # self.layout.operator(VMTImporter_OT_operator.bl_idname, text="Source material (.vmt)")
        self.layout.operator(DMXImporter_OT_operator.bl_idname, text="SFM session (.dmx)")
        self.layout.operator(VMDLImporter_OT_operator.bl_idname, text="Source2 model (.vmdl)")
        self.layout.operator(VTEXImporter_OT_operator.bl_idname, text="Source2 texture (.vtex)")

    #VMTImporter_OT_operator,
    classes = (MDLImporter_OT_operator, VTFExport_OT_operator, VTFImporter_OT_operator,
               DMXImporter_OT_operator, SourceIOPreferences, VMDLImporter_OT_operator, VTEXImporter_OT_operator)
    try:
        register_, unregister_ = bpy.utils.register_classes_factory(classes)
    except:
        register_ = lambda: 0
        unregister_ = lambda: 0


    def register():
        register_()
        bpy.types.TOPBAR_MT_file_import.append(menu_import)
        bpy.types.IMAGE_MT_image.append(export)


    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)
        bpy.types.IMAGE_MT_image.remove(export)
        unregister_()
else:
    def register():
        pass


    def unregister():
        pass

if __name__ == "__main__":
    register()
