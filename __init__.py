import os

NO_BPY = int(os.environ.get('NO_BPY', '0'))



bl_info = {
    "name": "Source1/Source2 Engine assets(.mdl, .vmdl_c, .vwrld_c, .vtex_c and etc)",
    "author": "RED_EYE",
    "version": (3, 7, 11),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine MDL (.mdl, .vmdl_c) ",
    "description": "Addon allows to import Source Engine models",
    "category": "Import-Export"
}

if not NO_BPY:

    try:
        import bpy
    except ImportError:
        NO_BPY = 1
    from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

    from .source1_operators import (BSPImporter_OT_operator,
                                    MDLImporter_OT_operator,
                                    DMXImporter_OT_operator,
                                    VTFExport_OT_operator,
                                    VTFImporter_OT_operator,
                                    export
                                    )
    from .source2_operators import (ChangeSkin_OT_operator,
                                    LoadPlaceholder_OT_operator,
                                    SourceIOUtils_PT_panel,
                                    VMATImporter_OT_operator,
                                    VTEXImporter_OT_operator,
                                    VMDLImporter_OT_operator,
                                    VWRLDImporter_OT_operator
                                    )


    # class SourceIOPreferences(bpy.types.AddonPreferences):
    #     bl_idname = __package__
    #
    #     sfm_path: StringProperty(default='', name='SFM path')
    #
    #     def draw(self, context):
    #         layout = self.layout
    #         layout.label(text='Enter SFM install path:')
    #         row = layout.row()
    #         row.prop(self, 'sfm_path')

    # noinspection PyPep8Naming
    class SourceIO_MT_Menu(bpy.types.Menu):
        bl_label = "Source engine"
        bl_idname = "IMPORT_MT_sourceio"

        def draw(self, context):
            layout = self.layout
            layout.operator(MDLImporter_OT_operator.bl_idname, text="Source model (.mdl)")
            layout.operator(BSPImporter_OT_operator.bl_idname, text="Source map (.bsp)")
            layout.operator(VTFImporter_OT_operator.bl_idname, text="Source texture (.vtf)")
            # self.layout.operator(VMTImporter_OT_operator.bl_idname, text="Source material (.vmt)")
            layout.operator(DMXImporter_OT_operator.bl_idname, text="SFM session (.dmx)")
            layout.operator(VMDLImporter_OT_operator.bl_idname, text="Source2 model (.vmdl)")
            layout.operator(VWRLDImporter_OT_operator.bl_idname, text="Source2 map (.vwrld)")
            layout.operator(VTEXImporter_OT_operator.bl_idname, text="Source2 texture (.vtex)")
            layout.operator(VMATImporter_OT_operator.bl_idname, text="Source2 material (.vmat)")


    def menu_import(self, context):
        self.layout.menu(SourceIO_MT_Menu.bl_idname)


    # VMTImporter_OT_operator,
    classes = (MDLImporter_OT_operator, BSPImporter_OT_operator, DMXImporter_OT_operator,
               VTFExport_OT_operator, VTFImporter_OT_operator,
               VMDLImporter_OT_operator, VTEXImporter_OT_operator, VMATImporter_OT_operator, VWRLDImporter_OT_operator,
               # SourceIOPreferences,
               SourceIO_MT_Menu, SourceIOUtils_PT_panel, LoadPlaceholder_OT_operator,
               ChangeSkin_OT_operator)

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
