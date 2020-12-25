import os
from pathlib import Path

from .source1.vtf.VTFWrapper.VTFLib import VTFLib
from .utilities.singleton import SingletonMeta

NO_BPY = int(os.environ.get('NO_BPY', '0'))
custom_icons = {}
bl_info = {
    "name": "SourceIO",
    "author": "RED_EYE",
    "version": (3, 9, 1),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine assets",
    "description": "Source1/Source2 Engine assets(.mdl, .bsp, .vmt, .vtf, .vmdl_c, .vwrld_c, .vtex_c)"
                   "Notice that you cannot delete this addon via blender UI, remove it manually from addons folder",
    "category": "Import-Export"
}

if not NO_BPY:

    try:
        import bpy
    except ImportError:
        NO_BPY = 1
    from bpy.props import StringProperty, BoolProperty, CollectionProperty, EnumProperty, FloatProperty

    from .source1_operators import (BSPImport_OT_operator,
                                    MDLImport_OT_operator,
        # DMXImporter_OT_operator,
                                    VTFExport_OT_operator,
                                    VTFImport_OT_operator,
                                    VMTImport_OT_operator,
                                    export
                                    )
    from .source2_operators import (VMATImport_OT_operator,
                                    VTEXImport_OT_operator,
                                    VMDLImport_OT_operator,
                                    VWRLDImport_OT_operator
                                    )
    from .shared_operators import (ChangeSkin_OT_operator,
                                   LoadPlaceholder_OT_operator,
                                   SourceIOUtils_PT_panel, )


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
        bl_label = "Source Engine Assets"
        bl_idname = "IMPORT_MT_sourceio"

        def draw(self, context):
            crowbar_icon = custom_icons["main"]["crowbar_icon"]
            bsp_icon = custom_icons["main"]["bsp_icon"]
            vtf_icon = custom_icons["main"]["vtf_icon"]
            vmt_icon = custom_icons["main"]["vmt_icon"]
            model_doc_icon = custom_icons["main"]["model_doc_icon"]
            vmat_icon = custom_icons["main"]["vmat_icon"]
            vtex_icon = custom_icons["main"]["vtex_icon"]
            vwrld_icon = custom_icons["main"]["vwrld_icon"]
            layout = self.layout
            layout.operator(MDLImport_OT_operator.bl_idname, text="Source model (.mdl)",
                            icon_value=crowbar_icon.icon_id)
            layout.operator(BSPImport_OT_operator.bl_idname, text="Source map (.bsp)",
                            icon_value=bsp_icon.icon_id)
            layout.operator(VTFImport_OT_operator.bl_idname, text="Source texture (.vtf)",
                            icon_value=vtf_icon.icon_id)
            layout.operator(VMTImport_OT_operator.bl_idname, text="Source material (.vmt)",
                            icon_value=vmt_icon.icon_id)
            # layout.operator(DMXImporter_OT_operator.bl_idname, text="SFM session (.dmx)")
            layout.operator(VMDLImport_OT_operator.bl_idname, text="Source2 model (.vmdl)",
                            icon_value=model_doc_icon.icon_id)
            layout.operator(VWRLDImport_OT_operator.bl_idname, text="Source2 map (.vwrld)",
                            icon_value=vwrld_icon.icon_id)
            layout.operator(VTEXImport_OT_operator.bl_idname, text="Source2 texture (.vtex)",
                            icon_value=vtex_icon.icon_id)
            layout.operator(VMATImport_OT_operator.bl_idname, text="Source2 material (.vmat)",
                            icon_value=vmat_icon.icon_id)


    def menu_import(self, context):
        source_io_icon = custom_icons["main"]["sourceio_icon"]
        self.layout.menu(SourceIO_MT_Menu.bl_idname, icon_value=source_io_icon.icon_id)


    def load_icon(loader, filename, name):
        script_path = Path(os.path.dirname(__file__))
        loader.load(name, str(script_path / 'icons' / filename), 'IMAGE')


    def register_custom_icon():
        import bpy.utils.previews
        pcoll = bpy.utils.previews.new()
        load_icon(pcoll, 'sourceio_icon.png', "sourceio_icon")
        load_icon(pcoll, 'crowbar_icon.png', "crowbar_icon")
        load_icon(pcoll, 'bsp_icon.png', "bsp_icon")
        load_icon(pcoll, 'vtf_icon.png', "vtf_icon")
        load_icon(pcoll, 'vmt_icon.png', "vmt_icon")
        load_icon(pcoll, 'model_doc_icon.png', "model_doc_icon")
        load_icon(pcoll, 'vmat_icon.png', "vmat_icon")
        load_icon(pcoll, 'vtex_icon.png', "vtex_icon")
        load_icon(pcoll, 'vwrld_icon.png', "vwrld_icon")
        custom_icons["main"] = pcoll


    def unregister_custom_icon():
        import bpy.utils.previews
        for pcoll in custom_icons.values():
            bpy.utils.previews.remove(pcoll)
        custom_icons.clear()


    classes = (
        # Source1 stuff
        MDLImport_OT_operator,
        BSPImport_OT_operator,
        # DMXImporter_OT_operator,
        VTFExport_OT_operator,
        VTFImport_OT_operator,
        VMTImport_OT_operator,

        # Source2 stuff
        VMDLImport_OT_operator,
        VTEXImport_OT_operator,
        VMATImport_OT_operator,
        VWRLDImport_OT_operator,

        # Addon tools
        # SourceIOPreferences,
        SourceIO_MT_Menu,
        SourceIOUtils_PT_panel,
        LoadPlaceholder_OT_operator,
        ChangeSkin_OT_operator
    )

    try:
        register_, unregister_ = bpy.utils.register_classes_factory(classes)
    except:
        register_ = lambda: 0
        unregister_ = lambda: 0


    def register():
        register_custom_icon()
        register_()
        VTFLib()
        bpy.types.TOPBAR_MT_file_import.append(menu_import)
        bpy.types.IMAGE_MT_image.append(export)


    def unregister():
        bpy.types.TOPBAR_MT_file_import.remove(menu_import)
        bpy.types.IMAGE_MT_image.remove(export)
        vtf_lib = VTFLib()
        vtf_lib.shutdown()
        SingletonMeta.cleanup()
        vtf_lib.free_dll()
        del vtf_lib
        unregister_custom_icon()
        unregister_()
else:
    def register():
        pass


    def unregister():
        pass

if __name__ == "__main__":
    register()
