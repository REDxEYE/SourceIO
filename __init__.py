import bpy
from pathlib import Path

bl_info = {
    "name": "Source Engine model(.mdl, .vvd, .vtx)",
    "author": "RED_EYE",
    "version": (3, 3),
    "blender": (2, 80, 0),
    "location": "File > Import-Export > SourceEngine MDL (.mdl, .vvd, .vtx) ",
    "description": "Addon allows to import Source Engine models",
    # 'warning': 'May crash blender',
    # "wiki_url": "http://www.barneyparker.com/blender-json-import-export-plugin",
    # "tracker_url": "http://www.barneyparker.com/blender-json-import-export-plugin",
    "category": "Import-Export"
}

from bpy.props import StringProperty, BoolProperty, CollectionProperty


class MDLImporter_OT_operator(bpy.types.Operator):
    """Load Source Engine MDL models"""
    bl_idname = "source_io.mdl"
    bl_label = "Import Source MDL file"
    bl_options = {'UNDO'}

    filepath = StringProperty(
        subtype='FILE_PATH',
    )
    files = CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)
    normal_bones = BoolProperty(name="Normalize bones", default=False, subtype='UNSIGNED')
    join_clamped = BoolProperty(name="Join clamped meshes", default=False, subtype='UNSIGNED')
    organize_bodygroups = BoolProperty(name="Organize bodygroups", default=True, subtype='UNSIGNED')
    write_qc = BoolProperty(name="Write QC file", default=True, subtype='UNSIGNED')
    filter_glob = StringProperty(default="*.mdl", options={'HIDDEN'})

    def execute(self, context):
        from . import mdl2model
        directory = Path(self.filepath).parent.absolute()
        for file in self.files:
            importer = mdl2model.Source2Blender(str(directory / file.name),
                                                normal_bones=self.normal_bones,
                                                join_clamped=self.join_clamped,
                                                )
            importer.sort_bodygroups = self.organize_bodygroups
            importer.load()
            if self.write_qc:
                import qc_renerator
                qc = qc_renerator.QC(importer.model)
                qc_file = bpy.data.texts.new('{}.qc'.format(Path(file.name).stem))
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


classes = ( MDLImporter_OT_operator,)
register_, unregister_ = bpy.utils.register_classes_factory(classes)

def menu_import(self, context):
    self.layout.operator(MDLImporter_OT_operator.bl_idname, text="Source model (.mdl)")
    # self.layout.operator(VmeshImporter_OT_operator.bl_idname, text="Source2 mesh (.vmesh_c)")
    # self.layout.operator(VmdlImporter_OT_operator.bl_idname, text="Source2 model (.vmdl_c)")


def register():
    register_()
    # bpy.utils.register_module(__name__)
    bpy.types.TOPBAR_MT_file_import.append(menu_import)


def unregister():
    # bpy.utils.unregister_module(__name__)
    bpy.types.TOPBAR_MT_file_import.remove(menu_import)
    unregister_()


if __name__ == "__main__":
    register()
