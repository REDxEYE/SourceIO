import bpy
from bpy.props import StringProperty, CollectionProperty

from SourceIO.blender_bindings.utils.bpy_utils import is_blender_4_1
from SourceIO.library.utils.reporter import Reporter
from SourceIO.library.utils.tiny_path import TinyPath

class OperatorHelper(bpy.types.Operator):
    def report_errors(self, reporter: Reporter):
        for warning in reporter.warnings():
            self.report({"WARNING"}, str(warning))
        for error in reporter.errors():
            self.report({"ERROR"}, str(error))


class ImportOperatorHelper(OperatorHelper):
    need_popup = True
    if is_blender_4_1():
        directory: bpy.props.StringProperty(subtype='FILE_PATH', options={'SKIP_SAVE', 'HIDDEN'})
    filepath: StringProperty(subtype='FILE_PATH', )
    files: CollectionProperty(name='File paths', type=bpy.types.OperatorFileListElement)

    def invoke_popup(self, context, confirm_text=""):
        if self.properties.is_property_set("filepath"):
            title = self.filepath
            if len(self.files) > 1:
                title = f"Import {len(self.files)} files"

            if not confirm_text:
                confirm_text = self.bl_label
            return context.window_manager.invoke_props_dialog(self, confirm_text=confirm_text, title=title,
                                                              translate=False)

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if is_blender_4_1() and self.directory and self.files:
            if self.need_popup:
                return self.invoke_popup(context)
            else:
                return self.execute(context)
        wm = context.window_manager
        wm.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def get_directory(self) -> TinyPath:
        if is_blender_4_1():
            return TinyPath(self.directory)
        else:
            filepath = TinyPath(self.filepath)
            if filepath.is_file():
                return filepath.parent.absolute()
            else:
                return filepath.absolute()
