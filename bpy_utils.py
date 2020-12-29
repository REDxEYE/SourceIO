import bpy


def get_log_file(filename):
    return bpy.data.texts.get(filename, None) or bpy.data.texts.new(filename)
