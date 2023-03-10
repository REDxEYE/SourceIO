import bpy
from bpy.props import (BoolProperty, CollectionProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)


def register_props():
    bpy.types.Scene.TextureCachePath = StringProperty(name="TextureCachePath", subtype="FILE_PATH")


def unregister_props():
    del bpy.types.Scene.TextureCachePath
