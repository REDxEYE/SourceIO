import os
import random
from typing import Dict

from .utilities.singleton import SingletonMeta

NO_BPY = int(os.environ.get('NO_BPY', '0'))
if NO_BPY:
    class BPYLoggingManager(metaclass=SingletonMeta):
        def __init__(self):
            self.loggers: Dict[str, BPYLogger] = {}

        def get_logger(self, name):
            logger = self.loggers[name] = BPYLogger(name)
            return logger


    class BPYLogger:
        def __init__(self, name):
            self.name = name

        def log(self, log_level, message, module=None):
            print(f'[{log_level:8}]--[{f"{module}:" if module is not None else ""}{self.name}] {message}')

        def debug(self, message, module=None):
            self.log('DEBUG', message, module)

        def info(self, message, module=None):
            self.log('INFO', message, module)

        def warn(self, message, module=None):
            self.log('WARN', message, module)

        def error(self, message, module=None):
            self.log('ERROR', message, module)


else:
    import bpy


    def get_material(mat_name, model_ob):
        if not mat_name:
            mat_name = "Material"
        mat_ind = 0
        md = model_ob.data
        mat = None
        for candidate in bpy.data.materials:  # Do we have this material already?
            if candidate.name == mat_name:
                mat = candidate
        if mat:
            if md.materials.get(mat.name):  # Look for it on this mesh_data
                for i in range(len(md.materials)):
                    if md.materials[i].name == mat.name:
                        mat_ind = i
                        break
            else:  # material exists, but not on this mesh_data
                md.materials.append(mat)
                mat_ind = len(md.materials) - 1
        else:  # material does not exist
            mat = bpy.data.materials.new(mat_name)
            md.materials.append(mat)
            # Give it a random colour
            rand_col = [random.uniform(.4, 1) for _ in range(3)]
            rand_col.append(1.0)
            mat.diffuse_color = rand_col

            mat_ind = len(md.materials) - 1

        return mat_ind


    def get_or_create_collection(name, parent: bpy.types.Collection) -> bpy.types.Collection:
        new_collection = (bpy.data.collections.get(name, None) or
                          bpy.data.collections.new(name))
        if new_collection.name not in parent.children:
            parent.children.link(new_collection)
        new_collection.name = name
        return new_collection

    def get_log_file(filename):
        return bpy.data.texts.get(filename, None) or bpy.data.texts.new(filename)


    class BPYLoggingManager(metaclass=SingletonMeta):
        def __init__(self):
            self.loggers: Dict[str, BPYLogger] = {}

        def get_logger(self, name):
            logger = self.loggers[name] = BPYLogger(name)
            return logger


    class BPYLogger:
        def __init__(self, name):
            self.name = name

        def log(self, log_level, message, module=None):
            file = get_log_file(self.name)
            file.write(f'[{log_level:8}]-[{f"{module}:" if module is not None else ""}{self.name}] {message}\n')
            print(f'[{log_level:8}]-[{self.name}] {message}')

        def debug(self, message, module=None):
            self.log('DEBUG', message, module)

        def info(self, message, module=None):
            self.log('INFO', message, module)

        def warn(self, message, module=None):
            self.log('WARN', message, module)

        def error(self, message, module=None):
            self.log('ERROR', message, module)
