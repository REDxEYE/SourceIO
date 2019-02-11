import math
from pathlib import Path


def get_class_var_name(class_, var):
    a = class_.__dict__  # type: dict
    for k, v in a.items():
        if id(getattr(class_, k)) == id(var) and v == var:
            return k


def case_insensitive_file_resolution(path):
    '''
    There are a lot of cases where the .mdl file is lowercase whereas
    the .vvd/vtx files are mixed case. Resolving the file based on the
    same file name will work fine on case-insensitive
    operating systems (Windows 🤮) but on Linux (and some specific macOS
    installations) we need to work around this behavior by walking through
    the files in the directory and do a lowercase comparison on
    all the files in there.
    '''
    import os
    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if filename.lower() == name.lower():
                # print(os.path.join(root, name))
                return os.path.join(root, name)


def resolve_root_directory_from_file(path):
    if path.parts[-1] == 'models':
        return path.parent
    else:
        return resolve_root_directory_from_file(path.parent)


def get_materials_path(path):
    path = Path(path)
    root_path = resolve_root_directory_from_file(path)
    material_path = root_path / 'materials'


def convert_rotation_matrix_to_degrees(m0, m1, m2, m3, m4, m5, m8):
    angleY = -math.asin(round(m2, 6))
    c = math.cos(angleY)
    if abs(c) > 0.005:
        translateX = m8 / c
        translateY = -m5 / c
        angleX = (math.atan2(translateY, translateX))
        translateX = m0 / c
        translateY = -m1 / c
        angleZ = (math.atan2(translateY, translateX))
    else:
        angleX = 0
        translateX = m4
        translateY = m3
        angleZ = (math.atan2(translateY, translateX))
    return angleX, angleY, angleZ
