from pathlib import Path
import os


def get_class_var_name(class_, var):
    a = class_.__dict__  # type: dict
    for k, v in a.items():
        if id(getattr(class_, k)) == id(var) and v == var:
            return k


def case_insensitive_file_resolution(path):
    """
    There are a lot of cases where the .mdl file is lowercase whereas
    the .vvd/vtx files are mixed case. Resolving the file based on the
    same file name will work fine on case-insensitive
    operating systems (Windows 🤮) but on Linux (and some specific macOS
    installations) we need to work around this behavior by walking through
    the files in the directory and do a lowercase comparison on
    all the files in there.
    """

    directory = os.path.dirname(path)
    filename = os.path.basename(path)
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            if filename.lower() == name.lower():
                # print(os.path.join(root, name))
                return os.path.join(root, name)


def resolve_root_directory_from_file(path):
    if type(path) is not Path:
        path = Path(path)
    if len(path.parts) < 2 or path == path.parent:
        return None
    if path.parts[-1] == 'models':
        return path.parent
    else:
        try:
            return resolve_root_directory_from_file(path.parent)
        except RecursionError:
            return None



def get_materials_path(path):
    path = Path(path)
    root_path = resolve_root_directory_from_file(path)
    material_path = root_path / 'materials'
    return material_path
