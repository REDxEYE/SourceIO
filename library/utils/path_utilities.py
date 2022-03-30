import platform
from pathlib import Path
import os


def pop_path_back(path: Path):
    if len(path.parts) > 1:
        return Path().joinpath(*path.parts[1:])
    else:
        return path


def pop_path_front(path: Path):
    if len(path.parts) > 1:
        return Path().joinpath(*path.parts[:-1])
    else:
        return path


def find_vtx(mdl_path: Path):
    possible_vtx_vertsion = [70, 80, 11, 90, 12]
    for vtx_version in possible_vtx_vertsion[::-1]:
        path = corrected_path(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path is not None and path.exists():
            return path


def find_vtx_cm(mdl_path: Path, content_manager):
    possible_vtx_vertsion = [70, 80, 11, 12, 90]
    for vtx_version in possible_vtx_vertsion[::-1]:
        path = content_manager.find_file(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path:
            return path


def backwalk_file_resolver(current_path, file_to_find):
    current_path = Path(current_path).absolute()
    file_to_find = Path(file_to_find)

    for _ in range(len(current_path.parts) - 1):
        second_part = file_to_find
        for _ in range(len(file_to_find.parts)):
            new_path = current_path / second_part
            if new_path.exists():
                return new_path

            second_part = pop_path_back(second_part)
        current_path = pop_path_front(current_path)


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
                print(os.path.join(root, name))
                return os.path.join(root, name)


def corrected_path(path: Path):
    if platform.system() == "Windows" or path.exists():  # Shortcut for windows
        return path
    root, *parts, fname = path.parts

    new_path = Path(root)
    for part in parts:
        for dir_name in new_path.iterdir():
            if dir_name.is_file():
                continue
            if dir_name.name.lower() == part.lower():
                new_path = dir_name
                break
    for file_name in new_path.iterdir():
        if file_name.is_file() and file_name.name.lower() == fname.lower():
            return file_name


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
    return root_path / 'materials'


class NonSourceInstall:

    def __init__(self, start_dir):
        self.start_dir = Path(start_dir)

    def find_file(self, filepath: str, additional_dir=None, extention=None, use_recursive=False):
        if additional_dir is not None:
            filepath = Path(additional_dir) / filepath

        if extention is not None:
            filepath = filepath.with_suffix(extention)

        return backwalk_file_resolver(self.start_dir, filepath)

    def find_texture(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vtf', use_recursive=use_recursive)

    def find_material(self, filepath, use_recursive=False):
        return self.find_file(filepath, 'materials',
                              extention='.vmt', use_recursive=use_recursive)


def get_mod_path(path: Path) -> Path:
    _path = path
    while len(path.parts) > 1:
        if (path / 'maps').exists():
            return path
        elif (path / 'materials').exists():
            return path
        elif (path / 'elements').exists():
            return path
        elif (path / 'models').exists() and path.parts[-1] != 'models' and \
                path.parts[-2] != 'materials' and path.parts[-1] != 'materials':
            return path
        if len(path.parts) == 1:
            return _path
        path = path.parent
    return _path


def is_valid_path(path: str):
    invalid_chars = '<>:"|?*'
    for char in invalid_chars:
        if char in path:
            return False
    return True
