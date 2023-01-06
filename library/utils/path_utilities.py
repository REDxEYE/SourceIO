import os
import platform
from pathlib import Path


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
    possible_vtx_version = [70, 80, 11, None, 90, 12]
    for vtx_version in possible_vtx_version[::-1]:
        if vtx_version is None:
            path = corrected_path(mdl_path.with_suffix(f'.vtx'))
        else:
            path = corrected_path(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path is not None and path.exists():
            return path


def find_vtx_cm(mdl_path: Path, content_manager):
    possible_vtx_version = [70, 80, 11, None, 12, 90]
    for vtx_version in possible_vtx_version[::-1]:
        if vtx_version is None:
            path = content_manager.find_file(mdl_path.with_suffix(f'.vtx'))
        else:
            path = content_manager.find_file(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path:
            return path


def backwalk_file_resolver(current_path, file_to_find):
    current_path = Path(current_path).absolute()
    file_to_find = Path(file_to_find)

    for _ in range(len(current_path.parts) - 1):
        second_part = file_to_find
        for _ in range(len(file_to_find.parts)):
            new_path = corrected_path(current_path / second_part)
            if new_path.exists():
                return new_path

            second_part = pop_path_back(second_part)
        current_path = pop_path_front(current_path)


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
    return path


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
