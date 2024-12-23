import os
import platform
from typing import Optional

from SourceIO.library.utils import TinyPath


def pop_path_back(path: TinyPath):
    if len(path.parts) > 1:
        return TinyPath(os.sep.join(path.parts[1:]))
    else:
        return path


def pop_path_front(path: TinyPath):
    if len(path.parts) > 1:
        return TinyPath(os.sep.join(path.parts[:-1]))
    else:
        return path


def find_vtx(mdl_path: TinyPath):
    possible_vtx_version = [70, 80, 11, None, 90, 12]
    for vtx_version in possible_vtx_version[::-1]:
        if vtx_version is None:
            path = corrected_path(mdl_path.with_suffix(f'.vtx'))
        else:
            path = corrected_path(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path is not None and path.exists():
            return path


def find_vtx_cm(mdl_path: TinyPath, content_manager):
    possible_vtx_version = [70, 80, 11, None, 12, 90]
    for vtx_version in possible_vtx_version[::-1]:
        if vtx_version is None:
            path = content_manager.find_file(mdl_path.with_suffix(f'.vtx'))
        else:
            path = content_manager.find_file(mdl_path.with_suffix(f'.dx{vtx_version}.vtx'))
        if path:
            return path


def backwalk_file_resolver(current_path, file_to_find) -> Optional[TinyPath]:
    current_path = TinyPath(current_path).absolute()
    file_to_find = TinyPath(file_to_find)

    for _ in range(len(current_path.parts) - 1):
        second_part = file_to_find
        for _ in range(len(file_to_find.parts)):
            new_path = corrected_path(current_path / second_part)
            if new_path.exists():
                return new_path

            second_part = pop_path_back(second_part)
        current_path = pop_path_front(current_path)
    return None


def corrected_path(path: TinyPath):
    if platform.system() == "Windows" or path.exists():  # Shortcut for windows
        return path
    root, *parts, fname = path.parts

    new_path = TinyPath(root)
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


def get_mod_path(path: TinyPath) -> TinyPath:
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


def path_stem(path: str):
    return os.path.basename(path).rsplit(".", 1)[0]


def collect_full_material_names(material_names: list[str], material_search_paths: list[str],
                                content_manager) -> dict[str, str]:
    full_mat_names = {}
    for material_path in material_search_paths:
        for material_name in material_names:
            if material_name in full_mat_names:
                continue
            real_material_path = content_manager.find_file(
                "materials" / TinyPath(material_path) / (material_name + ".vmt"))
            if real_material_path is not None:
                full_mat_names[material_name] = (TinyPath(material_path) / material_name).as_posix()
    for material_name in material_names:
        if material_name not in full_mat_names:
            full_mat_names[material_name] = material_name
    return full_mat_names
