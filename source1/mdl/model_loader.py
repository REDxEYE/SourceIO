from pathlib import Path
from typing import Union, BinaryIO, Optional

from ...content_providers.content_manager import ContentManager

from .v36.import_mdl import import_model as import_model_v36
from .v44.import_mdl import import_model as import_model_v44
from .v49.import_mdl import import_model as import_model_v49
from ...utilities.byte_io_mdl import ByteIO
from ...utilities.path_utilities import find_vtx


def import_model_from_full_path(mdl_path: Path, scale=1.0,
                                create_drives=False,
                                re_use_meshes: bool = False,
                                unique_material_names: bool = False):
    content_manager = ContentManager()
    content_root = content_manager.get_content_provider_from_path(mdl_path)
    rel_mdl_path = mdl_path.relative_to(content_root.root)
    vtx_file = find_vtx(mdl_path)
    vvd_file = content_manager.find_file(rel_mdl_path.with_suffix('.vvd'))
    vvc_file = content_manager.find_file(rel_mdl_path.with_suffix('.vvc'))
    return import_model_from_files(mdl_path, vvd_file, vtx_file, vvc_file, scale, create_drives, re_use_meshes,
                                   unique_material_names)


def import_model_from_files(mdl_file: Union[str, Path, BinaryIO],
                            vvd_file: Optional[Union[str, Path, BinaryIO]],
                            vtx_file: Union[str, Path, BinaryIO],
                            vvc_file: Optional[Union[str, Path, BinaryIO]],
                            scale=1.0,
                            create_drives=False,
                            re_use_meshes: bool = False,
                            unique_material_names: bool = False):
    mdl_reader = ByteIO(mdl_file)
    magic, version = mdl_reader.read_fmt('4sI')
    mdl_reader.seek(0)
    assert magic == b'IDST', f'Unknown Mdl magic "{magic}", expected "IDST"'
    if 35 <= version <= 37:
        return import_model_v36(mdl_file, vtx_file, scale, create_drives, re_use_meshes, unique_material_names)

    elif version == 44:
        return import_model_v44(mdl_file, vvd_file, vtx_file, None, scale, create_drives, re_use_meshes,
                                unique_material_names)

    elif 45 <= version <= 49:
        return import_model_v49(mdl_file, vvd_file, vtx_file, None, scale, create_drives, re_use_meshes,
                                unique_material_names)

    elif 49 < version <= 51:
        return import_model_v49(mdl_file, vtx_file, vvd_file, vvc_file, scale, create_drives, re_use_meshes,
                                unique_material_names)

    raise Exception(f'Unsupported version Mdl v{version}')
