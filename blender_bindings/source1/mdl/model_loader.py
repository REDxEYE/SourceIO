from pathlib import Path
from typing import Union, BinaryIO, Optional

from ....logger import SLoggingManager

from ..bsp.import_bsp import BPSPropCache
from ...source1.mdl.v36.import_mdl import import_model as import_model_v36
from ...source1.mdl.v44.import_mdl import import_model as import_model_v44
from ...source1.mdl.v49.import_mdl import import_model as import_model_v49
from ....library.utils.byte_io_mdl import ByteIO
from ....library.utils.path_utilities import find_vtx, find_vtx_cm, corrected_path
from ....library.shared.content_providers.content_manager import ContentManager

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model_from_full_path(mdl_path: Path, scale=1.0,
                                create_drives=False,
                                re_use_meshes: bool = False,
                                unique_material_names: bool = False):
    if re_use_meshes:
        container = BPSPropCache().get_object(mdl_path)
        if container is not None:
            return container.clone()
    content_manager = ContentManager()
    if mdl_path.is_absolute():
        vtx_file = corrected_path(find_vtx(mdl_path))
        vvd_file = corrected_path(mdl_path.with_suffix('.vvd'))
        vvc_file = corrected_path(mdl_path.with_suffix('.vvc'))
        assert vtx_file.exists(), f"failed to find vtx file {mdl_path.with_suffix('')}"
        assert vvd_file.exists(), f"failed to find vvd file {vvc_file}"
        content_root = content_manager.get_content_provider_from_path(mdl_path)
        name = mdl_path.relative_to(content_root.root)

    else:
        name = mdl_path
        vvd_file = content_manager.find_file(mdl_path, extension='.vvd')
        vvc_file = content_manager.find_file(mdl_path, extension='.vcc')
        vtx_file = find_vtx_cm(mdl_path, content_manager)
        mdl_path = content_manager.find_file(mdl_path)

        assert vtx_file is not None, f"failed to find vtx file {mdl_path.with_suffix('')}"
        assert vvd_file is not None, f"failed to find vvd file {vvc_file}"

    return import_model_from_files(str(name), mdl_path, vvd_file, vtx_file, vvc_file, scale, create_drives,
                                   re_use_meshes,
                                   unique_material_names)


def import_model_from_files(name: Union[str, Path],
                            mdl_file: Union[str, Path, BinaryIO],
                            vvd_file: Optional[Union[str, Path, BinaryIO]],
                            vtx_file: Union[str, Path, BinaryIO],
                            vvc_file: Optional[Union[str, Path, BinaryIO]],
                            scale=1.0,
                            create_drives=False,
                            re_use_meshes: bool = False,
                            unique_material_names: bool = False):
    if mdl_file is None:
        logger.warn(f'Model {name} not found!')
        return

    if re_use_meshes:
        container = BPSPropCache().get_object(name)
        if container is not None:
            return container.clone()

    mdl_reader = ByteIO(mdl_file)
    magic, version = mdl_reader.read_fmt('4sI')
    mdl_reader.seek(0)
    assert magic == b'IDST', f'Unknown Mdl magic "{magic}", expected "IDST"'
    if 35 <= version <= 37:
        container = import_model_v36(mdl_file, vtx_file, scale, create_drives, re_use_meshes, unique_material_names)

    elif version == 44:
        container = import_model_v44(mdl_file, vvd_file, vtx_file, None, scale, create_drives, re_use_meshes,
                                     unique_material_names)

    elif 45 <= version <= 49:
        container = import_model_v49(mdl_file, vvd_file, vtx_file, None, scale, create_drives, re_use_meshes,
                                     unique_material_names)

    elif 49 < version <= 51:
        container = import_model_v49(mdl_file, vtx_file, vvd_file, vvc_file, scale, create_drives, re_use_meshes,
                                     unique_material_names)
    else:
        raise Exception(f'Unsupported version Mdl v{version}')
    if re_use_meshes:
        BPSPropCache().add_object(name, container)
    return container
