from pathlib import Path
from typing import Union

from ....library.shared.content_providers.content_manager import ContentManager
from ....library.utils import FileBuffer
from ....library.utils.path_utilities import find_vtx, find_vtx_cm
from ....logger import SLoggingManager
from ...shared.model_container import Source1ModelContainer
from ...source1.mdl.v36.import_mdl import import_model as import_model_v36
from ...source1.mdl.v44.import_mdl import import_model as import_model_v44
from ...source1.mdl.v49.import_mdl import import_model as import_model_v49
from ...source1.mdl.v52.import_mdl import import_model as import_model_v52
from ..phy import import_physics
from . import FileImport

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1::ModelLoader')


def import_model_from_full_path(mdl_path: Path,
                                scale=1.0,
                                create_drives=False,
                                re_use_meshes: bool = False,
                                unique_material_names: bool = False,
                                bodygroup_grouping: bool = True,
                                load_physics: bool = False,
                                load_refpose: bool = False
                                ) -> Source1ModelContainer:
    content_manager = ContentManager()
    if mdl_path.is_absolute():
        mdl_file = FileBuffer(mdl_path)
        vtx_path = find_vtx(mdl_path)
        if vtx_path is None:
            raise FileNotFoundError(f"Failed to find vtx file for {mdl_path} model")
        vtx_file = FileBuffer(vtx_path)
        vvd_file = FileBuffer(mdl_path.with_suffix('.vvd')) if mdl_path.with_suffix('.vvd').exists() else None
        vvc_file = FileBuffer(mdl_path.with_suffix('.vvc')) if mdl_path.with_suffix('.vvc').exists() else None
        phy_file = (FileBuffer(mdl_path.with_suffix('.phy'))
                    if load_physics and mdl_path.with_suffix('.phy').exists() else None)
        content_root = content_manager.get_content_provider_from_path(mdl_path)
        name = mdl_path.relative_to(content_root.root)

    else:
        name = mdl_path
        vvd_file = content_manager.find_file(mdl_path, extension='.vvd')
        vvc_file = content_manager.find_file(mdl_path, extension='.vcc')
        file = content_manager.find_file(mdl_path, extension='.phy')
        phy_file = file if load_physics and file else None
        vtx_file = find_vtx_cm(mdl_path, content_manager)
        mdl_file = content_manager.find_file(mdl_path)

    file_list = FileImport(mdl_file, vvd_file, vtx_file, vvc_file, phy_file)

    return import_model_from_files(str(name), file_list, scale,
                                   create_drives,
                                   re_use_meshes,
                                   unique_material_names,
                                   bodygroup_grouping,
                                   load_physics,
                                   load_refpose)


def import_model_from_files(name: Union[str, Path],
                            file_list: FileImport,
                            scale=1.0,
                            create_drives=False,
                            re_use_meshes: bool = False,
                            unique_material_names: bool = False,
                            bodygroup_grouping: bool = True,
                            load_physics: bool = False,
                            load_refpose: bool = False
                            ) -> Source1ModelContainer:
    mdl_reader = file_list.mdl_file
    magic, version = mdl_reader.read_fmt('4sI')
    mdl_reader.seek(0)
    assert magic == b'IDST', f'Unknown Mdl magic "{magic}", expected "IDST"'
    if version != 2531 and version > 37:
        assert file_list.vvd_file is not None, f".VVD file is required for this mdl {version} version"

    assert file_list.vtx_file is not None, f".VTX file is required for this mdl {version} version"

    if 35 <= version <= 37:
        container = import_model_v36(file_list, scale, create_drives, re_use_meshes, unique_material_names,
                                     load_refpose)

    elif version == 44:
        container = import_model_v44(file_list, scale, create_drives, re_use_meshes, unique_material_names,
                                     load_refpose)

    elif 45 <= version <= 49:
        container = import_model_v49(file_list, scale, create_drives, re_use_meshes, unique_material_names,
                                     load_refpose)

    elif version == 52:
        container = import_model_v52(file_list, scale, create_drives, re_use_meshes, unique_material_names,
                                     load_refpose)
    else:
        raise Exception(f'Unsupported version Mdl v{version}')
    # if re_use_meshes:
    #     BPSPropCache().add_object(name, container)

    # put_into_collections(container, Path(container.mdl.header.name).stem, bodygroup_grouping=bodygroup_grouping)

    if load_physics:
        import_physics(file_list, container, scale)
    return container
