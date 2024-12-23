from typing import Optional

from SourceIO.blender_bindings.models.model_tags import register_model_importer
from SourceIO.blender_bindings.models.mdl49 import import_materials
from SourceIO.blender_bindings.models.mdl52.import_mdl import import_model
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.exceptions import RequiredFileNotFound
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.source1.phy import import_physics
from SourceIO.library.models.mdl.v52.mdl_file import MdlV52
from SourceIO.library.models.phy.phy import Phy
from SourceIO.library.models.vtx import open_vtx
from SourceIO.library.models.vvc import Vvc
from SourceIO.library.models.vvd import Vvd
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.path_utilities import find_vtx_cm
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('MDL loader')


@register_model_importer(b"IDST", 52)
def import_mdl52(model_path: TinyPath, buffer: Buffer,
                 content_manager: ContentManager, options: ModelOptions) -> Optional[ModelContainer]:
    mdl = MdlV52.from_buffer(buffer)
    vtx_buffer = find_vtx_cm(model_path, content_manager)
    vvd_buffer = content_manager.find_file(model_path.with_suffix(".vvd"))
    if vtx_buffer is None or vvd_buffer is None:
        logger.error(f"Could not find VTX and/or VVD file for {model_path}")
        raise RequiredFileNotFound(f"Could not find VTX and/or VVD file for {model_path}")
    vtx = open_vtx(vtx_buffer)
    vvd = Vvd.from_buffer(vvd_buffer)
    vvc_buffer = content_manager.find_file(model_path.with_suffix(".vvc"))
    if vvc_buffer is not None:
        vvc = Vvc.from_buffer(vvc_buffer)
    else:
        vvc = None

    if options.import_textures:
        try:
            import_materials(content_manager, mdl, use_bvlg=options.use_bvlg)
        except Exception as t_ex:
            logger.error(f'Failed to import materials, caused by {t_ex}')
            import traceback
            traceback.print_exc()

    container = import_model(content_manager, mdl, vtx, vvd, vvc, options.scale)
    if options.import_physics:
        phy_buffer = content_manager.find_file(model_path.with_suffix(".phy"))
        if phy_buffer is None:
            logger.error(f"Could not find PHY file for {model_path}")
        else:
            phy = Phy.from_buffer(phy_buffer)
            import_physics(phy, phy_buffer, mdl, container, options.scale)

    
    return container
