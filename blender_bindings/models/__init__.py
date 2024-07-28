from pathlib import Path
from typing import Optional

from SourceIO.blender_bindings.models.model_tags import MODEL_HANDLERS, choose_model_importer
from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from . import mdl4, mdl6, mdl10, mdl36, mdl44, mdl49, md3_15
from ..operators.import_settings_base import ModelOptions
from ..shared.model_container import ModelContainer
from ...library.shared.app_id import SteamAppId
from ...logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('MDL loader')


def import_model(model_path: Path, buffer: Buffer,
                 content_manager: ContentManager,
                 options: ModelOptions,
                 override_steam_id: Optional[SteamAppId] = None,
                 ) -> Optional[ModelContainer]:
    ident, version = buffer.read_fmt("4sI")
    logger.info(f"Detected magic: {ident!r}, version:{version}")
    cp = content_manager.get_content_provider_from_asset_path(model_path)
    handler = choose_model_importer(ident, version, (override_steam_id or ((cp.steam_id or None) if cp else None)))
    if handler is None:
        logger.error(f"No handler found for ident {ident} version: {version}")
        return None
    buffer.seek(0)
    container = handler(model_path, buffer, content_manager, options)
    return container
