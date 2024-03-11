from pathlib import Path
from typing import Optional

from SourceIO.blender_bindings.models.model_tags import register_model_importer
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from .import_mdl import import_model

@register_model_importer(b"IDST", 4)
def import_mdl4(model_path: Path, buffer: Buffer,
                 content_manager: ContentManager, options: ModelOptions) -> Optional[ModelContainer]:
    return import_model(model_path.stem, buffer, options)
