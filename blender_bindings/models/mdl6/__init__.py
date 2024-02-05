from pathlib import Path
from typing import Optional

from SourceIO.blender_bindings.models.model_tags import model_handler_tag
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.library.shared.content_providers.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from .import_mdl import import_model


@model_handler_tag(b"IDST", 6)
def import_mdl6(model_path: Path, buffer: Buffer,
                content_manager: ContentManager, options: ModelOptions) -> Optional[ModelContainer]:
    texture_mdl = content_manager.find_file(model_path.with_name(model_path.stem + "t.mdl"))

    return import_model(buffer, texture_mdl, options)
