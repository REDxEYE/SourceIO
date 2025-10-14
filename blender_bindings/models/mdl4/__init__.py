from typing import Optional

from SourceIO.blender_bindings.models.model_tags import register_model_importer
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.tiny_path import TinyPath
from .import_mdl import import_model


@register_model_importer(b"IDST", 4)
def import_mdl4(model_path: TinyPath, buffer: Buffer,
                content_manager: ContentProvider, options: ModelOptions) -> ModelContainer | None:
    return import_model(model_path.stem, buffer, options)
