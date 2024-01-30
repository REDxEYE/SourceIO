from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .lod import ModelLod
from ...v6.structs.model import Model as Model6


@dataclass(slots=True)
class Model(Model6):
    MODEL_LOD_CLASS = ModelLod


