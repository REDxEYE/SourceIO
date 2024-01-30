from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .mesh import Mesh
from ...v6.structs.lod import ModelLod as ModelLod6


@dataclass(slots=True)
class ModelLod(ModelLod6):
    MESH_CLASS = Mesh
