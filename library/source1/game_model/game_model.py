from abc import ABC
from typing import List

from .parts.bone import SourceBone
from .parts.mesh import SourceMesh


class GameModel(ABC):
    @property
    def bones(self) -> List[SourceBone]:
        raise NotImplementedError

    @property
    def meshes(self) -> List[SourceMesh]:
        raise NotImplementedError
