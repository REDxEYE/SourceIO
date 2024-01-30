from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .strip_group import StripGroup
from ...v6.structs.mesh import Mesh as Mesh6


@dataclass(slots=True)
class Mesh(Mesh6):
    STRIP_GROUP_CLASS = StripGroup
