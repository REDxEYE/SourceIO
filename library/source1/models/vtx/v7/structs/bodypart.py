from dataclasses import dataclass

from SourceIO.library.utils import Buffer
from .model import Model
from ...v6.structs.bodypart import BodyPart as BodyPart6


@dataclass(slots=True)
class BodyPart(BodyPart6):
    MODEL_CLASS = Model
