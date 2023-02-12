from enum import Enum
from typing import Any


class ExtendedEnum(Enum):
    @classmethod
    def is_valid(cls, value: Any):
        for item in cls:
            if item.value == value:
                return True
        return False
