from enum import Enum
from typing import Any


class ExtendedEnum(Enum):
    @classmethod
    def is_valid(cls, value: Any):
        for item in cls:
            if item.value == value:
                return True
        return False

    @classmethod
    def _missing_(cls, value):
        possible_member = cls._value2member_map_.get(value, None)

        if possible_member is None:
            possible_member = cls.create_pseudo_member(value)

        return possible_member

    @classmethod
    def create_pseudo_member(cls, value):
        """
        Create a default-name member.
        """

        default_member = cls._value2member_map_.get(0, None)

        if default_member is None:
            raise ValueError(f"{value!r} is not a valid {cls.__name__}")

        # construct a singleton enum pseudo-member
        other_member = int.__new__(cls)
        other_member._name_ = default_member._name_ + f"_{value}"
        other_member._value_ = value

        # use setdefault in case another thread already created a composite
        # with this value
        other_member = cls._value2member_map_.setdefault(value, other_member)

        return other_member