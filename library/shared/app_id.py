from enum import IntEnum
from typing import Any


class SteamAppId(IntEnum):
    UNKNOWN = 0
    HALF_LIFE_2 = 220
    HALF_LIFE_2_EP_1 = 380
    HALF_LIFE_2_EP_2 = 420
    PORTAL = 400
    PORTAL_2 = 620
    LEFT_4_DEAD = 500
    LEFT_4_DEAD_2 = 550
    TEAM_FORTRESS_2 = 440
    COUNTER_STRIKE_GO = 730
    SOURCE_FILMMAKER = 1840
    GARRYS_MOD = 4000
    BLACK_MESA = 362890
    HLA_STEAM_ID = 546560
    SBOX_STEAM_ID = 590830
    VINDICTUS = 212160
    THINKING_WITH_TIME_MACHINE = 286080
    PORTAL_STORIES_MEL = 317400

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
            raise ValueError("%r is not a valid %s" % (value, cls.__name__))

        # construct a singleton enum pseudo-member
        other_member = int.__new__(cls)
        other_member._name_ = default_member._name_+f"_{value}"
        other_member._value_ = value

        # use setdefault in case another thread already created a composite
        # with this value
        other_member = cls._value2member_map_.setdefault(value, other_member)

        return other_member

    def __ne__(self, other):
        return not self == other
