from abc import ABC


class SourceBone(ABC):

    @property
    def position(self):
        raise NotImplementedError

    @property
    def rotation_quat(self):
        raise NotImplementedError

    @property
    def rotation_euler(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError


