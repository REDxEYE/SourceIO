from abc import ABC


class SourceMesh(ABC):

    @property
    def vertices(self):
        raise NotImplementedError

    @property
    def indices(self):
        raise NotImplementedError

    @property
    def normals(self):
        raise NotImplementedError

    @property
    def weights(self):
        raise NotImplementedError

    @property
    def uv(self):
        raise NotImplementedError
