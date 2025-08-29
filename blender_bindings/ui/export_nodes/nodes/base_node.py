from typing import TextIO
from bpy.types import Node

class SourceIOModelTreeNode:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'SourceIOModelDefinition'

    def get_value(self):
        return None

    def update(self):
        return

    def write(self, buffer: TextIO):
        raise NotImplementedError

    def process(self, inputs: dict) -> dict|None:
        raise NotImplementedError(f"process method not implemented for {self.__class__.__name__}")


class SourceIOTextureTreeNode(Node, SourceIOModelTreeNode):

    def update(self):
        self._sync_io_visibility()

    def _sync_io_visibility(self):
        return