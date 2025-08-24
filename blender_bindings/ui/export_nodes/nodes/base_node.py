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
        self._refresh_internal_links()

    def _sync_io_visibility(self):
        return

    def _refresh_internal_links(self):
        links = self.internal_links
        print(dir(self))
        for link in links:
            links.remove(link)
        if "texture" in self.inputs and "texture" in self.outputs:
            links.new(self.inputs["texture"], self.outputs["texture"])
        if "channel" in self.inputs and "channel" in self.outputs:
            links.new(self.inputs["channel"], self.outputs["channel"])
