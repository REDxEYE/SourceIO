from bpy.types import Node


class SourceIOModelTreeNode:
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'sourceio.model_definition'


class SourceIOTextureTreeNode(Node, SourceIOModelTreeNode):

    def update(self):
        self._sync_io_visibility()

    def _sync_io_visibility(self):
        return
