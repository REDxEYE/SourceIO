from typing import TextIO


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
