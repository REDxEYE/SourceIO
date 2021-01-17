from ..utilities.path_utilities import backwalk_file_resolver
from ..source_shared.sub_manager import SubManager


class NonSourceSubManager(SubManager):
    def find_file(self, filepath: str):
        file = backwalk_file_resolver(self.filepath, filepath)
        if file:
            return file.open('rb')
