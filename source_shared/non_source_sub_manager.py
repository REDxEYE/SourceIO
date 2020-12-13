from ..source1.sub_manager import SubManager
from ..utilities.path_utilities import backwalk_file_resolver


class NonSourceSubManager(SubManager):
    def find_file(self, filepath: str):
        file = backwalk_file_resolver(self.filepath, filepath)
        if file:
            return file.open('rb')
