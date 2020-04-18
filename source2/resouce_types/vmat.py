from pathlib import Path

from SourceIO.source2.source2 import ValveFile


class Vmat:

    def __init__(self, vtex_path):
        self.valve_file = ValveFile(vtex_path)
        self.valve_file.read_block_info()

    def load(self):
        name = Path(self.valve_file.filename).stem
