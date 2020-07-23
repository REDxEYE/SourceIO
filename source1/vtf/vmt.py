import os

from ...utilities.path_utilities import NonSourceInstall
from ...utilities.valve_utils import GameInfoFile, KeyValueFile

from pathlib import Path

if os.environ.get('VProject', None):
    del os.environ['VProject']


class VMT:

    def _get_proj_root(self, path: Path):
        if len(path.parts) < 2:
            return path.parent
        if path.parts[-1] == 'materials':
            return path.parent
        else:
            return self._get_proj_root(path.parent)

    def __init__(self, filepath, game_dir=None):
        self.filepath = Path(filepath)
        if not game_dir:
            game_dir = self._get_proj_root(self.filepath)
        os.environ['VProject'] = str(game_dir)
        self.textures = {}
        self.kv = KeyValueFile(filepath=filepath)
        self.shader = self.kv.root_chunk.key
        self.material_data = self.kv.as_dict[self.shader]
        gameinfo_path = Path(game_dir, 'gameinfo.txt')
        if gameinfo_path.is_file():
            self.gameinfo = GameInfoFile(gameinfo_path)
        else:
            self.gameinfo = NonSourceInstall(self.filepath.parent)

    def parse(self):
        for key, value in self.material_data.items():
            if isinstance(value, str):
                texture = self.gameinfo.find_texture(value)
                if texture:
                    self.textures[key] = texture
                    # print(texture)
            # if textureAsGameTexture(value):
            #     self.textures[key[1:].lower()] = textureAsGameTexture(value)
