from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

from ...utilities.singleton import SingletonMeta
from ..wad import WadFile, WadEntry


class GoldSrcContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.use_hd = False
        self.game_root: Path
        self.game_root_mod: Path
        self.game_resource_cache: Dict[Path, WadFile] = {}
        self.game_resource_roots: List[Path] = []
        self.game_root: Path = Path('')
        self.game_root_mod: Path = Path('')

    def scan_for_content(self, path: Path):
        while True:
            if Path.exists(path / 'liblist.gam'):
                self.game_root: Path = path.parent
                self.game_root_mod = path
                print(f'Found game root: {self.game_root} ({self.game_root_mod})')
                break
            elif len(path.parts) == 1:
                self.game_root: Path = Path(path.parent.parent)
                self.game_root_mod = Path(path.parent)
                break
            path = path.parent

        if self.game_root is None:
            print('Cannot find game directory path')

        for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):
            self.add_game_resource_root(self.game_root / 'valve' / default_resource)

    def get_game_file(self, path: Path):
        return self.game_root / path

    def get_game_resource(self, name: str, path: Path = None) -> Optional[Union[WadEntry, Path]]:
        if path is not None:
            # print(f'Searching for game resource {name} in path {path}')
            if path.is_file() and path.suffix == '.wad':
                if path not in self.game_resource_cache:
                    self.game_resource_cache[path] = WadFile(path)
                return self.game_resource_cache[path].get_file(name)
            elif path.is_dir():
                resource_path = path / name
                if resource_path.exists():
                    return resource_path
                else:
                    return None
        local_storage = []
        if self.use_hd:
            local_storage.append(self.game_root_mod.with_name(f'{self.game_root_mod.name}_hd'))
        local_storage.append(self.game_root_mod)
        for root in chain(self.game_resource_roots, local_storage):
            resource = self.get_game_resource(name, root)
            if resource is not None:
                return resource

        print(f'Cannot find file {name}')

        return None

    def add_game_resource_root(self, path: Path):
        if path not in self.game_resource_roots:
            resource_path = self.game_root / path
            if not Path.exists(resource_path):
                resource_path = self.game_root_mod / path
            if not Path.exists(resource_path):
                resource_path = self.game_root / 'valve' / path
            if not Path.exists(resource_path):
                print(f'Invalid resource root path: {resource_path}')
                return
            self.game_resource_roots.append(resource_path)
            print(f'Added resource root: {path}')


def main():
    manager = GoldSrcContentManager(Path(r'E:\GoldSRC\Half-Life\valve\maps\c0a0.bsp'))
    print(manager)


if __name__ == '__main__':
    main()
