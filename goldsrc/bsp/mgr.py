from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union

from ..wad import WadFile, WadEntry
from ...bpy_utilities.logging import BPYLoggingManager
from ...utilities.singleton import SingletonMeta

log_manager = BPYLoggingManager()


class GoldSrcContentManager(metaclass=SingletonMeta):
    def __init__(self):
        self.use_hd = False
        self.game_root: Path
        self.game_root_mod: Path
        self.game_resource_cache: Dict[Path, WadFile] = {}
        self.game_resource_roots: List[Path] = []
        self.game_root: Path = Path('')
        self.game_root_mod: Path = Path('')
        self.logger = log_manager.get_logger(self.__class__.__name__)

    def scan_for_content(self, path: Path):
        tmp_path = path
        while True:
            if Path.exists(tmp_path / 'liblist.gam'):
                self.game_root: Path = tmp_path.parent
                self.game_root_mod = tmp_path
                self.logger.info(f'Found game root: {self.game_root} ({self.game_root_mod})')
                break
            elif len(tmp_path.parts) == 1:
                self.game_root: Path = Path(tmp_path.parent.parent)
                self.game_root_mod = Path(tmp_path.parent)
                break
            found = False
            for default_resource in ('decals.wad', 'halflife.wad', 'liquids.wad', 'xeno.wad'):

                if (tmp_path / default_resource).exists():
                    self.game_root: Path = Path(tmp_path).parent
                    self.game_root_mod = Path(tmp_path)
                    found = True
                    break
            if found:
                break
            tmp_path = tmp_path.parent

        if self.game_root is None:
            self.logger.warn('Cannot find game directory path')

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

        self.logger.error(f'Cannot find file {name}')
        return None

    def add_game_resource_root(self, path: Path):
        if path not in self.game_resource_roots:
            resource_path = self.game_root / path
            try:
                if not Path.exists(resource_path):
                    resource_path = self.game_root_mod / path
                if not Path.exists(resource_path):
                    resource_path = self.game_root / 'valve' / path
                if not Path.exists(resource_path):
                    self.logger.warn(f'Invalid resource root path: {resource_path}')
                    return
            except OSError as e:
                # May be security-related or invalid path-related error, log and ignore
                self.logger.warn(f'Cannot access resource {resource_path}: {e}')
                return
            self.game_resource_roots.append(resource_path)
            self.logger.info(f'Added resource root: {path}')
