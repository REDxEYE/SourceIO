import traceback
from abc import ABCMeta
from pathlib import Path
from typing import Dict

from .....logger import SLoggingManager
from ..content_provider_base import ContentDetectorBase, ContentProviderBase

from ..source1_content_provider import GameinfoContentProvider
from ..non_source_sub_manager import NonSourceContentProvider
from ..vpk_provider import VPKContentProvider
from ...vpk.vpk_file import InvalidMagic

log_manager = SLoggingManager()
logger = log_manager.get_logger('Source1DetectorBase')


class Source1DetectorBase(ContentDetectorBase, metaclass=ABCMeta):

    @classmethod
    def scan_for_vpk(cls, root_dir: Path, content_providers: Dict[str, ContentProviderBase]):
        for vpk in root_dir.glob('*_dir.vpk'):
            try:
                content_providers[f'{root_dir.stem}_{vpk.stem}'] = VPKContentProvider(vpk)
            except InvalidMagic as ex:
                print(f'Failed to load "{vpk}" VPK due to {ex}.')
                traceback.print_exc()
                print(f'Skipping {vpk}.')

    @classmethod
    def recursive_traversal(cls, game_root: Path, name: str, content_providers: Dict[str, ContentProviderBase]):
        if name in content_providers or not (game_root / name / 'gameinfo.txt').exists():
            return
        gh_provider = GameinfoContentProvider(game_root / name / 'gameinfo.txt')
        content_providers[name] = gh_provider
        cls.scan_for_vpk(game_root / name, content_providers)
        for game in gh_provider.gameinfo.all_paths:
            if game.name.startswith('|'):
                logger.warn(f"Encountered game path: with \"|\" in name: {game.as_posix()!r}")
                continue
            elif game.is_absolute() and game.is_dir():
                content_providers[game.stem] = NonSourceContentProvider(game)
            elif game.name.endswith('.vpk'):
                game = game.with_name(game.stem + '_dir.vpk')
                if (game_root / game).exists():
                    content_providers[Path(game).stem] = VPKContentProvider(game_root / Path(game))
            else:
                cls.recursive_traversal(game_root, game.stem, content_providers)
