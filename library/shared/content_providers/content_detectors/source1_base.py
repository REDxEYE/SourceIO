import traceback
from abc import ABCMeta
from pathlib import Path

from .....logger import SourceLogMan
from ..content_provider_base import ContentDetectorBase, ContentProviderBase
from ..non_source_sub_manager import NonSourceContentProvider
from ..source1_content_provider import GameinfoContentProvider
from ..vpk_provider import VPKContentProvider

log_manager = SourceLogMan()
logger = log_manager.get_logger('Source1DetectorBase')


class Source1DetectorBase(ContentDetectorBase, metaclass=ABCMeta):

    @classmethod
    def scan_for_vpk(cls, root_dir: Path, content_providers: dict[str, ContentProviderBase]):
        for vpk in root_dir.glob('*_dir.vpk'):
            try:
                content_providers[f'{root_dir.stem}_{vpk.stem}'] = VPKContentProvider(vpk)
            except IOError as ex:
                print(f'Failed to load "{vpk}" VPK due to {ex}.')
                traceback.print_exc()
                print(f'Skipping {vpk}.')

    @classmethod
    def recursive_traversal(cls, game_root: Path, name: str, content_providers: dict[str, ContentProviderBase]):
        if name in content_providers or not (game_root / name / 'gameinfo.txt').exists():
            return
        try:
            gh_provider = GameinfoContentProvider(game_root / name / 'gameinfo.txt')
            cls.add_provider(name, gh_provider, content_providers)
            for game in gh_provider.gameinfo.all_paths:
                if game.name.startswith('|'):
                    logger.warn(f"Encountered game path: with \"|\" in name: {game.as_posix()!r}")
                    continue
                elif game.is_absolute() and game.is_dir():
                    cls.add_provider(game.stem, NonSourceContentProvider(game), content_providers)
                elif game.name.endswith('.vpk'):
                    if "_dir" not in game.stem:
                        game = game.with_name(game.stem + '_dir.vpk')
                    if game.is_absolute() and game.exists():
                        content_providers[game.stem] = VPKContentProvider(game_root / game)
                    elif (game_root / game).exists():
                        content_providers[game.stem] = VPKContentProvider(game_root / game)
                else:
                    cls.recursive_traversal(game_root, game.stem, content_providers)
        except ValueError as ex:
            logger.exception(f"Failed to parse gameinfo for {game_root / name}", ex)

        cls.scan_for_vpk(game_root / name, content_providers)
