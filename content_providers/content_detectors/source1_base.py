from abc import ABCMeta
from pathlib import Path
from typing import Dict

from ..content_provider_base import ContentDetectorBase, ContentProviderBase

from ..source1_content_provider import GameinfoContentProvider
from ..non_source_sub_manager import NonSourceContentProvider
from ..vpk_sub_manager import VPKContentProvider


class Source1DetectorBase(ContentDetectorBase, metaclass=ABCMeta):

    @classmethod
    def scan_for_vpk(cls, root_dir: Path, content_providers: Dict[str, ContentProviderBase]):
        for vpk in root_dir.glob('*_dir.vpk'):
            content_providers[f'{root_dir.stem}_{vpk.stem}'] = VPKContentProvider(vpk)

    @classmethod
    def recursive_traversal(cls, game_root: Path, name: str, content_providers: Dict[str, ContentProviderBase]):
        if name in content_providers or not (game_root / name / 'gameinfo.txt').exists():
            return
        gh_provider = GameinfoContentProvider(game_root / name / 'gameinfo.txt')
        content_providers[name] = gh_provider
        cls.scan_for_vpk(game_root/name, content_providers)
        for game in gh_provider.gameinfo.file_system.search_paths.all_paths:
            game = Path(game)
            if game.name.startswith('|'):
                continue
            elif game.name.endswith('*'):
                if not (game_root / game).parent.exists():
                    continue
                for folder in (game_root / game).parent.iterdir():
                    if folder.is_dir():
                        content_providers[folder.stem] = NonSourceContentProvider(folder)
            elif game.name.endswith('.vpk'):
                game = game.with_name(game.stem + '_dir.vpk')
                if (game_root / game).exists():
                    content_providers[Path(game).stem] = VPKContentProvider(game_root / Path(game))
            else:
                cls.recursive_traversal(game_root, game.stem, content_providers)
