from abc import ABCMeta
from pathlib import Path
from typing import Dict

from ..content_provider_base import ContentDetectorBase, ContentProviderBase

from ..source2_content_provider import Gameinfo2ContentProvider
from ..non_source_sub_manager import NonSourceContentProvider
from ..vpk_provider import VPKContentProvider


class Source2DetectorBase(ContentDetectorBase, metaclass=ABCMeta):

    @classmethod
    def scan_for_vpk(cls, root_dir: Path, content_providers: Dict[str, ContentProviderBase]):
        for vpk in root_dir.glob('*_dir.vpk'):
            content_providers[f'{root_dir.stem}_{vpk.stem}'] = VPKContentProvider(vpk)

    @classmethod
    def recursive_traversal(cls, game_root: Path, name: str, content_providers: Dict[str, ContentProviderBase]):
        if name in content_providers:
            return
        elif not (game_root / name / 'gameinfo.gi').exists():
            content_providers[name] = NonSourceContentProvider(game_root / name)
            cls.scan_for_vpk(game_root / name, content_providers)
        else:
            gh_provider = Gameinfo2ContentProvider(game_root / name / 'gameinfo.gi')
            content_providers[name] = gh_provider
            cls.scan_for_vpk(game_root / name, content_providers)
            for game in gh_provider.get_paths():
                game = Path(game)
                if game.name.startswith('|'):
                    continue
                elif game.name.endswith('*'):
                    if not (game_root / game).parent.exists():
                        continue
                    for folder in (game_root / game).parent.iterdir():
                        content_providers[folder.stem] = NonSourceContentProvider(folder)
                elif game.name.endswith('.vpk'):
                    game = game.with_name(game.stem + '_dir.vpk')
                    if (game_root / game).exists():
                        content_providers[Path(game).stem] = VPKContentProvider(game_root / Path(game))
                else:
                    cls.recursive_traversal(game_root, game.stem, content_providers)
