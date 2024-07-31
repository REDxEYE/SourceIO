from pathlib import Path

from ....utils.reporter import SourceIOException, Reporter
from .....blender_bindings.shared.exceptions import RAISE_EXCEPTIONS_ANYWAYS
from .....library.utils.path_utilities import backwalk_file_resolver
from ..content_provider_base import ContentProviderBase
from ..gma_provider import GMAContentProvider
from ..non_source_sub_manager import NonSourceContentProvider
from .source1_common import Source1Common
from .....logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('GModDetector')


class GModDetector(Source1Common):
    @classmethod
    def scan(cls, path: Path) -> dict[str, ContentProviderBase]:
        gmod_root = None
        gmod_dir = backwalk_file_resolver(path, 'garrysmod')
        if gmod_dir is not None:
            gmod_root = gmod_dir.parent
        if gmod_root is None:
            return {}
        content_providers = {}
        cls.recursive_traversal(gmod_root, 'garrysmod', content_providers)
        cls.register_common(gmod_root, content_providers)
        if (gmod_dir / 'addons').exists():
            for addon in (gmod_dir / 'addons').iterdir():
                if addon.suffix == '.gma':
                    try:
                        provider = GMAContentProvider(addon, 4000)
                    except SourceIOException as e:
                        logger.exception("Failed to open gma due to exception", e)
                        Reporter().error(e)
                        if RAISE_EXCEPTIONS_ANYWAYS:
                            raise e
                        continue
                    content_providers[addon.name] = provider
                elif addon.is_dir():
                    content_providers[addon.stem] = NonSourceContentProvider(addon, 4000)

        return content_providers

    @classmethod
    def register_common(cls, root_path: Path, content_providers: dict[str, ContentProviderBase]):
        super().register_common(root_path, content_providers)
