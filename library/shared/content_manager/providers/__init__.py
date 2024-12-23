from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.utils import TinyPath

ALL_PROVIDERS = {}


def check_provider_exists(provider_filepath: TinyPath) -> ContentProvider | None:
    return ALL_PROVIDERS.get(provider_filepath, None)


def register_provider(provider: ContentProvider):
    if provider.filepath in ALL_PROVIDERS:
        return ALL_PROVIDERS[provider.filepath]
    ALL_PROVIDERS[provider.filepath] = provider
    return provider
