from typing import Type

from SourceIO.library.shared.content_manager.provider import ContentProvider
from SourceIO.library.shared.content_manager.providers import register_provider
from SourceIO.library.utils import TinyPath


class ContentDetector:

    @classmethod
    def scan(cls, path: TinyPath) -> list[ContentProvider]:
        raise NotImplementedError("Implement me")

    @staticmethod
    def add_provider(provider: ContentProvider, content_providers: dict[str, ContentProvider]):
        if provider.unique_name not in content_providers:
            content_providers[provider.unique_name] = register_provider(provider)

    @classmethod
    def add_if_exists(cls, path: TinyPath,
                      content_provider_class: Type[ContentProvider],
                      content_providers: dict[str, ContentProvider]):
        if path.exists():
            cls.add_provider(content_provider_class(path), content_providers)
