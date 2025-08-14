from __future__ import annotations

from typing import Any

"""
SourceIO helper library
"""
class VPKFile:
    def __init__(self: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Initialize self.  See help(type(self)) for accurate signature.
        """
        ...

    def check(self: Any, name: Any) -> Any:
        """
        Check if a file exists in the VPK by its name. Returns True if found, False otherwise.
        """
        ...

    def find_file(self: Any, name: Any) -> Any:
        """
        Find a file in the VPK by its name. Returns a tuple (offset, size, archive_id) if found, or None if not found.
        """
        ...

    def glob(self: Any, pattern: Any) -> Any:
        """
        Find files in the VPK matching a glob pattern.
        """
        ...

