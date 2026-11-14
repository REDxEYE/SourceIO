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
        Return True if the file `name` (case-insensitive) exists in the VPK, False otherwise.
        """
        ...

    def find_file(self: Any, name: Any) -> Any:
        """
        Return the contents of the file `name` (case-insensitive) as bytes, or None if
        the VPK has no such file. Raises OSError/ValueError if the data cannot be read.
        """
        ...

    def glob(self: Any, pattern: Any) -> Any:
        """
        Return an iterator of (name, data) tuples for files matching `pattern`.
        Matching follows fnmatch.fnmatchcase on the lowercased name and pattern:
        '*' matches any run of characters including '/', '?' one character,
        '[seq]'/'[!seq]' character classes.
        """
        ...

