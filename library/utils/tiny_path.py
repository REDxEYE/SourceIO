import io
import os
import platform
from os import PathLike
from pathlib import Path
from typing import Union

os_sep = os.path.sep

PathTypes = Union[Path, str, PathLike, 'TinyPath']

sep = '/'
is_windows = platform.system() == "Windows"


class TinyPath(str, PathLike):

    def __fspath__(self):
        return Path(self)

    def __new__(cls, value: str):
        value = str(value)
        if "\\" in value:
            value = value.replace("\\", "/")
        return super().__new__(cls, value.rstrip("/"))

    @property
    def stem(self):
        if "/" in self:
            sep_index = self.rindex("/") + 1
            name = self[sep_index:]
        else:
            name = self
        if "." in name:
            return name[:name.rindex(".")]
        return name

    @property
    def name(self):
        if "/" in self:
            sep_index = self.rindex("/") + 1
            return self[sep_index:]
        return self

    @property
    def parent(self):
        if "/" in self:
            sep_index = self.rindex("/")
            return TinyPath(self[:sep_index])
        return self

    @property
    def parts(self):
        return self.split("/")

    @property
    def suffix(self):
        if "." in self:
            return self[self.rindex("."):]
        return ""

    @property
    def root(self):
        return TinyPath(self[:self.index("/")])

    def is_absolute(self):
        if is_windows:
            return ":" in self
        return os.path.isabs(self)

    def is_relative_to(self, other: PathTypes) -> bool:
        if os.name == "nt":
            return self.lower().startswith(str(other).lower())
        return self.startswith(str(other))

    def relative_to(self, other: PathTypes) -> 'TinyPath':
        if self.is_relative_to(other):
            return TinyPath(self[len(other) + 1:])
        raise ValueError(f"{self!r} is not in the subpath of {other!r}"
                         " OR one path is relative and the other is absolute.")

    def absolute(self):
        if self.is_absolute():
            return self
        return os.getcwd() / self

    def resolve(self):
        return TinyPath(Path(self).resolve().as_posix())

    def glob(self, pattern: str):
        for item in Path(self).glob(pattern):
            yield item

    def rglob(self, pattern: str):
        for item in Path(self).rglob(pattern):
            yield item

    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        if "b" not in mode:
            encoding = io.text_encoding(encoding)
        return open(self, mode, buffering, encoding, errors, newline)

    def as_posix(self):
        return str(self)

    def exists(self):
        return os.path.exists(self)

    def is_file(self):
        return os.path.isfile(self)

    def is_dir(self):
        return os.path.isdir(self)

    def with_suffix(self, suffix: str):
        if self.suffix:
            return TinyPath(self[:-len(self.suffix)] + suffix)
        return TinyPath(self + suffix)

    def iterdir(self):
        if not self.exists():
            return []
        for item in Path(self).iterdir():
            yield TinyPath(item.as_posix())

    def with_name(self, name: str):
        suffix = self.suffix
        return (self.parent / name).with_suffix(suffix)

    def __repr__(self):
        return f"{self.__class__.__name__}(\"{self!s}\")"

    def __truediv__(self, other: PathTypes):
        if self.is_absolute() and TinyPath(other).is_absolute():
            raise ValueError(f"Cannot join absolute paths: {self!r}|{other!r}.")
        return TinyPath(self + "/" + str(other))

    def __rtruediv__(self, other: PathTypes):
        if self.is_absolute():
            raise ValueError("Cannot add relative path to absolute path.")
        return TinyPath(other) / self

    def __eq__(self, other: PathTypes):
        other = TinyPath(other)
        if os.name == "nt":
            return other.as_posix().lower() == self.as_posix().lower()
        return other.as_posix() == self.as_posix()

    def __hash__(self):
        return hash(str(self))


if __name__ == '__main__':
    tmp = TinyPath(r"C:/test/abc.dep")
    print(repr(tmp))
    print(tmp.name)
    print(tmp.stem)
    print(tmp.parent)
    print(repr(tmp.parent))

    print("Test/123/" / TinyPath("123123"))
    print(tmp.parent / "bcvd.res")

    print(tmp.suffix)
    print(tmp.root)

    print(tmp.is_relative_to(tmp.parent))
    print(tmp.relative_to(tmp.parent))
    print(tmp.is_absolute())

    print(TinyPath("./test").absolute())
    print(Path("./test").absolute())
    print(tmp.with_suffix(".txt"))
    print(Path(tmp).with_name("test"))
    print(tmp.with_name("test"))
