from SourceIO import running_in_blender, loaded_as_addon
from SourceIO.library.utils.reporter import SourceIOException, SourceIOWarning

RAISE_EXCEPTIONS_ANYWAYS = not running_in_blender() or not loaded_as_addon()


class SourceIOMissingFileException(SourceIOException):
    pass


class SourceIOUnsupportedFormatException(SourceIOException):
    pass


class SourceIOWrongMagic(SourceIOException):
    pass


class SourceIOFileNotFoundWarning(SourceIOWarning):
    pass


class SourceIOModelDidNotLoadWarning(SourceIOWarning):
    pass

