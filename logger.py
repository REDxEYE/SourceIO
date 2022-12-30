from .library import loaded_as_addon, running_in_blender

if loaded_as_addon() and running_in_blender():
    from .blender_bindings.utils.logging_impl import BPYLogger as _BPYLogger
    from .blender_bindings.utils.logging_impl import \
        BPYLoggingManager as _BPYLoggingManager
else:
    from .library.utils.logging_stub import BPYLogger as _BPYLogger
    from .library.utils.logging_stub import \
        BPYLoggingManager as _BPYLoggingManager


class SLogger(_BPYLogger):
    pass


class SLoggingManager(_BPYLoggingManager):
    pass
