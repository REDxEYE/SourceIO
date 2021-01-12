import os

NO_BPY = int(os.environ.get('NO_BPY', '0'))

if not NO_BPY:
    from .logging_impl import BPYLogger as _BPYLogger
    from .logging_impl import BPYLoggingManager as _BPYLoggingManager
else:
    from .logging_stub import BPYLoggingManager as _BPYLoggingManager
    from .logging_stub import BPYLogger as _BPYLogger


class BPYLogger(_BPYLogger):
    pass


class BPYLoggingManager(_BPYLoggingManager):
    pass
