import platform

platform_name = platform.system()


class UnsupportedSystem(Exception):
    pass


pylib = None
if platform_name == "Windows":
    from .win import pylib
elif platform_name == 'Linux':
    from .unix import pylib
elif platform_name == 'Darwin':
    try:
        from macos_m1 import pylib
    except ImportError:
        from macos_x86 import pylib
else:
    raise UnsupportedSystem(f'System {platform_name} not suppported')
