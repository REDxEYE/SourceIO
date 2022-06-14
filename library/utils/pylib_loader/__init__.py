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
        print('M1 binary imported')
    except ImportError:
        from macos_x86 import pylib
        print('X86 binary imported')
else:
    raise UnsupportedSystem(f'System {platform_name} not suppported')
