__all__ = [
    "VPKFile",
    "compression", "image", "mesh", "vtf",
]

import importlib
import sys


def _plat_pkg() -> str:
    sysplat = sys.platform  # 'win32', 'linux', 'darwin'
    if sysplat == "win32":
        return ".windows.pylib"
    if sysplat.startswith("linux"):
        return ".linux.pylib"
    if sysplat == "darwin":
        return ".macos.pylib"
    raise ImportError(f"{__name__}: unsupported platform {sysplat}")

def _load_native_and_alias():
    native_mod_name = _plat_pkg()
    native = importlib.import_module(native_mod_name, __package__)

    # Re-export top-level API (add/remove names as you like)
    for name in ("VPKFile",):
        if hasattr(native, name):
            globals()[name] = getattr(native, name)

    # Alias native submodules under our *current* package path
    # so 'from my_addon.sub1.sub2.pylib.compression import ...' works.
    for sub in ("compression", "image", "mesh", "vtf"):
        submod = getattr(native, sub, None)
        if submod is None:
            continue
        setattr(sys.modules[__name__], sub, submod)
        sys.modules[f"{__name__}.{sub}"] = submod

_load_native_and_alias()