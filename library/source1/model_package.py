"""Copy a compiled Source 1 model and its material assets into a game folder layout."""

from pathlib import Path, PurePosixPath
from shutil import copy2

from SourceIO.library.models.mdl.v49 import MdlV49
from SourceIO.library.shared.content_manager.providers.loose_files import LooseFilesContentProvider
from SourceIO.library.shared.content_manager.providers.source1_gameinfo_provider import Source1GameInfoProvider
from SourceIO.library.utils import FileBuffer, TinyPath, kv1
from SourceIO.library.utils.path_utilities import find_vtx


def _asset_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace('\\', '/'))
    if not path.parts or path.is_absolute() or '..' in path.parts or ':' in name or '\0' in name:
        raise ValueError(f'Invalid game-relative asset path: {name!r}')
    return path


def _material_values(block):
    for key, value in block.items():
        if isinstance(value, kv1.KV1Block):
            yield from _material_values(value)
        else:
            yield key.casefold(), value


def package_model(game_root: Path, model_path: str, destination: Path) -> tuple[int, tuple[str, ...]]:
    """Copy an existing MDL, sidecars and referenced VMT/VTF files; report missing materials."""
    model = _asset_path(model_path)
    if model.parts[0] != 'models' or model.suffix != '.mdl':
        raise ValueError('Model path must be game-relative models/.../*.mdl')
    mdl_file = game_root / model
    if not mdl_file.is_file():
        raise FileNotFoundError(f'Compiled model not found: {mdl_file}. Compile it first.')
    vvd = mdl_file.with_suffix('.vvd')
    vtx = find_vtx(TinyPath(mdl_file))
    if not vvd.is_file() or vtx is None:
        missing = '.vvd' if not vvd.is_file() else '.vtx'
        raise FileNotFoundError(f'Missing {missing} for {mdl_file}. Compile it first.')
    with FileBuffer(mdl_file) as buffer:
        mdl = MdlV49.from_buffer(buffer)

    destination = destination.resolve()
    written = set()
    missing = set()

    def target(path):
        out = destination / path
        if not out.resolve().is_relative_to(destination):
            raise ValueError(f'Package path escapes the destination: {out}')
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    for source in (mdl_file, vvd, Path(vtx), mdl_file.with_suffix('.phy'), mdl_file.with_suffix('.ani')):
        if source.is_file():
            path = model.with_name(source.name)
            out = target(path)
            if not out.is_file() or not source.samefile(out):
                copy2(source, out)
            written.add(path)

    loose = LooseFilesContentProvider(TinyPath(game_root))
    gameinfo = TinyPath(game_root / 'gameinfo.txt')
    mounted = Source1GameInfoProvider(gameinfo) if gameinfo.is_file() else None
    materials = PurePosixPath('materials')

    def copy_asset(path):
        buffer = loose.find_file(TinyPath(path))
        if buffer is None and mounted is not None:
            buffer = mounted.find_file(TinyPath(path))
        if buffer is None:
            return None
        with buffer:
            data = buffer.read()
        target(path).write_bytes(data)
        written.add(path)
        return data

    def pack_material(path):
        if path in written:
            return True
        data = copy_asset(path)
        if data is None:
            return False
        shader, values = kv1.loads(kv1.decode_text(data, str(path)), str(path)).top()
        if not isinstance(values, kv1.KV1Block):
            return True
        for key, value in _material_values(values):
            if not value or value.startswith(('[', '{')):
                continue
            if key == 'include' and shader.casefold() == 'patch':
                include = _asset_path(value)
                if include.parts[0].casefold() != 'materials':
                    include = materials / include
                if not include.suffix:
                    include = include.with_suffix('.vmt')
                if not pack_material(include):
                    missing.add(str(include))
                continue
            if key == 'include' or not key.startswith('$'):
                continue
            name = value.replace('\\', '/')
            if name.startswith(('/', '$', '!')) or ':' in name or '..' in name.split('/'):
                continue
            if not name.casefold().endswith('.vtf'):
                name += '.vtf'
            texture = materials / _asset_path(name)
            if texture not in written and copy_asset(texture) is None and '/' in name:
                missing.add(str(texture))
        return True

    search_paths = tuple(_asset_path(path.rstrip('/\\')) for path in mdl.materials_paths if path.strip('/\\'))
    for material in mdl.materials:
        name = _asset_path(material.name + '.vmt')
        if not any(pack_material(materials / folder / name) for folder in (*search_paths, PurePosixPath())):
            missing.add(str(materials / name))
    return len(written), tuple(sorted(missing))
