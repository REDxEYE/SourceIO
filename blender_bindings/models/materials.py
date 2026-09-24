"""Material path resolution shared by Source 1 material and mesh import."""
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.shared.content_manager import ContentManager


def resolve_model_material(content_manager: ContentManager, mdl, material_name: str) -> TinyPath | None:
    """Resolve with existence checks; cache paths and misses for this import."""
    paths = getattr(mdl, '_resolved_material_paths', None)
    if paths is None:
        paths = mdl._resolved_material_paths = {}
    if material_name in paths:
        return paths[material_name]

    # The bare name may already contain its full path. Deduplicate candidates
    # because MDLs can also contain empty or repeated material search paths.
    candidates = [TinyPath(material_name)]
    for directory in mdl.materials_paths:
        if not directory:
            continue
        directory = TinyPath(directory)
        if not directory.is_absolute():
            candidates.append(directory / material_name)
    seen = set()
    for path in candidates:
        candidate = 'materials' / TinyPath(path.as_posix() + '.vmt')
        key = candidate.as_posix().casefold()
        if key in seen:
            continue
        seen.add(key)
        if content_manager.check(candidate):
            paths[material_name] = path
            return path
    paths[material_name] = None
    return None


def get_model_material_names(content_manager: ContentManager, mdl):
    """Reuse material import's resolution results without reopening any files."""
    names = {}
    for material in mdl.materials:
        paths = getattr(mdl, '_resolved_material_paths', {})
        if material.name not in paths:
            resolve_model_material(content_manager, mdl, material.name)
        path = mdl._resolved_material_paths[material.name]
        names[material.name] = path.as_posix().lstrip('/') if path is not None else material.name
    return names
