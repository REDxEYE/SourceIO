from collections import defaultdict
from typing import Optional

import bpy
import numpy as np
from mathutils import Matrix

from SourceIO.blender_bindings.material_loader.shaders.idtech3.idtech3 import IdTech3Shader
from SourceIO.blender_bindings.models.model_tags import register_model_importer
from SourceIO.blender_bindings.operators.import_settings_base import ModelOptions
from SourceIO.blender_bindings.shared.model_container import ModelContainer
from SourceIO.blender_bindings.utils.bpy_utils import get_or_create_material, add_material
from SourceIO.library.utils.tiny_path import TinyPath
from SourceIO.library.models.md3 import read_md3_model
from SourceIO.library.shared.content_manager import ContentManager
from SourceIO.library.utils import Buffer
from SourceIO.library.utils.idtech3_shader_parser import parse_shader_materials
from SourceIO.logger import SourceLogMan

log_manager = SourceLogMan()
logger = log_manager.get_logger('IDTech3::ModelLoader')


@register_model_importer(b"IDP3", 15)
def import_md3_15(model_path: TinyPath, buffer: Buffer,
                  content_manager: ContentManager, options: ModelOptions) -> Optional[ModelContainer]:
    model = read_md3_model(buffer)
    model_name = model_path.stem

    objects = []

    material_definitions = {}
    for _, buffer in content_manager.glob("*.shader"):
        materials = parse_shader_materials(buffer.read(-1).decode("utf-8"))
        material_definitions.update(materials)

    for surface in model.surfaces:
        if surface.frames.size == 0:
            continue

        model_mesh = bpy.data.meshes.new(f'{model_name}_mesh')
        model_object = bpy.data.objects.new(f'{model_name}', model_mesh)

        objects.append(model_object)

        model_mesh.from_pydata(surface.positions() * options.scale, [], surface.indices)
        model_mesh.update()

        vertex_indices = np.zeros((len(model_mesh.loops, )), dtype=np.uint32)
        model_mesh.loops.foreach_get('vertex_index', vertex_indices)

        uv_data = model_mesh.uv_layers.new()
        uvs = surface.uv.copy()
        uvs[:, 1] = 1 - uvs[:, 1]
        uv_data.data.foreach_set('uv', uvs[vertex_indices].flatten())

        model_mesh.normals_split_custom_set_from_vertices(surface.normals())
        model_object.shape_key_add(name='base')
        for i, frame in enumerate(surface.frames[1:]):
            shape_key = model_object.shape_key_add(name=f"frame_{i}")

            shape_key.data.foreach_set("co", surface.positions(i + 1).ravel() * options.scale)
        model_mesh.validate()
        material_name = surface.shaders[0].name
        mat = get_or_create_material(TinyPath(material_name).stem, material_name)
        add_material(mat, model_object)

        if mat.get('source1_loaded'):
            logger.debug(
                f'Skipping loading of {material_name} as it already loaded')
            continue
        logger.info(f"Loading {material_name} material")

        if material_name in material_definitions:
            loader = IdTech3Shader(content_manager)
            loader.create_nodes(mat, material_definitions[material_name])
        else:
            logger.error(f'Failed to find {material_name} texture')

    attachments = []

    for tag in model.tags:
        empty = bpy.data.objects.new(tag.name, None)
        mat = Matrix.Translation(tag.origin) @ Matrix(tag.axis).to_4x4()

        empty.matrix_basis.identity()
        empty.matrix_local = mat
        attachments.append(empty)

    return ModelContainer(objects, defaultdict(list), [], attachments)
