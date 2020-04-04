import io
import os.path
import random
import time
from contextlib import redirect_stdout
from pathlib import Path

from ...utilities import progressbar, valve_utils
from ..mdl.vvd_readers.vvd_v4 import SourceVvdFile4
from ..mdl.vtx_readers.vtx_v7 import SourceVtxFile7
from ..mdl.source_model import SourceModel
from ..mdl.mdl_readers.mdl_v49 import SourceMdlFile49
from ..data_structures import source_shared, mdl_data, vtx_data
from ..vtf.vmt import VMT
from ..vtf.blender_material import BlenderMaterial
from ...utilities.path_utilities import resolve_root_directory_from_file

# Blender imports
try:
    import bpy

    # bpy.app.debug = True
    import mathutils
    from mathutils import Vector, Matrix, Euler
except ImportError:
    raise Exception("Cannot be run without bpy (blender) module")

stdout = io.StringIO()


def split(array, n=3):
    return [array[i:i + n] for i in range(0, len(array), n)]


class Source2Blender:
    def __init__(self, path: str = None, import_textures=False, work_directory=None, custom_name=None,
                 normal_bones=False, join_clamped=False, context=None):
        self.import_textures = import_textures
        self.filepath = Path(path)
        if work_directory:
            self.work_directory = work_directory
        else:
            self.work_directory = resolve_root_directory_from_file(path)
        if self.work_directory is None:
            try:
                preferences = context.preferences
                addon_prefs = preferences.addons['SourceIO.prefs'].preferences
                self.work_directory = addon_prefs.sfm_path
            except:
                self.work_directory = ''
        else:
            self.work_directory = ''
        self.main_collection = None

        self.current_collection = None
        self.join_clamped = join_clamped
        self.normal_bones = normal_bones
        self.custom_name = custom_name

        self.name = self.filepath.stem
        self.vertex_offset = 0
        self.sort_bodygroups = True

        self.model = SourceModel(self.filepath)
        self.mdl = None  # type: SourceMdlFile49
        self.vvd = None  # type: SourceVvdFile4
        self.vtx = None  # type: SourceVtxFile7

        self.mesh_obj = None
        self.armature_obj = None
        self.armature = None
        self.mesh_data = None

    def load(self, dont_build_mesh=False):

        self.model.read()
        self.mdl = self.model.mdl  # type: SourceMdlFile49
        self.vvd = self.model.vvd  # type: SourceVvdFile4
        self.vtx = self.model.vtx  # type: SourceVtxFile7

        if not dont_build_mesh:
            print("Building mesh")
            self.main_collection = bpy.data.collections.new(os.path.basename(self.mdl.file_data.name))
            bpy.context.scene.collection.children.link(self.main_collection)
            self.armature_obj = None
            self.armature = None
            self.create_skeleton(self.normal_bones)
            if self.custom_name:
                self.armature_obj.name = self.custom_name

            # just a temp containers
            self.mesh_obj = None
            self.mesh_data = None

            self.create_models()
            self.create_attachments()

            bpy.ops.object.mode_set(mode='OBJECT')

        if self.import_textures:
            self.load_textures()

    def create_skeleton(self, normal_bones=False):

        bpy.ops.object.armature_add(enter_editmode=True)

        self.armature_obj = bpy.context.object
        try:
            bpy.context.scene.collection.objects.unlink(self.armature_obj)
        except:
            pass
        self.main_collection.objects.link(self.armature_obj)
        self.armature_obj.name = self.name + '_ARM'

        self.armature = self.armature_obj.data
        self.armature.name = self.name + "_ARM_DATA"
        self.armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.armature_obj
        bpy.ops.object.mode_set(mode='EDIT')
        self.armature.edit_bones.remove(self.armature.edit_bones[0])

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action="DESELECT")

        bpy.ops.object.mode_set(mode='EDIT')
        bones = []
        for se_bone in self.mdl.file_data.bones:  # type: mdl_data.SourceMdlBone
            bones.append((self.armature.edit_bones.new(se_bone.name), se_bone))

        for bl_bone, se_bone in bones:  # type: bpy.types.EditBone, mdl_data.SourceMdlBone
            if se_bone.parent_bone_index != -1:
                bl_parent, parent = bones[se_bone.parent_bone_index]
                bl_bone.parent = bl_parent
            bl_bone.tail = Vector([0, 0, 1]) + bl_bone.head

        bpy.ops.object.mode_set(mode='POSE')
        for se_bone in self.mdl.file_data.bones:  # type: mdl_data.SourceMdlBone
            bl_bone = self.armature_obj.pose.bones.get(se_bone.name)
            pos = Vector(se_bone.position.as_list)
            rot = Euler(se_bone.rotation.as_list)
            mat = Matrix.Translation(pos) @ rot.to_matrix().to_4x4()
            bl_bone.matrix_basis.identity()

            if bl_bone.parent:
                bl_bone.matrix = bl_bone.parent.matrix @ mat
            else:
                bl_bone.matrix = mat

        bpy.ops.pose.armature_apply()
        bpy.ops.object.mode_set(mode='EDIT')
        if normal_bones:
            for name, bl_bone in self.armature.edit_bones.items():
                if not bl_bone.parent:
                    continue
                parent = bl_bone.parent
                if len(parent.children) > 1:
                    bl_bone.use_connect = False
                    parent.tail = sum([ch.head for ch in parent.children], mathutils.Vector()) / len(parent.children)
                else:
                    parent.tail = bl_bone.head
                    if bl_bone.parent:
                        bl_bone.use_connect = True
                    if bl_bone.children == 0:
                        par = bl_bone.parent
                        if par.children > 1:
                            bl_bone.tail = bl_bone.head + (par.tail - par.head)
                    if bl_bone.parent == 0 and bl_bone.children > 1:
                        bl_bone.tail = (bl_bone.head + bl_bone.tail) * 2
                if not bl_bone.children:
                    vec = bl_bone.parent.head - bl_bone.head
                    bl_bone.tail = bl_bone.head - vec / 2
                bone_size = bl_bone.tail - bl_bone.head
                if bone_size.x < 0.0001 or bone_size.y < 0.0001 or bone_size.z < 0.0001:
                    bl_bone.tail = bl_bone.head + Vector([1, 0, 0])
            bpy.ops.armature.calculate_roll(type='GLOBAL_POS_Z')
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def get_material(mat_name, model_ob):
        if mat_name:
            mat_name = mat_name
        else:
            mat_name = "Material"
        mat_ind = 0
        md = model_ob.data
        mat = None
        for candidate in bpy.data.materials:  # Do we have this material already?
            if candidate.name == mat_name:
                mat = candidate
        if mat:
            if md.materials.get(mat.name):  # Look for it on this mesh_data
                for i in range(len(md.materials)):
                    if md.materials[i].name == mat.name:
                        mat_ind = i
                        break
            else:  # material exists, but not on this mesh_data
                md.materials.append(mat)
                mat_ind = len(md.materials) - 1
        else:  # material does not exist
            mat = bpy.data.materials.new(mat_name)
            md.materials.append(mat)
            # Give it a random colour
            rand_col = []
            for i in range(3):
                rand_col.append(random.uniform(.4, 1))
            rand_col.append(1.0)
            mat.diffuse_color = rand_col

            mat_ind = len(md.materials) - 1

        return mat_ind

    def get_polygon(self, strip_group: vtx_data.SourceVtxStripGroup, vtx_index_index: int, lod_index,
                    mesh_vertex_offset):
        del lod_index
        vertex_indices = []
        vn_s = []
        offset = self.vertex_offset + mesh_vertex_offset
        for i in [0, 2, 1]:  # type: int
            vtx_vertex_index = strip_group.vtx_indexes[vtx_index_index + i]  # type: int
            vtx_vertex = strip_group.vtx_vertexes[vtx_vertex_index]
            vertex_index = vtx_vertex.original_mesh_vertex_index + offset
            if vertex_index > self.vvd.max_verts:
                print('vertex index out of bounds, skipping this mesh_data')
                return False, False
            try:
                vn = self.vvd.file_data.vertexes[vertex_index].normal.as_list
            except IndexError:
                vn = [0, 1, 0]
            vertex_indices.append(vertex_index)
            vn_s.append(vn)
        return vertex_indices, vn_s

    def convert_mesh(self, vtx_model: vtx_data.SourceVtxModel, lod_index, model: mdl_data.SourceMdlModel):
        vtx_meshes = vtx_model.vtx_model_lods[lod_index].vtx_meshes
        indexes = []
        vertex_normals = []
        # small speedup
        i_ex = indexes.append
        material_indexes = []
        m_ex = material_indexes.append
        vn_ex = vertex_normals.extend

        for mesh_index, vtx_mesh in enumerate(vtx_meshes):  # type: int,vtx_data.SourceVtxMesh
            material_index = model.meshes[mesh_index].material_index
            mesh_vertex_start = model.meshes[mesh_index].vertex_index_start
            if vtx_mesh.vtx_strip_groups:

                for group_index, strip_group in enumerate(
                        vtx_mesh.vtx_strip_groups):  # type: vtx_data.SourceVtxStripGroup

                    if strip_group.vtx_strips and strip_group.vtx_indexes and strip_group.vtx_vertexes:
                        field = progressbar.ProgressBar('Converting mesh', len(strip_group.vtx_indexes), 20)
                        for vtx_index in range(0, len(strip_group.vtx_indexes), 3):
                            if not vtx_index % 3 * 10:
                                field.increment(30)
                            f, vn = self.get_polygon(strip_group, vtx_index, lod_index, mesh_vertex_start)
                            if not f and not vn:
                                break
                            i_ex(f)
                            vn_ex(vn)
                            m_ex(material_index)
                        field.is_done = True
                        field.draw()
                    else:
                        pass

            else:
                pass
        return indexes, material_indexes, vertex_normals

    @staticmethod
    def convert_vertex(vertex: source_shared.SourceVertex):
        return vertex.position.as_list, (vertex.texCoordX, 1 - vertex.texCoordY)

    @staticmethod
    def remap_materials(used_materials, all_materials):
        remap = {}
        for n, used_material in enumerate(used_materials):
            remap[all_materials.index(used_material)] = n

        return remap

    def create_model(self, model: mdl_data.SourceMdlModel, vtx_model: vtx_data.SourceVtxModel):
        name = model.name.replace('.smd', '').replace('.dmx', '')
        if '/' in name or '\\' in name:
            name = os.path.basename(name)
        if len(vtx_model.vtx_model_lods[0].vtx_meshes) < 1:
            print('No meshes in vtx model')
            return
        self.mesh_obj = bpy.data.objects.new(name, bpy.data.meshes.new('{}_MESH'.format(name)))
        self.mesh_obj.parent = self.armature_obj

        self.current_collection.objects.link(self.mesh_obj)

        modifier = self.mesh_obj.modifiers.new(
            type="ARMATURE", name="Armature")
        modifier.object = self.armature_obj

        self.mesh_data = self.mesh_obj.data

        used_materials = {m_id: False for m_id in range(
            len(self.mdl.file_data.textures))}
        weight_groups = {bone.name: self.mesh_obj.vertex_groups.new(name=bone.name) for bone in
                         self.mdl.file_data.bones}
        vtx_model_lod = vtx_model.vtx_model_lods[0]  # type: vtx_data.SourceVtxModelLod
        print('Converting {} model'.format(name))
        if vtx_model_lod.meshCount > 0:
            t = time.time()
            polygons, polygon_material_indexes, normals = self.convert_mesh(vtx_model, 0, model)
            print('Mesh conversion took {} sec'.format(round(time.time() - t), 3))
        else:
            return
        self.vertex_offset += model.vertex_count

        for mat_index in set(polygon_material_indexes):
            used_materials[mat_index] = True

        used_materials_names = [self.mdl.file_data.textures[mat_id] for mat_id, used in used_materials.items() if used]
        mat_remap = self.remap_materials(used_materials_names, self.mdl.file_data.textures)
        mats = sorted(mat_remap.items(), key=lambda m: m[1])
        for old_mat_id, new_mat_id in mats:
            mat_name = self.mdl.file_data.textures[old_mat_id].path_file_name
            self.get_material(mat_name, self.mesh_obj)

        vertexes = []
        uvs = []
        for vertex in self.vvd.file_data.vertexes:
            vert_co, uv = self.convert_vertex(vertex)
            vertexes.append(vert_co)
            uvs.append(uv)

        self.mesh_data.from_pydata(vertexes, [], polygons)
        self.mesh_data.update()

        self.mesh_data.uv_layers.new()
        uv_data = self.mesh_data.uv_layers[0].data
        for i in range(len(uv_data)):
            u = uvs[self.mesh_data.loops[i].vertex_index]
            uv_data[i].uv = u

        for polygon, mat_index in zip(
                self.mesh_data.polygons, polygon_material_indexes):
            polygon.material_index = mat_remap[mat_index]

        if self.mdl.file_data.flex_descs:
            self.add_flexes(model)

        for n, vertex in enumerate(self.vvd.file_data.vertexes):
            for bone_index, weight in zip(vertex.boneWeight.bone, vertex.boneWeight.weight):
                if weight == 0.0:
                    continue
                weight_groups[self.mdl.file_data.bones[bone_index].name].add([n], weight, 'REPLACE')

        bpy.ops.object.select_all(action="DESELECT")
        self.mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = self.mesh_obj

        with redirect_stdout(stdout):
            bpy.ops.object.mode_set(mode='EDIT')
            self.mesh_data.validate()
            self.mesh_data.validate()
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.shade_smooth()
            self.mesh_data.normals_split_custom_set(normals)
            self.mesh_data.use_auto_smooth = True
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.delete_loose()
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        return self.mesh_obj

    def create_models(self):

        self.mdl.file_data = self.mdl.file_data  # type: mdl_data.SourceMdlFileData
        for bodyparts in self.mdl.file_data.bodypart_frames:
            to_join = []
            for bodypart_index, bodypart in bodyparts:
                if self.sort_bodygroups:
                    if bodypart.model_count > 1:
                        self.current_collection = bpy.data.collections.new(bodypart.name)
                        self.main_collection.children.link(self.current_collection)
                    else:
                        self.current_collection = self.main_collection
                else:
                    self.current_collection = self.main_collection
                for model_index, model in enumerate(bodypart.models):
                    if model.mesh_count == 0:
                        continue
                    vtx_model = self.vtx.vtx.vtx_body_parts[bodypart_index].vtx_models[model_index]
                    to_join.append(self.create_model(model, vtx_model))
            if self.join_clamped:
                for ob in to_join:
                    if not ob:
                        continue
                    if ob.type == 'MESH':
                        ob.select_set(True)
                        bpy.context.view_layer.objects.active = ob
                    else:
                        ob.select_set(False)
                        bpy.context.view_layer.objects.active = to_join[0]
                if len(bpy.context.selected_objects) < 2:
                    continue
                with redirect_stdout(stdout):
                    bpy.ops.object.join()
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.remove_doubles(threshold=0.00001)
                    bpy.ops.object.mode_set(mode='OBJECT')

    def add_flexes(self, mdl_model: mdl_data.SourceMdlModel):
        self.mesh_obj.shape_key_add(name='base')

        for flex_frame in mdl_model.flex_frames:
            for flex, vertex_offset in zip(
                    flex_frame.flexes, flex_frame.vertex_offsets):

                flex_desc = self.mdl.file_data.flex_descs[flex.flex_desc_index]
                flex_name = flex_desc.name

                if not self.mesh_obj.data.shape_keys.key_blocks.get(flex_name):
                    self.mesh_obj.shape_key_add(name=flex_name)

                for flex_vert in flex.the_vert_anims:  # type: mdl_data.SourceMdlVertAnim
                    vertex_index = flex_vert.index + vertex_offset
                    vertex = self.mesh_obj.data.vertices[vertex_index]
                    vx = vertex.co.x
                    vy = vertex.co.y
                    vz = vertex.co.z
                    fx, fy, fz = flex_vert.the_delta
                    self.mesh_obj.data.shape_keys.key_blocks[flex_name].data[vertex_index].co = (
                        fx + vx, fy + vy, fz + vz)

    def create_attachments(self):
        if self.mdl.file_data.attachments:

            attachment_collection = bpy.data.collections.new('attachments')

            for attachment in self.mdl.file_data.attachments:
                bone = self.armature.bones.get(
                    self.mdl.file_data.bones[attachment.localBoneIndex].name)

                empty = bpy.data.objects.new("empty", None)
                # bpy.context.scene.objects.link(empty)
                attachment_collection.objects.link(empty)
                empty.name = attachment.name
                pos = Vector(
                    [attachment.pos.x, attachment.pos.y, attachment.pos.z])
                rot = Euler([attachment.rot.x, attachment.rot.y, attachment.rot.z])
                empty.matrix_basis.identity()
                empty.parent = self.armature_obj
                empty.parent_type = 'BONE'
                empty.parent_bone = bone.name
                empty.location = pos
                empty.rotation_euler = rot

    def load_textures(self):
        mod_path = valve_utils.get_mod_path(self.filepath)
        game_info_path = mod_path / 'gameinfo.txt'
        if not game_info_path.exists():
            return
        else:
            print('Found gameinfo.txt file')
        gi = valve_utils.GameInfoFile(game_info_path)
        materials = []
        used_textures = []
        textures = []
        for texture in self.mdl.file_data.textures:
            for tex_path in self.mdl.file_data.texture_paths:
                if tex_path and (tex_path[0] == '/' or tex_path[0] == '\\'):
                    tex_path = tex_path[1:]
                if tex_path:
                    mat = gi.find_material(Path(tex_path) / texture.path_file_name, use_recursive=True)
                    if mat:
                        temp = valve_utils.get_mod_path(mat)
                        materials.append((Path(mat), Path(mat).relative_to(temp)))
        for mat in set(materials):
            bmat = BlenderMaterial(VMT(mat[0], mod_path))
            print(f"Importing {mat[0].stem}")
            bmat.load_textures()
            bmat.create_material(True)
            # kv = valve_utils.KeyValueFile(mat[0])
            # for v in list(kv.as_dict.values())[0].values():
            #     if '/' in v or '\\' in v:
            #         used_textures.append(Path(v))
            #         tex = gi.find_texture(v, True)
            #         if tex:
            #             temp = valve_utils.get_mod_path(tex)
            #             textures.append((Path(tex), Path(tex).relative_to(temp)))


if __name__ == '__main__':
    # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\tf_movies\models\player\hwm\spy'
    model_path = r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\Furry\models\red_eye\lewd_models\daxzor.mdl"
    # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\usermod\models\red_eye\tyranno\raptor.mdl'
    # model_path = r'./test_data/hl/box01a.mdl'
    # model_path = r'G:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\usermod\models\red_eye\rick-and-morty\pink_raptor.mdl'
    a = Source2Blender(model_path, normal_bones=True, join_clamped=False)
    a.load()
    a.create_models()
    # a = Source2Blender(r'test_data\titan_buddy.mdl', normal_bones=False)
    # a = IO_MDL(r'E:\PYTHON\MDL_reader\test_data\nick_hwm.mdl', normal_bones=True)
