import math
import os.path
from pathlib import Path

from ... import bl_info
from ..data_structures.mdl_data import SourceMdlBone
from ..data_structures.mdl_data import SourceMdlBodyPart, SourceMdlAttachment
from ..data_structures.source_shared import SourceVector
from ...utilities.math_utilities import convert_rotation_matrix_to_degrees, vector_i_transform
from ..mdl.mdl_readers.mdl_v49 import SourceMdlFile49, SourceMdlModel
from ..mdl.smd_generator import SMD
from ..mdl.vtx_readers.vtx_v7 import SourceVtxFile7
from ..mdl.vvd_readers.vvd_v4 import SourceVvdFile4


class QC:
    version = '.'.join(map(str,bl_info['version']))

    def __init__(self, source_model):
        self.source_model = source_model
        self.mdl = source_model.mdl  # type:SourceMdlFile49
        self.vvd = source_model.vvd  # type:SourceVvdFile4
        self.vtx = source_model.vtx  # type:SourceVtxFile7
        self.smd = None
        self.vta = None
        self.fst_model = None

    def write_qc(self, output_dir=os.path.dirname(__file__)):
        file_path = Path(output_dir)
        file_path.mkdir(parents=True, exist_ok=True)
        file_path = file_path / self.source_model.filepath.stem
        with file_path.with_suffix('.qc').open('w') as fileh:
            self.smd = SMD(self.source_model)
            self.write_header(fileh)
            self.write_models(fileh)
            self.write_skins(fileh)
            self.write_misc(fileh)
            self.write_jiggle_bones(fileh)
            self.write_sequences(fileh)

    def write_header(self, fileh):
        fileh.write('// Created by SourceIO v{}\n\n'.format(self.version))
        fileh.write('$modelname "{}"\n\n'.format(Path(self.mdl.file_data.name).parent/Path(self.mdl.file_data.name).stem))

    def write_models(self, fileh):
        for n, bp in enumerate(
                self.mdl.file_data.body_parts):  # type: SourceMdlBodyPart
            if bp.model_count > 1:
                self.write_bodygroup(fileh, bp)
            if bp.model_count == 1:
                self.write_model(fileh, bp, bp.models[0])
                pass  # write model
            if bp.model_count == 0:
                print('No models in bodygpoup!!!!')

    def write_model(self, fileh, bp, model: SourceMdlModel):
        name = model.name if (model.name and model.name != 'blank') else "mesh_{}-{}".format(
            bp.name, model.name)
        if not self.fst_model:
            self.fst_model = name
        model_name = str(Path(name).with_suffix('').with_suffix(''))
        if model.flex_frames or model.eyeball_count:
            fileh.write('$model "{0}" "{0}" '.format(model_name))
            fileh.write('{\n')
            fileh.write('\n')

            if model.eyeball_count:
                for n, eyeball in enumerate(model.eyeballs):
                    fileh.write('\t')
                    fileh.write('eyeball')
                    diameter = eyeball.radius * 2
                    angle = round(math.degrees(math.atan(eyeball.z_offset)), 6)

                    iris_scale = 1 / eyeball.iris_scale
                    if n == 0 and angle > 0:
                        fileh.write(' "eye_right"')
                    elif n == 1 and angle < 0:
                        fileh.write(' "eye_left"')
                    else:
                        fileh.write(' "eye_{}"'.format(n))
                    bone = self.mdl.file_data.bones[eyeball.bone_index]  # type:SourceMdlBone
                    fileh.write(' "{}"'.format(bone.name))
                    fileh.write(' {}'.format(vector_i_transform(eyeball.org,
                                                                bone.pose_to_bone_column0,
                                                                bone.pose_to_bone_column1,
                                                                bone.pose_to_bone_column2,
                                                                bone.pose_to_bone_column3,
                                                                ).as_rounded(4)))
                    fileh.write(' "{}"'.format(self.mdl.file_data.textures[eyeball.texture_index].path_file_name))
                    fileh.write(' {}'.format(diameter))
                    fileh.write(' {}'.format(angle))
                    fileh.write(' "iris_unused"')
                    fileh.write(' {}'.format(int(iris_scale)))
                    fileh.write('\n')
                fileh.write('\n')
            if model.flex_frames:
                fileh.write('\t//FLEXES:\n')
                for flex in model.flex_frames:
                    fileh.write('\t\t//{} {} {}\n'.format('stereo' if flex.has_partner else 'mono',
                                                          flex.flex_name,
                                                          self.mdl.file_data.flex_descs[
                                                              flex.partner].name if flex.has_partner else ''))
                fileh.write('\n')
            fileh.write('\t}\n')
            fileh.write('}\n\n')

        else:
            fileh.write('$model "{0}" "{0}"\n\n'.format(model_name))

    def write_bodygroup(self, fileh, bp):
        fileh.write('$bodygroup "{}"\n'.format(bp.name))
        fileh.write('{\n')
        for model in bp.models:  # type: SourceMdlModel
            if model.mesh_count == 0:
                fileh.write("\tblank\n")
            else:
                if not self.fst_model:
                    self.fst_model = model.name
                model_name = str(
                    Path(model.name).with_suffix('').with_suffix(''))
                fileh.write('\tstudio "{}"\n'.format(model_name))
                if model.flex_frames and self.vvd:
                    fileh.write(
                        "//WARNING: this {} have flexes! Additional VTA will be written, you can import them manually\n"
                        "//If you want to compile it back correctly - use Session format\n".format(
                            model.name)
                    )
                    if self.vta:
                        self.vta.write_vta(model)
        fileh.write('}\n\n')

    def write_skins(self, fileh):
        fileh.write('$texturegroup "skinfamilies"{\n')
        for skin_fam in self.mdl.file_data.skin_families:
            fileh.write('{')
            for mat in skin_fam:
                mat_name = self.mdl.file_data.textures[mat].path_file_name
                fileh.write('"{}" '.format(mat_name))
                pass
            fileh.write('}\n')

        fileh.write('}\n\n')

    def write_misc(self, fileh):
        fileh.write('$surfaceprop "{}"\n\n'.format(self.mdl.file_data.surface_prop_name))
        deflection = math.acos(self.mdl.file_data.max_eye_deflection)
        deflection = math.degrees(deflection)
        fileh.write('$maxeyedeflection {:.1f}\n\n'.format(deflection))
        fileh.write('$eyeposition {}\n\n'.format(self.mdl.file_data.eye_position.as_string_smd))

        if self.mdl.file_data.flags & self.mdl.file_data.flags.STUDIOHDR_FLAGS_AMBIENT_BOOST:
            fileh.write('$ambientboost\n')
        if self.mdl.file_data.flags & self.mdl.file_data.flags.STUDIOHDR_FLAGS_TRANSLUCENT_TWOPASS:
            fileh.write('$mostlyopaque\n')
        if self.mdl.file_data.flags & self.mdl.file_data.flags.STUDIOHDR_FLAGS_STATIC_PROP:
            fileh.write('$staticprop\n')
        if self.mdl.file_data.flags & self.mdl.file_data.flags.STUDIOHDR_FLAGS_SUBDIVISION_SURFACE:
            fileh.write('$subd\n')

        fileh.write('\n')

        self.write_texture_paths(fileh)
        self.write_used_materials(fileh)
        self.write_attachment(fileh)
        fileh.write('$cbox {} {}\n\n'.format(self.mdl.file_data.view_bounding_box_min_position.as_rounded(3),
                                             self.mdl.file_data.view_bounding_box_max_position.as_rounded(3)))
        fileh.write('$bbox {} {}\n\n'.format(self.mdl.file_data.hull_min_position.as_rounded(3),
                                             self.mdl.file_data.hull_max_position.as_rounded(3)))

    def write_texture_paths(self, fileh):
        for n, texture_path in enumerate(self.mdl.file_data.texture_paths):
            if n == 0 and not texture_path:
                fileh.write('$cdmaterials "{}"\n'.format(texture_path))
            elif texture_path:
                fileh.write('$cdmaterials "{}"\n'.format(texture_path))
        fileh.write('\n')

    def write_used_materials(self, fileh):
        fileh.write('//USED MATERISLS:\n')
        for texture in self.mdl.file_data.textures:
            fileh.write('\t//{}\n'.format(texture.path_file_name))
        fileh.write('\n')

    def write_attachment(self, fileh):
        for attachment in self.mdl.file_data.attachments:  # type: SourceMdlAttachment

            bone = self.mdl.file_data.bones[attachment.localBoneIndex]
            fileh.write('$attachment "{}" "{}" {} rotate {}\n'.format(attachment.name, bone.name,
                                                                      SourceVector([attachment.localM14,
                                                                                    attachment.localM24,
                                                                                    attachment.localM34]).as_rounded(2),
                                                                      SourceVector(convert_rotation_matrix_to_degrees(
                                                                          attachment.localM11, attachment.localM21,
                                                                          attachment.localM31, attachment.localM12,
                                                                          attachment.localM22, attachment.localM32,
                                                                          attachment.localM33)).to_degrees().as_rounded(
                                                                          2)))
        fileh.write('\n')

    def write_jiggle_bones(self, fileh):
        for bone in self.mdl.file_data.bones:
            if bone.jiggle_bone is not None:
                jbone = bone.jiggle_bone
                fileh.write('$jigglebone {} '.format(bone.name))
                fileh.write('{\n')
                if jbone.flags & jbone.flags.IS_FLEXIBLE:
                    fileh.write("\tis_flexible {\n")
                    fileh.write('\t\tlength {}\n'.format(jbone.length))
                    fileh.write('\t\ttip_mass {}\n'.format(jbone.tip_mass))
                    fileh.write('\t\tpitch_stiffness {}\n'.format(jbone.pitch_stiffness))
                    fileh.write('\t\tpitch_damping {}\n'.format(jbone.pitch_damping))
                    fileh.write('\t\tyaw_stiffness {}\n'.format(jbone.yaw_stiffness))
                    fileh.write('\t\tyaw_damping {}\n'.format(jbone.yaw_damping))
                    if jbone.flags & jbone.flags.HAS_LENGTH_CONSTRAINT:
                        fileh.write('\t\talong_stiffness {}\n'.format(jbone.along_stiffness))
                        fileh.write('\t\talong_damping {}\n'.format(jbone.along_damping))
                    if jbone.flags & jbone.flags.HAS_ANGLE_CONSTRAINT:
                        fileh.write('\t\tangle_constraint {}\n'.format(round(jbone.angle_limit * 180 / 3.1415, 3)))
                    if jbone.flags & jbone.flags.HAS_PITCH_CONSTRAINT:
                        fileh.write('\t\tpitch_constraint {} {}\n'.format(jbone.min_pitch, jbone.max_pitch))
                        fileh.write('\t\tpitch_friction {}\n'.format(jbone.pitch_friction))
                    if jbone.flags & jbone.flags.HAS_YAW_CONSTRAINT:
                        fileh.write('\t\tyaw_constraint {} {}\n'.format(jbone.min_yaw, jbone.max_yaw))
                        fileh.write('\t\tyaw_friction  {}\n'.format(jbone.yaw_friction))
                    fileh.write('\t}\n')

                if jbone.flags & jbone.flags.IS_RIGID:
                    fileh.write('is_rigid {\n')
                    fileh.write('}\n')
                fileh.write('}\n\n')
        pass

    def write_sequences(self, fileh):
        fileh.write(
            '$sequence "{}" '.format('idle'))
        fileh.write('{\n')
        fileh.write('\t"{}"\n'.format(self.fst_model))
        fileh.write('\tactivity "{}" 1\n'.format("ACT_DIERAGDOLL"))
        fileh.write('\tfadein {:.2f}\n'.format(0.2))
        fileh.write('\tfadeout {:.2f}\n'.format(0.2))
        fileh.write('\tfps {}\n'.format(30))
        fileh.write('}')
