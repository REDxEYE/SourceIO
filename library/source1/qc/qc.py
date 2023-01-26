import math
from pathlib import Path

from ...utils.math_utilities import vector_transform
from ..mdl.structs.bodygroup import BodyPart
from ..mdl.structs.bone import ProceduralBoneType
from ..mdl.structs.header import StudioHDRFlags
from ..mdl.v49.mdl_file import MdlV49


def generate_qc(mdl: MdlV49, buffer, plugin_version="UNKNOWN"):
    buffer.write(f"// Created by SourceIO v{plugin_version}\n\n")

    buffer.write(f"$modelname \"{mdl.header.name}\"\n")

    def write_model(bodygroup: BodyPart):
        model = bodygroup.models[0]
        name = Path(model.name if (model.name and model.name != 'blank') else model.name).stem
        buffer.write(f"$model \"{name}\" \"{name}\"")
        if model.has_flexes or model.has_eyebals:
            buffer.write("{\n\n")

            if model.has_eyebals:
                for n, eyeball in enumerate(model.eyeballs):
                    buffer.write('\teyeball')
                    diameter = eyeball.radius * 2
                    angle = round(math.degrees(math.atan(eyeball.z_offset)), 6)
                    iris_scale = 1 / eyeball.iris_scale

                    if n == 0 and angle > 0:
                        buffer.write(' "eye_right"')
                    elif n == 1 and angle < 0:
                        buffer.write(' "eye_left"')
                    else:
                        buffer.write(' "eye_{}"'.format(n))

                    bone = mdl.bones[eyeball.bone_index]
                    buffer.write(f" \"{bone.name}\"")
                    pos = vector_transform(eyeball.org, bone.pose_to_bone)
                    buffer.write(f" {pos[0]:.4} {pos[1]:.4} {pos[2]:.4}")
                    buffer.write(f" \"{mdl.materials[eyeball.material_id].name}\"")

                    buffer.write(" {}".format(diameter))
                    buffer.write(" {}".format(angle))
                    buffer.write(" \"iris_unused\"")
                    buffer.write(" {}".format(int(iris_scale)))
                    buffer.write("\n")
                buffer.write("\n")
            if model.has_flexes:
                all_flexes = []
                for mesh in model.meshes:
                    all_flexes.extend(mesh.flexes)
                for flex in set(all_flexes):
                    flex_name = mdl.flex_names[flex.flex_desc_index]
                    buffer.write('\t//{} {} {}\n'.format('stereo' if flex.partner_index > 0 else 'mono',
                                                         flex_name,
                                                         mdl.flex_names[
                                                             flex.partner_index] if flex.partner_index > 0 else ''))
            buffer.write("}\n")
        else:
            buffer.write("\n")

    def write_bodygroup(bodygroup: BodyPart):
        buffer.write(f"$bodygroup \"{bodygroup.name}\" {{\n")
        for model in bodygroup.models:
            if len(model.meshes) == 0:
                buffer.write("\tblank\n")
            else:
                model_name = Path(model.name).stem
                buffer.write(f"\tstudio \"{model_name}\"\n")
        buffer.write("}\n")

    def write_skins():
        buffer.write('$texturegroup "skinfamilies"{\n')
        for skin_fam in mdl.skin_groups:
            buffer.write('{')
            for mat in skin_fam:
                mat_name = mat
                buffer.write('"{}" '.format(mat_name))
            buffer.write('}\n')

        buffer.write('}\n\n')

    def write_misc():
        buffer.write(f"$surfaceprop \"{mdl.header.surface_prop}\"\n")
        deflection = math.degrees(math.acos(mdl.header.max_eye_deflection))
        buffer.write(f"$maxeyedeflection {deflection:.1f}\n")
        eye_pos = mdl.header.eye_position
        buffer.write(f"$eyeposition {eye_pos[0]:.3} {eye_pos[1]:.3} {eye_pos[2]:.3}\n")

        if mdl.header.flags & StudioHDRFlags.AMBIENT_BOOST:
            buffer.write('$ambientboost\n')
        if mdl.header.flags & StudioHDRFlags.TRANSLUCENT_TWOPASS:
            buffer.write('$mostlyopaque\n')
        if mdl.header.flags & StudioHDRFlags.STATIC_PROP:
            buffer.write('$staticprop\n')
        if mdl.header.flags & StudioHDRFlags.SUBDIVISION_SURFACE:
            buffer.write('$subd\n')

        buffer.write('\n')

    def write_texture_paths():
        for n, texture_path in enumerate(mdl.materials_paths):
            if n == 0 and not texture_path:
                buffer.write('$cdmaterials "{}"\n'.format(texture_path))
            elif texture_path:
                buffer.write('$cdmaterials "{}"\n'.format(texture_path))
        buffer.write('\n')

    def write_used_materials():
        buffer.write('//USED MATERIALS:\n')
        for texture in mdl.materials:
            buffer.write('\t//{}\n'.format(texture.name))
        buffer.write('\n')

    for bodygroup in mdl.body_parts:
        if len(bodygroup.models) == 1:
            write_model(bodygroup)
        elif len(bodygroup.models) > 1:
            write_bodygroup(bodygroup)

    def write_jiggle_bones():
        for bone in mdl.bones:
            if bone.procedural_rule is not None:
                if bone.procedural_rule_type == ProceduralBoneType.JIGGLE:
                    jbone = bone.procedural_rule
                    buffer.write(f"$jigglebone {bone.name} ")
                    buffer.write('{\n')
                    if jbone.flags & jbone.flags.IS_FLEXIBLE:
                        buffer.write("\tis_flexible {\n")
                        buffer.write('\t\tlength {}\n'.format(jbone.length))
                        buffer.write('\t\ttip_mass {}\n'.format(jbone.tip_mass))
                        buffer.write('\t\tpitch_stiffness {}\n'.format(jbone.pitch_stiffness))
                        buffer.write('\t\tpitch_damping {}\n'.format(jbone.pitch_damping))
                        buffer.write('\t\tyaw_stiffness {}\n'.format(jbone.yaw_stiffness))
                        buffer.write('\t\tyaw_damping {}\n'.format(jbone.yaw_damping))
                        if jbone.flags & jbone.flags.HAS_LENGTH_CONSTRAINT:
                            buffer.write('\t\talong_stiffness {}\n'.format(jbone.along_stiffness))
                            buffer.write('\t\talong_damping {}\n'.format(jbone.along_damping))
                        if jbone.flags & jbone.flags.HAS_ANGLE_CONSTRAINT:
                            buffer.write('\t\tangle_constraint {}\n'.format(round(jbone.angle_limit * 180 / 3.1415, 3)))
                        if jbone.flags & jbone.flags.HAS_PITCH_CONSTRAINT:
                            buffer.write('\t\tpitch_constraint {} {}\n'.format(jbone.min_pitch, jbone.max_pitch))
                            buffer.write('\t\tpitch_friction {}\n'.format(jbone.pitch_friction))
                        if jbone.flags & jbone.flags.HAS_YAW_CONSTRAINT:
                            buffer.write('\t\tyaw_constraint {} {}\n'.format(jbone.min_yaw, jbone.max_yaw))
                            buffer.write('\t\tyaw_friction  {}\n'.format(jbone.yaw_friction))
                        buffer.write('\t}\n')

                    if jbone.flags & jbone.flags.IS_RIGID:
                        buffer.write('is_rigid {\n')
                        buffer.write('}\n')
                    buffer.write('}\n\n')

    def write_sequences():
        buffer.write(f"$sequence \"idle\" ")
        buffer.write("{\n")
        file_name = Path(mdl.body_parts[0].models[0].name).stem
        buffer.write(f"\t\"{file_name}\"\n")
        buffer.write("\tactivity \"ACT_DIERAGDOLL\" 1\n")
        buffer.write(f"\tfadein {0.2:.2f}\n")
        buffer.write(f"\tfadeout {0.2:.2f}\n")
        buffer.write("\tfps 30\n")
        buffer.write("}\n")

    buffer.write('\n')
    write_skins()
    write_misc()
    write_texture_paths()
    write_used_materials()
    write_jiggle_bones()
    write_sequences()
