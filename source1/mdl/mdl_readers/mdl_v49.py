from ...data_structures.mdl_anim_data import *
from ...data_structures.mdl_data import *
from ....utilities.progressbar import ProgressBar


class SourceMdlFile49:

    def __init__(self, reader: ByteIO):
        self.reader = reader
        self.file_data = SourceMdlFileData()

    def read(self):
        self.file_data.read(self.reader)
        self.read_bones()
        self.read_bone_controllers()
        self.read_skin_families()
        self.read_flex_descs()
        self.read_flex_controllers()
        self.read_flex_rules()

        self.read_local_animation_descs()

        self.read_attachments()
        self.read_mouths()
        self.read_bone_flex_drivers()
        self.read_flex_controllers_ui()
        self.read_body_parts()
        self.read_textures()
        self.read_texture_paths()
        self.build_flex_frames()
        self.prepare_models()

    def read_bones(self):
        if self.file_data.bone_count > 0:
            pb = ProgressBar(
                desc='Reading bones',
                max_=self.file_data.bone_count,
                len_=20)
            self.reader.seek(self.file_data.bone_offset, 0)
            for i in range(self.file_data.bone_count):
                pb.draw()
                bone = SourceMdlBone()
                self.file_data.register(bone)
                bone.read(self.reader, self.file_data)
                pb.increment(1)
                pb.draw()

    def read_bone_controllers(self):
        if self.file_data.bone_controller_count > 0:
            pb = ProgressBar(
                desc='Reading Bone Controllers',
                max_=self.file_data.bone_controller_count,
                len_=20)
            for _ in range(self.file_data.bone_controller_count):
                pb.draw()
                SourceMdlBoneController().read(self.reader, self.file_data)
                pb.increment(1)

    def read_skin_families(self):
        if self.file_data.skin_family_count and self.file_data.skin_reference_count:
            self.reader.seek(self.file_data.skin_family_offset)
            for _ in range(self.file_data.skin_family_count):
                skin_ref = []
                for _ in range(self.file_data.skin_reference_count):
                    skin_ref.append(self.reader.read_int16())
                self.file_data.skin_families.append(skin_ref)

    def read_flex_descs(self):
        if self.file_data.flex_desc_count > 0:
            self.reader.seek(self.file_data.flex_desc_offset, 0)
            pb = ProgressBar(
                desc='Reading flex descriptions',
                max_=self.file_data.flex_desc_count,
                len_=20)
            for _ in range(self.file_data.flex_desc_count):
                pb.draw()
                flex_desc = SourceMdlFlexDesc()
                flex_desc.read(self.reader)
                pb.increment(1)
                self.file_data.flex_descs.append(flex_desc)

    def read_flex_controllers(self):
        if self.file_data.flex_controller_count > 0:
            self.reader.seek(self.file_data.flex_controller_offset, 0)
            pb = ProgressBar(
                desc='Reading flex Controllers',
                max_=self.file_data.flex_controller_count,
                len_=20)
            for i in range(self.file_data.flex_controller_count):
                pb.draw()
                SourceMdlFlexController().read(self.reader, self.file_data)
                pb.increment(1)

    def read_flex_rules(self):
        self.reader.seek(self.file_data.flex_rule_offset, 0)
        pb = ProgressBar(
            desc='Reading flex rules',
            max_=self.file_data.flex_rule_count,
            len_=20)
        for i in range(self.file_data.flex_rule_count):
            pb.draw()
            SourceMdlFlexRule().read(self.reader, self.file_data)
            pb.increment(1)

    def read_attachments(self):
        if self.file_data.local_attachment_count > 0:
            self.reader.seek(self.file_data.local_attachment_offset, 0)
            pb = ProgressBar(
                desc='Reading attachments',
                max_=self.file_data.local_attachment_count,
                len_=20)
            for _ in range(self.file_data.local_attachment_count):
                pb.draw()
                SourceMdlAttachment().read(self.reader, self.file_data)
                pb.increment(1)

    def read_body_parts(self):
        if self.file_data.body_part_count > 0:
            self.reader.seek(self.file_data.body_part_offset)
            pb = ProgressBar(
                desc='Reading body parts',
                max_=self.file_data.body_part_count,
                len_=20)
            for _ in range(self.file_data.body_part_count):
                pb.draw()
                SourceMdlBodyPart().read(self.reader, self.file_data)
                pb.increment(1)

    def read_textures(self):
        if self.file_data.texture_count < 1:
            return
        self.reader.seek(self.file_data.texture_offset)
        for _ in range(self.file_data.texture_count):
            SourceMdlTexture().read(self.reader, self.file_data)

    def read_texture_paths(self):
        if self.file_data.texture_path_count > 0:
            self.reader.seek(self.file_data.texture_path_offset)
            for _ in range(self.file_data.texture_path_count):
                texture_path_offset = self.reader.read_uint32()
                entry = self.reader.tell()
                if texture_path_offset != 0:
                    self.file_data.texture_paths.append(
                        self.reader.read_from_offset(texture_path_offset, self.reader.read_ascii_string))
                else:
                    self.file_data.texture_paths.append("")
                self.reader.seek(entry)

    def read_local_animation_descs(self):
        self.reader.seek(self.file_data.local_animation_offset)
        for i in range(self.file_data.local_animation_count):
            anim_desc = SourceAnimDesc()
            anim_desc.read(self.reader)
            self.file_data.register(anim_desc)

    #     self.reader.seek(self.file_data.local_animation_offset)
    #     with self.reader.save_current_pos():
    #         for _ in range(self.file_data.local_animation_count):
    #             self.file_data.animation_descs.append(SourceMdlAnimationDesc49().read(self.reader, self.file_data))
    #     self.read_animations()
    #
    # def read_sequences(self):
    #     with self.reader.save_current_pos():
    #         self.reader.seek(self.file_data.local_sequence_offset)
    #         pb = ProgressBar(desc='Reading sequences', max_=self.file_data.local_sequence_count, len_=20)
    #         for _ in range(self.file_data.local_sequence_count):
    #             self.file_data.sequence_descs.append(SourceMdlSequenceDesc().read(self.reader, self.file_data))
    #             pb.increment(1)

    def read_mouths(self):
        if self.file_data.mouth_count and self.file_data.mouth_offset:
            self.reader.seek(self.file_data.mouth_offset)
            for _ in range(self.file_data.mouth_count):
                mouth = SourceMdlMouth()
                mouth.read(self.reader)
                self.file_data.mouths.append(mouth)

    def read_flex_controllers_ui(self):
        if self.file_data.flex_controller_ui_count and self.file_data.flex_controller_ui_offset:
            self.reader.seek(self.file_data.flex_controller_ui_offset)
            for _ in range(self.file_data.flex_controller_ui_count):
                flex_controller_ui = SourceFlexControllerUI()
                flex_controller_ui.read(self.reader)
                self.file_data.flex_controllers_ui.append(flex_controller_ui)

    def read_bone_flex_drivers(self):
        if self.file_data.bone_flex_driver_count and self.file_data.bone_flex_driver_offset:
            self.reader.seek(self.file_data.bone_flex_driver_count)
            for _ in range(self.file_data.bone_flex_driver_count):
                mouth = SourceMdlMouth()
                mouth.read(self.reader)
                self.file_data.mouths.append(mouth)

    # def read_animations(self):
    #     for i in range(self.file_data.local_animation_count):
    #         anim_desc = self.file_data.animation_descs[i]  # type: SourceMdlAnimationDesc49
    #         print('Reading anim', anim_desc.theName, 'flags', anim_desc.flags.get_flags)
    #         print(anim_desc)
    #         anim_desc.theSectionsOfAnimations = [[]]
    #         entry = anim_desc.entry + i * anim_desc.size
    #
    #         if anim_desc.flags.flag & anim_desc.STUDIO.ALLZEROS == 0:
    #
    #             if anim_desc.flags.flag & anim_desc.STUDIO.FRAMEANIM != 0:
    #                 if anim_desc.sectionOffset != 0 and anim_desc.sectionFrameCount > 0:
    #                     self.file_data.section_frame_count = anim_desc.sectionFrameCount
    #                     if self.file_data.section_frame_min_frame_count >= anim_desc.frameCount:
    #                         self.file_data.section_frame_min_frame_count = anim_desc.frameCount - 1
    #                     section_count = math.trunc(anim_desc.frameCount / anim_desc.sectionFrameCount) + 2
    #                     for sectionIndex in range(section_count):
    #                         anim_desc.theSectionsOfAnimations.append([])
    #                     with self.reader.save_current_pos():
    #                         self.reader.seek(entry + anim_desc.sectionOffset)
    #                         for _ in range(section_count):
    #                             pass
    #                             anim_desc.theSections.append(SourceMdlAnimationSection().read(self.reader))
    #             else:
    #                 if anim_desc.sectionOffset != 0 and anim_desc.sectionFrameCount > 0:
    #                     self.file_data.section_frame_count = anim_desc.sectionFrameCount
    #                     if self.file_data.section_frame_min_frame_count >= anim_desc.frameCount:
    #                         self.file_data.section_frame_min_frame_count = anim_desc.frameCount - 1
    #                     section_count = math.trunc(anim_desc.frameCount / anim_desc.sectionFrameCount) + 2
    #                     # print(section_count)
    #                     for sectionIndex in range(section_count):
    #                         anim_desc.theSectionsOfAnimations.append([])
    #                     with self.reader.save_current_pos():
    #                         self.reader.seek(entry + anim_desc.sectionOffset)
    #                         for _ in range(section_count):
    #                             pass
    #                             anim_desc.theSections.append(SourceMdlAnimationSection().read(self.reader))
    #                 if anim_desc.animBlock == 0:
    #                     with self.reader.save_current_pos():
    #                         self.reader.seek(entry + anim_desc.animOffset)
    #                         for _ in range(self.file_data.bone_count):
    #                             entry_anim = self.reader.tell()
    #                             print('Trying to read animation from offset', entry_anim)
    #                             pass
    #                             anim, stat = SourceMdlAnimation().read(anim_desc.frameCount,
    #                                                                    anim_desc.theSectionsOfAnimations[0],
    #                                                                    self.file_data,
    #                                                                    self.reader)
    #                             if stat == -1:
    #                                 print('Success, breaking the loop')
    #                                 break
    #                             if stat == 1:
    #                                 anim_desc.theSectionsOfAnimations.append(anim)
    #                             if stat == 0:
    #                                 print('ERROR, breaking the loop')
    #                                 break

    # pprint(anim_desc.__dict__)

    def build_flex_frames(self):
        flex_dest_flex_frame = []  # type:List[List[FlexFrame]]
        for x in range(len(self.file_data.flex_descs)):
            flex_dest_flex_frame.append([])

        cumulative_vertex_offset = 0
        for body_part in self.file_data.body_parts:
            print('Building flex frame for {}'.format(body_part.name))

            for model in body_part.models:
                print(
                    '\tProcessing model {} with {} flexes'.format(
                        model.name, model.flex_count))

                for mesh in model.meshes:
                    vertex_offset = mesh.vertex_index_start

                    for flex_index, flex in enumerate(mesh.flexes):
                        flex_frame = None
                        if flex_dest_flex_frame[flex.flex_desc_index]:
                            for s_flex in flex_dest_flex_frame[flex.flex_desc_index]:
                                if s_flex.flexes[0].target0 == flex.target0 and \
                                        s_flex.flexes[0].target1 == flex.target1 and \
                                        s_flex.flexes[0].target2 == flex.target2 and \
                                        s_flex.flexes[0].target3 == flex.target3:
                                    flex_frame = s_flex

                        if not flex_frame:
                            flex_frame = FlexFrame()
                            flex_frame.flex_name = self.file_data.flex_descs[flex.flex_desc_index].name
                            flex_desc_partner_index = mesh.flexes[flex_index].flex_desc_partner_index

                            if flex_desc_partner_index > 0:
                                flex_frame.has_partner = True
                                flex_frame.partner = flex_desc_partner_index

                            flex_dest_flex_frame[flex.flex_desc_index].append(
                                flex_frame)

                        flex_frame.vertex_offsets.append(
                            vertex_offset + cumulative_vertex_offset)
                        flex_frame.flexes.append(flex)
                        model.flex_frames.append(flex_frame)

                cumulative_vertex_offset += model.vertex_count

    @staticmethod
    def comp_flex_frames(flex_frame1, flex_frame2):
        if len(flex_frame1) + len(flex_frame2) == 0:
            return False
        if len(flex_frame1) != len(flex_frame2):
            return False
        flex_frame1 = sorted(flex_frame1, key=lambda a: a.flex_name)
        flex_frame2 = sorted(flex_frame2, key=lambda a: a.flex_name)
        for flex1, flex2 in zip(flex_frame1, flex_frame2):
            if flex1 != flex2:
                return False
        return True

    def prepare_models(self):
        for n, body_part in enumerate(self.file_data.body_parts):
            if body_part.model_count > 1:
                self.file_data.bodypart_frames.append([(n, body_part)])
                continue
            if body_part.model_count == 0:
                continue
            model = body_part.models[0]
            # print('Scanning,', body_part.name)
            if 'clamped' not in body_part.name:
                # print(
                #     'Skipping',
                #     model.name,
                #     'cuz it\'s not a clamped mesh_data')
                self.file_data.bodypart_frames.append([(n, body_part)])
                continue
            added = False
            for body_part_frames in self.file_data.bodypart_frames:
                for _, _model in body_part_frames:
                    # print('Comparing', model.name, 'to', _model)
                    if self.comp_flex_frames(
                            model.flex_frames, _model.models[0].flex_frames):
                        # print('Adding', model.name, 'to', body_part_frames)
                        body_part_frames.append((n, body_part))
                        added = True
                        break
            if not added:
                self.file_data.bodypart_frames.append([(n, body_part)])

    def test(self):
        with open(r'.\test_data\nick_hwm_bac.mdl', 'wb') as fp:
            writer = ByteIO(file=fp)
            self.file_data.write(writer)
