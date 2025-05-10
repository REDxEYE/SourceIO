from SourceIO.blender_bindings.source1.bsp.entities.base_entity_classes import Targetname, RenderFields


def parse_source_value(value):
    if type(value) is str:
        value: str
        if value.replace('.', '', 1).replace('-', '', 1).isdecimal():
            return float(value) if '.' in value else int(value)
        return 0
    else:
        return value


def parse_int_vector(string):
    return [parse_source_value(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


class Base:
    hammer_id_counter = 0

    def __init__(self, entity_data: dict):
        self._hammer_id = -1
        self._raw_data = entity_data

    @classmethod
    def new_hammer_id(cls):
        new_id = cls.hammer_id_counter
        cls.hammer_id_counter += 1
        return new_id

    @property
    def class_name(self):
        return self._raw_data.get('classname')
        
    @property
    def hammer_id(self):
        if self._hammer_id == -1:
            if 'hammerid' in self._raw_data:
                self._hammer_id = int(self._raw_data.get('hammerid'))
            else:  # Titanfall
                self._hammer_id = Base.new_hammer_id()
        return self._hammer_id


class c_VEnableDisable(Base):

    @property
    def start_enabled(self):
        return self._raw_data.get('start_enabled', "1")



class VItemContainerBase1(Base):

    @property
    def model(self):
        return self._raw_data.get('model', "models/scenery/furniture/refrigerator/refrigeratorold_anim2.mdl")



class VItemContainerEquip(Base):

    @property
    def sep_itceqip(self):
        return self._raw_data.get('sep_itceqip', None)

    @property
    def equip0(self):
        return self._raw_data.get('equip0', None)

    @property
    def equip1(self):
        return self._raw_data.get('equip1', None)

    @property
    def equip2(self):
        return self._raw_data.get('equip2', None)

    @property
    def equip3(self):
        return self._raw_data.get('equip3', None)

    @property
    def equip4(self):
        return self._raw_data.get('equip4', None)



class VItemContainerBase2(Base):

    @property
    def sep_itcadd1(self):
        return self._raw_data.get('sep_itcadd1', None)

    @property
    def soundgroup(self):
        return self._raw_data.get('soundgroup', "push_arm_door")

    @property
    def lootable_type(self):
        return self._raw_data.get('lootable_type', "3")

    @property
    def use_icon(self):
        return self._raw_data.get('use_icon', "6")

    @property
    def sep_itcadd2(self):
        return self._raw_data.get('sep_itcadd2', None)

    @property
    def move_dest(self):
        return self._raw_data.get('move_dest', None)

    @property
    def move_speed(self):
        return parse_source_value(self._raw_data.get('move_speed', 45))

    @property
    def rot_dist(self):
        return parse_source_value(self._raw_data.get('rot_dist', 90))

    @property
    def rot_speed(self):
        return parse_source_value(self._raw_data.get('rot_speed', 45))

    @property
    def rot_axis(self):
        return self._raw_data.get('rot_axis', "1")

    @property
    def sep_itcadd3(self):
        return self._raw_data.get('sep_itcadd3', None)

    @property
    def solid(self):
        return self._raw_data.get('solid', "6")

    @property
    def health(self):
        return parse_source_value(self._raw_data.get('health', 0))

    @property
    def dmgmodel(self):
        return self._raw_data.get('dmgmodel', None)

    @property
    def npc_transparent(self):
        return self._raw_data.get('npc_transparent', "1")

    @property
    def npc_opaque(self):
        return self._raw_data.get('npc_opaque', None)

    @property
    def demo_sequence(self):
        return self._raw_data.get('demo_sequence', "None")

    @property
    def use_pref(self):
        return self._raw_data.get('use_pref', None)



class item_container(RenderFields, VItemContainerBase1, VItemContainerEquip, VItemContainerBase2):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_container_animated(RenderFields, VItemContainerBase1, VItemContainerEquip, VItemContainerBase2):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_container_one_item_filtered(RenderFields, VItemContainerBase1, VItemContainerBase2):
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
    pass


class item_a_body_armor(Base):
    model = "models/scenery/misc/armor/RiotGear/RiotGear.mdl"
    pass


class item_a_hvy_cloth(Base):
    model = "models/scenery/misc/armor/ClothesHeavy/ClothesHeavy.mdl"
    pass


class item_a_hvy_leather(Base):
    model = "models/scenery/misc/armor/LeatherHeavy/LeatherHeavy.mdl"
    pass


class item_a_lt_cloth(Base):
    model = "models/scenery/misc/armor/ClothesLight/ClothesLight.mdl"
    pass


class item_d_animalism(Base):
    pass


class item_d_dementation(Base):
    pass


class item_d_dominate(Base):
    pass


class item_d_holy_light(Base):
    model = "models/weapons/holylight/view/v_holylight.mdl"
    pass


class item_d_thaumaturgy(Base):
    model = "models/weapons/thaumaturgy/view/v_thaumaturgy.mdl"
    pass


class item_g_bloodpack(Base):
    model = "models/items/bloodpack/ground/bloodpack.mdl"
    pass


class item_g_bluebloodpack(Base):
    model = "models/items/bloodpack/ground/bloodpack.mdl"
    pass


class item_g_eldervitaepack(Base):
    model = "models/items/bloodpack/ground/bloodpack.mdl"
    pass


class item_g_keyring(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_g_lockpick(Base):
    model = "models/weapons/lockpicks/world/w_lockpicks.mdl"
    pass


class item_g_wallet(Base):
    model = "models/items/Wallet/Ground/Wallet.mdl"
    pass


class item_g_sewerbook_1(Base):
    model = "models/items/sewerbook/Ground/sewerbook.mdl"
    pass


class item_g_animaltrainingbook(Base):
    model = "models/items/AnimalTrainingBook/Ground/AnimalTrainingBook.mdl"
    pass


class item_g_astrolite(Base):
    model = "models/items/Astrolite/Ground/Astrolite.mdl"
    pass


class item_g_bach_journal(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_badlucktalisman(Base):
    model = "models/items/Talisman/ground/Talisman.mdl"
    pass


class item_g_bailbond_receipt(Base):
    model = "models/items/BailBond/Ground/BailBond.mdl"
    pass


class item_g_bertrams_cd(Base):
    model = "models/items/cdcase/Ground/cdcase.mdl"
    pass


class item_g_brotherhood_flyer(Base):
    model = "models/items/brotherhoodflyer/Ground/brotherhoodflyer.mdl"
    pass


class item_g_driver_license_gimble(Base):
    model = "models/items/License/Ground/License.mdl"
    pass


class item_g_edane_print_report(Base):
    model = "models/items/manifest/Ground/manifest_g.mdl"
    pass


class item_g_edane_report(Base):
    model = "models/items/clipboard/ground/clipboard.mdl"
    pass


class item_g_eyes(Base):
    model = "models/items/Eyeballs/Ground/Eyeballs.mdl"
    pass


class item_g_gargoyle_book(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_ghost_pendant(Base):
    model = "models/items/Pendant/ground/Pendant.mdl"
    pass


class item_g_giovanni_invitation_maria(Base):
    model = "models/items/invitation/ground/invitation.mdl"
    pass


class item_g_giovanni_invitation_victor(Base):
    model = "models/items/invitation/ground/invitation.mdl"
    pass


class item_g_guy_magazine(Base):
    model = "models/items/guymag/Ground/guymag.mdl"
    pass


class item_g_hannahs_appt_book(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_hatters_screenplay(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_horrortape_1(Base):
    model = "models/items/HorrorTape_1/Ground/HorrorTape_1.mdl"
    pass


class item_g_horrortape_2(Base):
    model = "models/items/HorrorTape_2/Ground/HorrorTape_2.mdl"
    pass


class item_g_idol_cat(Base):
    model = "models/items/jadecat/Ground/jadecat.mdl"
    pass


class item_g_idol_crane(Base):
    model = "models/items/jadecrane/Ground/jadecrane.mdl"
    pass


class item_g_idol_dragon(Base):
    model = "models/items/jadedragon/Ground/jadedragon.mdl"
    pass


class item_g_idol_elephant(Base):
    model = "models/items/jadeelephant/Ground/jadeelephant.mdl"
    pass


class item_g_jumbles_flyer(Base):
    model = "models/items/Flyer/ground/flyer.mdl"
    pass


class item_g_junkyard_businesscard(Base):
    model = "models/items/businesscard/Ground/businesscard2.mdl"
    pass


class item_g_larry_briefcase(Base):
    model = "models/items/briefcase/Ground/briefcase.mdl"
    pass


class item_g_lilly_diary(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_lilly_photo(Base):
    model = "models/items/LillyOnBeachPhoto/Ground/LillyOnBeachPhoto.mdl"
    pass


class item_g_lilly_purse(Base):
    model = "models/items/Purse/Ground/Purse.mdl"
    pass


class item_g_lillyonbeachphoto(Base):
    model = "models/items/LillyOnBeachPhoto/Ground/LillyOnBeachPhoto.mdl"
    pass


class item_g_mercurio_journal(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_milligans_businesscard(Base):
    model = "models/items/businesscard/Ground/businesscard.mdl"
    pass


class item_g_oh_diary(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_pisha_book(Base):
    model = "models/items/skinbook/Ground/skinbook.mdl"
    pass


class item_g_pisha_fetish(Base):
    model = "models/items/FetishStatue/ground/FetishStatue.mdl"
    pass


class item_g_werewolf_bloodpack(Base):
    model = "models/items/bloodpack/ground/bloodpack.mdl"
    pass


class item_g_wireless_camera_1(Base):
    model = "models/items/webcam/Ground/webcam.mdl"
    pass


class item_g_wireless_camera_2(Base):
    model = "models/items/webcam/Ground/webcam.mdl"
    pass


class item_g_wireless_camera_3(Base):
    model = "models/items/webcam/Ground/webcam.mdl"
    pass


class item_g_wireless_camera_4(Base):
    model = "models/items/webcam/Ground/webcam.mdl"
    pass


class item_g_warrens4_passkey(Base):
    model = "models/items/sewercard/Ground/sewercard.mdl"
    pass


class item_k_ash_cell_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_carson_apartment_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_chinese_theatre_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_clinic_cs_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_clinic_maintenance_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_clinic_stairs_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_edane_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_empire_jezebel_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_empire_mafia_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_fu_cell_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_fu_office_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_gallery_noir_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_gimble_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_hannahs_safe_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_hitman_ji_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_hitman_lu_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_kiki_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_leopold_int_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_lilly_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_lucky_star_murder_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_malcolm_office_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_malkavian_refrigerator_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_murietta_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_museum_basement_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_museum_office_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_museum_storage_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_museum_storeroom_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_netcafe_office_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_oceanhouse_basement_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_oceanhouse_sewer_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_oceanhouse_upstairs_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_oh_front_key(Base):
    model = "models/items/Key/Ground/Key.mdl"
    pass


class item_k_sarcophagus_key(Base):
    model = "models/scenery/furniture/Sarcophagus/Sarcophagus_KEY.mdl"
    pass


class item_k_shrekhub_one_key(Base):
    model = "models/items/orangecomputing/Ground/orangecomputing.mdl"
    pass


class item_k_shrekhub_four_key(Base):
    model = "models/items/kamikazizen/Ground/kamikazizen.mdl"
    pass


class item_k_shrekhub_three_key(Base):
    model = "models/items/metalhead/Ground/metalhead.mdl"
    pass


class item_k_skyline_haven_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_tatoo_parlor_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_tawni_apartment_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_k_tutorial_chopshop_stairs_key(Base):
    model = "models/items/Keycard/Ground/g_keycard.mdl"
    pass


class item_g_car_stereo(Base):
    model = "models/items/carstereo/Ground/carstereo.mdl"
    pass


class item_g_cash_box(Base):
    model = "models/items/Cashbox/Ground/CashBox.mdl"
    pass


class item_g_chewinggum(Base):
    model = "models/items/ChewingGum/Ground/ChewingGum.mdl"
    pass


class item_g_computerbookhighgrade(Base):
    model = "models/items/ComputerBookHighGrade/Ground/ComputerBookHighGrade.mdl"
    pass


class item_g_computerbooklowgrade(Base):
    model = "models/items/ComputerBookLowGrade/Ground/ComputerBookLowGrade.mdl"
    pass


class item_g_drugs_drug_box(Base):
    model = "models/items/Drugs/ground/DrugBox.mdl"
    pass


class item_g_drugs_morphine_bottle(Base):
    model = "models/items/Drugs/ground/MorphineBottle.mdl"
    pass


class item_g_drugs_perscription_bottle(Base):
    model = "models/items/Drugs/ground/PerscriptionBottle.mdl"
    pass


class item_g_drugs_pill_bottle(Base):
    model = "models/items/Drugs/ground/PillBottle.mdl"
    pass


class item_g_linedpaper(Base):
    model = "models/items/LinedPaper/Ground/LinedPaper.mdl"
    pass


class item_g_pulltoy(Base):
    model = "models/items/PullToy/Ground/PullToy.mdl"
    pass


class item_g_ring_gold(Base):
    model = "models/items/Rings/Ground/Ring01.mdl"
    pass


class item_g_ring_serial_killer_1(Base):
    model = "models/items/Rings/Ground/Ring01.mdl"
    pass


class item_g_ring_serial_killer_2(Base):
    model = "models/items/Rings/Ground/Ring01.mdl"
    pass


class item_g_ring_silver(Base):
    model = "models/items/Rings/Ground/Ring02.mdl"
    pass


class item_g_ring03(Base):
    model = "models/items/Rings/Ground/Ring03.mdl"
    pass


class item_g_watch_fancy(Base):
    model = "models/items/Watch/Ground/Watch.mdl"
    pass


class item_g_watch_normal(Base):
    model = "models/items/Watch/Ground/Watch_02.mdl"
    pass


class item_m_money_clip(Base):
    model = "models/items/MoneyClip/Ground/MoneyClip.mdl"

    @property
    def starting_money(self):
        return parse_source_value(self._raw_data.get('starting_money', 100))



class item_m_money_envelope(Base):
    model = "models/items/MoneyEnvelope/Ground/MoneyEnvelope.mdl"

    @property
    def starting_money(self):
        return parse_source_value(self._raw_data.get('starting_money', 100))



class item_m_wallet(Base):
    model = "models/items/Wallet/Ground/Wallet.mdl"

    @property
    def starting_money(self):
        return parse_source_value(self._raw_data.get('starting_money', 100))



class item_g_locket(Base):
    model = "models/items/locket/Ground/locket.mdl"
    pass


class item_g_garys_photo(Base):
    model = "models/items/garysphoto/Ground/garysphoto.mdl"
    pass


class item_g_garys_cd(Base):
    model = "models/items/cdcase/Ground/cdcase.mdl"
    pass


class item_g_garys_film(Base):
    model = "models/items/HorrorTape_2/Ground/HorrorTape_2.mdl"
    pass


class item_g_garys_tape(Base):
    model = "models/items/HorrorTape_2/Ground/HorrorTape_2.mdl"
    pass


class item_g_stake(Base):
    model = "models/items/stake/Ground/stake.mdl"
    pass


class item_g_vv_photo(Base):
    model = "models/items/LillyOnBeachPhoto/Ground/VV_Photo.mdl"
    pass


class item_g_vampyr_apocrypha(Base):
    model = "models/items/Diary/Ground/Diary.mdl"
    pass


class item_g_warr_clipboard(Base):
    model = "models/items/Clipboard/Ground/Clipboard.mdl"
    pass


class item_g_warr_ledger_1(Base):
    model = "models/items/dayplanner/ground/dayplanner.mdl"
    pass


class item_g_warr_ledger_2(Base):
    model = "models/items/dayplanner/ground/dayplanner2.mdl"
    pass


class item_p_gargoyle_talisman(Base):
    model = "models/items/occult_gargoyle/ground/pendant.mdl"
    pass


class item_p_occult_blood_buff(Base):
    model = "models/items/occult/ground/organ.mdl"
    pass


class item_p_occult_dexterity(Base):
    model = "models/items/occult/ground/birdskull.mdl"
    pass


class item_p_occult_dodge(Base):
    model = "models/items/occult/ground/thorn.mdl"
    pass


class item_p_occult_experience(Base):
    model = "models/items/occult/ground/handle.mdl"
    pass


class item_p_occult_frenzy(Base):
    model = "models/items/occult_Fang/ground/fang.mdl"
    pass


class item_p_occult_hacking(Base):
    model = "models/weapons/lockpicks/world/w_lockpicks.mdl"
    pass


class item_p_occult_heal_rate(Base):
    model = "models/items/occult/ground/bone.mdl"
    pass


class item_p_occult_lockpicking(Base):
    model = "models/weapons/lockpicks/world/w_lockpicks.mdl"
    pass


class item_p_occult_obfuscate(Base):
    model = "models/items/occult/ground/ring.mdl"
    pass


class item_p_occult_passive_durations(Base):
    model = "models/items/occult/ground/amber.mdl"
    pass


class item_p_occult_presence(Base):
    model = "models/items/occult_alamut/ground/alamut.mdl"
    pass


class item_p_occult_regen(Base):
    model = "models/items/occult_chalice/ground/chalice.mdl"
    pass


class item_p_occult_strength(Base):
    model = "models/items/occult/ground/marble.mdl"
    pass


class item_p_occult_thaum_damage(Base):
    model = "models/items/occult/ground/pendant.mdl"
    pass


class item_p_research_hg_computers(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_hg_dodge(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_hg_firearms(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_hg_melee(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_lg_computers(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_lg_dodge(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_lg_firearms(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_lg_stealth(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_mg_brawl(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_mg_finance(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_mg_melee(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_p_research_mg_security(Base):
    model = "models/items/beckett_book/ground/item_g_beckett_book.mdl"
    pass


class item_i_written(Base):
    pass


class item_s_physicshand(Base):
    pass


class item_w_baseball_bat(Base):
    model = "models/weapons/baseball_bat/ground/g_baseball_bat.mdl"
    pass


class item_w_baton(Base):
    model = "models/weapons/baton/ground/g_baton.mdl"
    pass


class item_w_bush_hook(Base):
    model = "models/weapons/bushhook/world/g_bushhook.mdl"
    pass


class item_w_colt_anaconda(Base):
    model = "models/weapons/anaconda/ground/g_anaconda.mdl"
    pass


class item_w_crossbow(Base):
    model = "models/weapons/crossbow/ground/g_crossbow.mdl"
    pass


class item_w_crossbow_flaming(Base):
    model = "models/weapons/crossbow/ground/g_crossbow.mdl"
    pass


class item_w_deserteagle(Base):
    model = "models/weapons/desert_eagle/ground/g_desert_eagle.mdl"
    pass


class item_w_fireaxe(Base):
    model = "models/weapons/fire_axe/ground/g_fire_axe.mdl"
    pass


class item_w_flamethrower(Base):
    model = "models/weapons/flamethrower/ground/g_flamethrower_r.mdl"
    pass


class item_w_glock_17c(Base):
    model = "models/weapons/pistol_glock/ground/g_pistol_glock.mdl"
    pass


class item_w_grenade_frag(Base):
    model = "models/weapons/grenade/pineapple/ground/g_pineapple.mdl"
    pass


class item_w_ithaca_m_37(Base):
    model = "models/weapons/m37/world/w_m37.mdl"
    pass


class item_w_katana(Base):
    model = "models/weapons/katana/world/g_katana.mdl"
    pass


class item_w_knife(Base):
    model = "models/weapons/knife/world/g_knife.mdl"
    pass


class item_w_mac_10(Base):
    model = "models/weapons/submachine_mac10/ground/g_submachine_mac10.mdl"
    pass


class item_w_occultblade(Base):
    model = "models/items/occult_katana/ground/katana.mdl"
    pass


class item_w_remington_m_700(Base):
    model = "models/weapons/rifle_rem700/ground/g_rifle_rem700.mdl"
    pass


class item_w_severed_arm(Base):
    model = "models/weapons/severed_arm/ground/g_severed_arm.mdl"
    pass


class item_w_sheriff_sword(Base):
    model = "models/weapons/katana/world/g_katana.mdl"
    pass


class item_w_sledgehammer(Base):
    model = "models/weapons/sledgehammer/ground/g_sledgehammer.mdl"
    pass


class item_w_steyr_aug(Base):
    model = "models/weapons/rifle_steyraug/ground/g_rifle_steyraug_r.mdl"
    pass


class item_w_supershotgun(Base):
    model = "models/weapons/supershotgun/ground/g_supershotgun.mdl"
    pass


class item_w_thirtyeight(Base):
    model = "models/weapons/ThirtyEight/ground/g_ThirtyEight.mdl"
    pass


class item_w_throwing_star(Base):
    model = "models/weapons/throwing_star/ground/g_throwing_star.mdl"
    pass


class item_w_tire_iron(Base):
    model = "models/weapons/tireiron/world/g_tireiron.mdl"
    pass


class item_w_torch(Base):
    model = "models/weapons/torch/ground/g_torch.mdl"
    pass


class item_w_uzi(Base):
    model = "models/weapons/submachine_uzi/ground/g_uzi.mdl"
    pass


class item_w_chang_blade(Base):
    pass


class item_w_chang_claw(Base):
    pass


class item_w_claws(Base):
    pass


class item_w_claws_ghoul(Base):
    pass


class item_w_claws_protean4(Base):
    pass


class item_w_claws_protean5(Base):
    pass


class item_w_gargoyle_fist(Base):
    pass


class item_w_hengeyokai_fist(Base):
    pass


class item_w_manbat_claw(Base):
    pass


class item_w_mingxiao_melee(Base):
    pass


class item_w_mingxiao_spit(Base):
    pass


class item_w_mingxiao_tentacle(Base):
    pass


class item_w_sabbatleader_attack(Base):
    pass


class item_w_tzimisce_melee(Base):
    pass


class item_w_tzimisce2_claw(Base):
    pass


class item_w_tzimisce2_head(Base):
    pass


class item_w_tzimisce3_claw(Base):
    pass


class item_w_unarmed(Base):
    pass


class item_w_werewolf_attacks(Base):
    pass


class item_w_wolf_head(Base):
    pass


class item_w_avamp_blade(Base):
    model = "models/weapons/katana/world/g_katana.mdl"
    pass


class item_w_chang_energy_ball(Base):
    model = "models/scenery/misc/changball/changball.mdl"
    pass


class item_w_chang_ghost(Base):
    model = "models/scenery/misc/gio_spirit/gio_spirit.mdl"
    pass


class item_w_fists(Base):
    model = "models/weapons/fists/info/i_fists.mdl"
    pass


class item_w_rem_m_700_bach(Base):
    model = "models/weapons/rifle_rem700/ground/g_rifle_rem700_bach.mdl"
    pass


class item_w_zombie_fists(Base):
    model = "models/weapons/fists/info/i_fists.mdl"
    pass


class ambient_soundscheme(c_VEnableDisable, Targetname):
    icon_sprite = "editor/env_soundscape.vmt"
    @property
    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))

    @property
    def sep_ascmain(self):
        return self._raw_data.get('sep_ascmain', None)

    @property
    def scheme_file(self):
        return self._raw_data.get('scheme_file', "sm_hub_streets")




entity_class_handle = {
    'c_VEnableDisable': c_VEnableDisable,
    'VItemContainerBase1': VItemContainerBase1,
    'VItemContainerEquip': VItemContainerEquip,
    'VItemContainerBase2': VItemContainerBase2,
    'item_container': item_container,
    'item_container_animated': item_container_animated,
    'item_container_one_item_filtered': item_container_one_item_filtered,
    'item_a_body_armor': item_a_body_armor,
    'item_a_hvy_cloth': item_a_hvy_cloth,
    'item_a_hvy_leather': item_a_hvy_leather,
    'item_a_lt_cloth': item_a_lt_cloth,
    'item_d_animalism': item_d_animalism,
    'item_d_dementation': item_d_dementation,
    'item_d_dominate': item_d_dominate,
    'item_d_holy_light': item_d_holy_light,
    'item_d_thaumaturgy': item_d_thaumaturgy,
    'item_g_bloodpack': item_g_bloodpack,
    'item_g_bluebloodpack': item_g_bluebloodpack,
    'item_g_eldervitaepack': item_g_eldervitaepack,
    'item_g_keyring': item_g_keyring,
    'item_g_lockpick': item_g_lockpick,
    'item_g_wallet': item_g_wallet,
    'item_g_sewerbook_1': item_g_sewerbook_1,
    'item_g_animaltrainingbook': item_g_animaltrainingbook,
    'item_g_astrolite': item_g_astrolite,
    'item_g_bach_journal': item_g_bach_journal,
    'item_g_badlucktalisman': item_g_badlucktalisman,
    'item_g_bailbond_receipt': item_g_bailbond_receipt,
    'item_g_bertrams_cd': item_g_bertrams_cd,
    'item_g_brotherhood_flyer': item_g_brotherhood_flyer,
    'item_g_driver_license_gimble': item_g_driver_license_gimble,
    'item_g_edane_print_report': item_g_edane_print_report,
    'item_g_edane_report': item_g_edane_report,
    'item_g_eyes': item_g_eyes,
    'item_g_gargoyle_book': item_g_gargoyle_book,
    'item_g_ghost_pendant': item_g_ghost_pendant,
    'item_g_giovanni_invitation_maria': item_g_giovanni_invitation_maria,
    'item_g_giovanni_invitation_victor': item_g_giovanni_invitation_victor,
    'item_g_guy_magazine': item_g_guy_magazine,
    'item_g_hannahs_appt_book': item_g_hannahs_appt_book,
    'item_g_hatters_screenplay': item_g_hatters_screenplay,
    'item_g_horrortape_1': item_g_horrortape_1,
    'item_g_horrortape_2': item_g_horrortape_2,
    'item_g_idol_cat': item_g_idol_cat,
    'item_g_idol_crane': item_g_idol_crane,
    'item_g_idol_dragon': item_g_idol_dragon,
    'item_g_idol_elephant': item_g_idol_elephant,
    'item_g_jumbles_flyer': item_g_jumbles_flyer,
    'item_g_junkyard_businesscard': item_g_junkyard_businesscard,
    'item_g_larry_briefcase': item_g_larry_briefcase,
    'item_g_lilly_diary': item_g_lilly_diary,
    'item_g_lilly_photo': item_g_lilly_photo,
    'item_g_lilly_purse': item_g_lilly_purse,
    'item_g_lillyonbeachphoto': item_g_lillyonbeachphoto,
    'item_g_mercurio_journal': item_g_mercurio_journal,
    'item_g_milligans_businesscard': item_g_milligans_businesscard,
    'item_g_oh_diary': item_g_oh_diary,
    'item_g_pisha_book': item_g_pisha_book,
    'item_g_pisha_fetish': item_g_pisha_fetish,
    'item_g_werewolf_bloodpack': item_g_werewolf_bloodpack,
    'item_g_wireless_camera_1': item_g_wireless_camera_1,
    'item_g_wireless_camera_2': item_g_wireless_camera_2,
    'item_g_wireless_camera_3': item_g_wireless_camera_3,
    'item_g_wireless_camera_4': item_g_wireless_camera_4,
    'item_g_warrens4_passkey': item_g_warrens4_passkey,
    'item_k_ash_cell_key': item_k_ash_cell_key,
    'item_k_carson_apartment_key': item_k_carson_apartment_key,
    'item_k_chinese_theatre_key': item_k_chinese_theatre_key,
    'item_k_clinic_cs_key': item_k_clinic_cs_key,
    'item_k_clinic_maintenance_key': item_k_clinic_maintenance_key,
    'item_k_clinic_stairs_key': item_k_clinic_stairs_key,
    'item_k_edane_key': item_k_edane_key,
    'item_k_empire_jezebel_key': item_k_empire_jezebel_key,
    'item_k_empire_mafia_key': item_k_empire_mafia_key,
    'item_k_fu_cell_key': item_k_fu_cell_key,
    'item_k_fu_office_key': item_k_fu_office_key,
    'item_k_gallery_noir_key': item_k_gallery_noir_key,
    'item_k_gimble_key': item_k_gimble_key,
    'item_k_hannahs_safe_key': item_k_hannahs_safe_key,
    'item_k_hitman_ji_key': item_k_hitman_ji_key,
    'item_k_hitman_lu_key': item_k_hitman_lu_key,
    'item_k_kiki_key': item_k_kiki_key,
    'item_k_leopold_int_key': item_k_leopold_int_key,
    'item_k_lilly_key': item_k_lilly_key,
    'item_k_lucky_star_murder_key': item_k_lucky_star_murder_key,
    'item_k_malcolm_office_key': item_k_malcolm_office_key,
    'item_k_malkavian_refrigerator_key': item_k_malkavian_refrigerator_key,
    'item_k_murietta_key': item_k_murietta_key,
    'item_k_museum_basement_key': item_k_museum_basement_key,
    'item_k_museum_office_key': item_k_museum_office_key,
    'item_k_museum_storage_key': item_k_museum_storage_key,
    'item_k_museum_storeroom_key': item_k_museum_storeroom_key,
    'item_k_netcafe_office_key': item_k_netcafe_office_key,
    'item_k_oceanhouse_basement_key': item_k_oceanhouse_basement_key,
    'item_k_oceanhouse_sewer_key': item_k_oceanhouse_sewer_key,
    'item_k_oceanhouse_upstairs_key': item_k_oceanhouse_upstairs_key,
    'item_k_oh_front_key': item_k_oh_front_key,
    'item_k_sarcophagus_key': item_k_sarcophagus_key,
    'item_k_shrekhub_one_key': item_k_shrekhub_one_key,
    'item_k_shrekhub_four_key': item_k_shrekhub_four_key,
    'item_k_shrekhub_three_key': item_k_shrekhub_three_key,
    'item_k_skyline_haven_key': item_k_skyline_haven_key,
    'item_k_tatoo_parlor_key': item_k_tatoo_parlor_key,
    'item_k_tawni_apartment_key': item_k_tawni_apartment_key,
    'item_k_tutorial_chopshop_stairs_key': item_k_tutorial_chopshop_stairs_key,
    'item_g_car_stereo': item_g_car_stereo,
    'item_g_cash_box': item_g_cash_box,
    'item_g_chewinggum': item_g_chewinggum,
    'item_g_computerbookhighgrade': item_g_computerbookhighgrade,
    'item_g_computerbooklowgrade': item_g_computerbooklowgrade,
    'item_g_drugs_drug_box': item_g_drugs_drug_box,
    'item_g_drugs_morphine_bottle': item_g_drugs_morphine_bottle,
    'item_g_drugs_perscription_bottle': item_g_drugs_perscription_bottle,
    'item_g_drugs_pill_bottle': item_g_drugs_pill_bottle,
    'item_g_linedpaper': item_g_linedpaper,
    'item_g_pulltoy': item_g_pulltoy,
    'item_g_ring_gold': item_g_ring_gold,
    'item_g_ring_serial_killer_1': item_g_ring_serial_killer_1,
    'item_g_ring_serial_killer_2': item_g_ring_serial_killer_2,
    'item_g_ring_silver': item_g_ring_silver,
    'item_g_ring03': item_g_ring03,
    'item_g_watch_fancy': item_g_watch_fancy,
    'item_g_watch_normal': item_g_watch_normal,
    'item_m_money_clip': item_m_money_clip,
    'item_m_money_envelope': item_m_money_envelope,
    'item_m_wallet': item_m_wallet,
    'item_g_locket': item_g_locket,
    'item_g_garys_photo': item_g_garys_photo,
    'item_g_garys_cd': item_g_garys_cd,
    'item_g_garys_film': item_g_garys_film,
    'item_g_garys_tape': item_g_garys_tape,
    'item_g_stake': item_g_stake,
    'item_g_vv_photo': item_g_vv_photo,
    'item_g_vampyr_apocrypha': item_g_vampyr_apocrypha,
    'item_g_warr_clipboard': item_g_warr_clipboard,
    'item_g_warr_ledger_1': item_g_warr_ledger_1,
    'item_g_warr_ledger_2': item_g_warr_ledger_2,
    'item_p_gargoyle_talisman': item_p_gargoyle_talisman,
    'item_p_occult_blood_buff': item_p_occult_blood_buff,
    'item_p_occult_dexterity': item_p_occult_dexterity,
    'item_p_occult_dodge': item_p_occult_dodge,
    'item_p_occult_experience': item_p_occult_experience,
    'item_p_occult_frenzy': item_p_occult_frenzy,
    'item_p_occult_hacking': item_p_occult_hacking,
    'item_p_occult_heal_rate': item_p_occult_heal_rate,
    'item_p_occult_lockpicking': item_p_occult_lockpicking,
    'item_p_occult_obfuscate': item_p_occult_obfuscate,
    'item_p_occult_passive_durations': item_p_occult_passive_durations,
    'item_p_occult_presence': item_p_occult_presence,
    'item_p_occult_regen': item_p_occult_regen,
    'item_p_occult_strength': item_p_occult_strength,
    'item_p_occult_thaum_damage': item_p_occult_thaum_damage,
    'item_p_research_hg_computers': item_p_research_hg_computers,
    'item_p_research_hg_dodge': item_p_research_hg_dodge,
    'item_p_research_hg_firearms': item_p_research_hg_firearms,
    'item_p_research_hg_melee': item_p_research_hg_melee,
    'item_p_research_lg_computers': item_p_research_lg_computers,
    'item_p_research_lg_dodge': item_p_research_lg_dodge,
    'item_p_research_lg_firearms': item_p_research_lg_firearms,
    'item_p_research_lg_stealth': item_p_research_lg_stealth,
    'item_p_research_mg_brawl': item_p_research_mg_brawl,
    'item_p_research_mg_finance': item_p_research_mg_finance,
    'item_p_research_mg_melee': item_p_research_mg_melee,
    'item_p_research_mg_security': item_p_research_mg_security,
    'item_i_written': item_i_written,
    'item_s_physicshand': item_s_physicshand,
    'item_w_baseball_bat': item_w_baseball_bat,
    'item_w_baton': item_w_baton,
    'item_w_bush_hook': item_w_bush_hook,
    'item_w_colt_anaconda': item_w_colt_anaconda,
    'item_w_crossbow': item_w_crossbow,
    'item_w_crossbow_flaming': item_w_crossbow_flaming,
    'item_w_deserteagle': item_w_deserteagle,
    'item_w_fireaxe': item_w_fireaxe,
    'item_w_flamethrower': item_w_flamethrower,
    'item_w_glock_17c': item_w_glock_17c,
    'item_w_grenade_frag': item_w_grenade_frag,
    'item_w_ithaca_m_37': item_w_ithaca_m_37,
    'item_w_katana': item_w_katana,
    'item_w_knife': item_w_knife,
    'item_w_mac_10': item_w_mac_10,
    'item_w_occultblade': item_w_occultblade,
    'item_w_remington_m_700': item_w_remington_m_700,
    'item_w_severed_arm': item_w_severed_arm,
    'item_w_sheriff_sword': item_w_sheriff_sword,
    'item_w_sledgehammer': item_w_sledgehammer,
    'item_w_steyr_aug': item_w_steyr_aug,
    'item_w_supershotgun': item_w_supershotgun,
    'item_w_thirtyeight': item_w_thirtyeight,
    'item_w_throwing_star': item_w_throwing_star,
    'item_w_tire_iron': item_w_tire_iron,
    'item_w_torch': item_w_torch,
    'item_w_uzi': item_w_uzi,
    'item_w_chang_blade': item_w_chang_blade,
    'item_w_chang_claw': item_w_chang_claw,
    'item_w_claws': item_w_claws,
    'item_w_claws_ghoul': item_w_claws_ghoul,
    'item_w_claws_protean4': item_w_claws_protean4,
    'item_w_claws_protean5': item_w_claws_protean5,
    'item_w_gargoyle_fist': item_w_gargoyle_fist,
    'item_w_hengeyokai_fist': item_w_hengeyokai_fist,
    'item_w_manbat_claw': item_w_manbat_claw,
    'item_w_mingxiao_melee': item_w_mingxiao_melee,
    'item_w_mingxiao_spit': item_w_mingxiao_spit,
    'item_w_mingxiao_tentacle': item_w_mingxiao_tentacle,
    'item_w_sabbatleader_attack': item_w_sabbatleader_attack,
    'item_w_tzimisce_melee': item_w_tzimisce_melee,
    'item_w_tzimisce2_claw': item_w_tzimisce2_claw,
    'item_w_tzimisce2_head': item_w_tzimisce2_head,
    'item_w_tzimisce3_claw': item_w_tzimisce3_claw,
    'item_w_unarmed': item_w_unarmed,
    'item_w_werewolf_attacks': item_w_werewolf_attacks,
    'item_w_wolf_head': item_w_wolf_head,
    'item_w_avamp_blade': item_w_avamp_blade,
    'item_w_chang_energy_ball': item_w_chang_energy_ball,
    'item_w_chang_ghost': item_w_chang_ghost,
    'item_w_fists': item_w_fists,
    'item_w_rem_m_700_bach': item_w_rem_m_700_bach,
    'item_w_zombie_fists': item_w_zombie_fists,
    'ambient_soundscheme': ambient_soundscheme,
}