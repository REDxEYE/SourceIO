from SourceIO.library.utils.kv1 import loads


def test_kv1_duplicated_keys():
    data = """FileSystem
{
	SearchPaths
	{
		Game_LowViolence	csgo_lv // Perfect World content override

		Game	csgo
		Game	csgo_imported
		Game	csgo_core
		Game	core

		Mod		csgo
		Mod		csgo_imported
		Mod		csgo_core

		AddonRoot			csgo_addons
		OfficialAddonRoot	csgo_community_addons

		LayeredGameRoot		"../game_otherplatforms/etc" [$MOBILE || $ETC_TEXTURES] //Some platforms do not support DXT compression. ETC is a well-supported alternative.
		LayeredGameRoot		"../game_otherplatforms/low_bitrate" [$MOBILE]
	}

	"UserSettingsPathID"	"USRLOCAL"
	"UserSettingsFileEx"	"cs2_"
}"""

    root = loads(data, 'cs2 gameinfo')
    search_paths = root["FileSystem"]["SearchPaths"]

    # `get` returns the first match, like KeyValues::FindKey
    assert search_paths["Game"] == "csgo"
    assert search_paths.get_multiple("game") == ["csgo", "csgo_imported", "csgo_core",
                                                 "core"]
    assert search_paths.get_all("mod") == ["csgo", "csgo_imported", "csgo_core"]

    # ordered, duplicates preserved -- mount precedence depends on it
    keys = [key for key, _ in search_paths.items()]
    assert keys.count("game") == 4
    assert keys.index("game_lowviolence") < keys.index("game")

    # both LayeredGameRoot entries are gated on $MOBILE, which is false on PC
    assert "layeredgameroot" not in search_paths

    assert root["FileSystem"]["UserSettingsPathID"] == "USRLOCAL"
