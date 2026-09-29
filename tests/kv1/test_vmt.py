"""VMT-level semantics layered on top of the KV1 grammar.

``[$WIN32]`` suffixes are grammar and the parser handles them. These two are material
system conventions and live in :mod:`SourceIO.library.source1.vmt`.
"""
from SourceIO.library.source1.vmt import VMT
from SourceIO.library.utils import MemoryBuffer


class DummyCM:
    def find_file(self, *args, **kwargs):
        return None


def make(text: str) -> VMT:
    return VMT(MemoryBuffer(text.encode()), 'test.vmt', DummyCM())


def test_shader_name_and_basic_params():
    vmt = make('"LightmappedGeneric"\n{\n"$basetexture" "brick/wall"\n}')
    assert vmt.shader == 'lightmappedgeneric'
    assert vmt.get_string('$basetexture') == 'brick/wall'


def test_unparseable_material_reports_failed_to_load():
    assert make('').shader == 'FAILED_TO_LOAD'


# ------------------------------------------------------- condition-named blocks

def test_dx90_block_is_promoted_and_overrides():
    vmt = make('''Shader
{
    "$basetexture" "low"
    ">=dx90"
    {
        "$basetexture" "high"
        "$bumpmap" "high_n"
    }
}''')
    assert vmt.get_string('$basetexture') == 'high'
    assert vmt.get_string('$bumpmap') == 'high_n'
    assert '>=dx90' not in vmt


def test_false_condition_block_is_dropped():
    vmt = make('Shader\n{\n"$a" "1"\n"<dx90"\n{\n"$a" "2"\n}\n}')
    assert vmt.get_string('$a') == '1'
    assert '<dx90' not in vmt


def test_ordinary_subblock_is_never_treated_as_a_condition():
    """`proxies` is not a condition; evaluating it would resolve false and delete it."""
    vmt = make('Shader\n{\n"proxies" { "sine" { "resultvar" "$alpha" } }\n}')
    assert 'proxies' in vmt
    assert vmt['proxies']['sine']['resultvar'] == '$alpha'


# ------------------------------------------------------ `<condition>?<key>` params

def test_prefixed_condition_applies_when_true():
    # shipped 111 times: PC-only phong
    vmt = make('Shader\n{\n"$phong" "0"\n"!gameconsole?$phong" "1"\n}')
    assert vmt.get_string('$phong') == '1'


def test_prefixed_condition_dropped_when_false():
    # shipped 457 times: Xbox 360 only
    vmt = make('Shader\n{\n"360?$color2" "[0.78 0.8 0.8]"\n}')
    assert '$color2' not in vmt


def test_prefixed_condition_picks_one_of_a_pair():
    vmt = make('Shader\n{\n"srgb?$overbrightfactor" "2"\n'
               '"!srgb?$overbrightfactor" "8"\n}')
    assert vmt.get_string('$overbrightfactor') == '2'


def test_prefixed_gpu_conditions():
    vmt = make('Shader\n{\n"gpu<1?$envmap" "env_cubemap"\n'
               '"gpu>=1?$reflecttexture" "_rt_WaterReflection"\n}')
    assert '$envmap' not in vmt
    assert vmt.get_string('$reflecttexture') == '_rt_WaterReflection'


def test_unrecognised_prefix_is_left_alone():
    """A key containing '?' whose prefix is not a known symbol must survive intact.

    Treating it as a condition would resolve the unknown symbol to false and delete
    the key.
    """
    vmt = make('Shader\n{\n"what?ever" "kept"\n}')
    assert vmt.get_string('what?ever') == 'kept'


def test_prefixed_conditions_inside_nested_blocks():
    vmt = make('Shader\n{\n"sub" { "360?$a" "1"\n"!gameconsole?$b" "2" }\n}')
    sub = vmt['sub']
    assert '$a' not in sub
    assert sub['$b'] == '2'


# ------------------------------------------------------------------- accessors

def test_get_vector_tolerates_padded_values():
    # shipped: TF2's sniper_lens writes `"$envmaptint" " [1 .8 .4]"`
    vmt = make('Shader\n{\n"$envmaptint" " [1 .8 .4]"\n}')
    values, kind = vmt.get_vector('$envmaptint')
    assert values == (1.0, 0.8, 0.4)
    assert kind is float


def test_get_vector_brace_syntax_reports_int_range():
    values, kind = make('Shader\n{\n"$color" "{255 128 0}"\n}').get_vector('$color')
    assert values == (255, 128, 0)
    assert kind is int


def test_get_vector_unquoted_folded_value():
    # shipped: `"$color2" .25 .25 .25`
    values, _ = make('Shader\n{\n"$color2" .25 .25 .25\n}').get_vector('$color2')
    assert values == (0.25, 0.25, 0.25)


def test_backslash_paths_reach_the_consumer_intact():
    vmt = make(r'Shader{"$basetexture" "models\props\crate"}')
    # kept verbatim; TinyPath normalises separators where a path is actually used
    from SourceIO.library.utils import TinyPath
    assert TinyPath(vmt.get_string('$basetexture')).parts == ['models', 'props', 'crate']


def test_numeric_accessors_tolerate_typos():
    vmt = make('Shader\n{\n"$detailblendfactor" ".4`"\n"$alpha" ".5 .5 .5"\n}')
    assert vmt.get_float('$detailblendfactor') == 0.4
    assert vmt.get_float('$alpha') == 0.5


def test_duplicate_keys_first_wins_like_findkey():
    vmt = make('Shader\n{\n"$basetexture" "first"\n"$basetexture" "second"\n}')
    assert vmt.get_string('$basetexture') == 'first'
