"""Tests for the KV1 parser.

Cases marked "shipped" were found by surveying 47,604 real files (Black Mesa, HL2,
Portal 2, CS:S and TF2 materials plus gameinfo/vdf/acf), not invented.
"""
import pytest

from SourceIO.library.utils.kv1 import (ConditionContext, KV1Block, Severity, dump,
                                        dumps, load_bytes, loads, loads_blocks,
                                        parse_vector)


def errors(block: KV1Block):
    return [d for d in block.diagnostics if d.severity is Severity.ERROR]


# --------------------------------------------------------------------------- basics

def test_simple_pairs():
    block = loads('''"LightmappedGeneric"
{
    "$basetexture" "brick/brickwall001a"
    "$surfaceprop" "brick"
}''')
    shader, body = block.top()
    assert shader == 'lightmappedgeneric'
    assert body['$basetexture'] == 'brick/brickwall001a'
    assert body.get_string('$surfaceprop') == 'brick'
    assert not errors(block)


def test_keys_are_lowercased_values_are_not():
    body = loads('X { "$BaseTexture" "Brick/BrickWall001A" }').top()[1]
    assert body['$basetexture'] == 'Brick/BrickWall001A'


def test_unquoted_keys_and_values():
    body = loads('Shader { $basetexture brick/wall\n$model 1 }').top()[1]
    assert body['$basetexture'] == 'brick/wall'
    assert body['$model'] == '1'


def test_nested_blocks():
    root = loads('''"gameinfo"
{
    "filesystem"
    {
        "steamappid" "220"
        "searchpaths" { "game" "hl2" }
    }
}''')
    fs = root['gameinfo']['filesystem']
    assert fs.get_int('steamappid') == 220
    assert fs['searchpaths']['game'] == 'hl2'


def test_brace_on_same_line_as_key():
    # shipped: 391 lines put the opening brace beside the key
    root = loads('Shader { $a 1 }')
    assert root.top()[1]['$a'] == '1'


def test_comments():
    body = loads('''S {
    // whole line
    $a "1" // trailing
    /* block
       comment */
    $b "2"
    \\\\ $c "3"
}''').top()[1]
    assert body['$a'] == '1'
    assert body['$b'] == '2'
    # `\\` is used as a comment marker by some authoring tools (shipped)
    assert '$c' not in body


# ------------------------------------------------------------------------ multi-key

def test_duplicate_keys_preserved_in_order():
    # gameinfo SearchPaths relies on this
    paths = loads('''FileSystem { SearchPaths {
        Game csgo
        Game csgo_imported
        Game csgo_core
    } }''')['filesystem']['searchpaths']
    assert paths['game'] == 'csgo'                      # first, like FindKey
    assert paths.get_all('game') == ['csgo', 'csgo_imported', 'csgo_core']
    assert [k for k, _ in paths.items()] == ['game', 'game', 'game']
    assert len(paths) == 3


def test_plus_key_expands_to_several_keys():
    # shipped: game+mod, mod+mod_write+default_write_path
    paths = loads('F { S { game+mod hl2\n mod+mod_write+default_write_path ep2 } }'
                  )['f']['s']
    assert paths['game'] == 'hl2'
    assert paths['mod'] == 'hl2'
    assert paths.get_all('mod') == ['hl2', 'ep2']
    assert paths['default_write_path'] == 'ep2'


def test_to_dict_collapses_duplicates_into_list():
    data = loads('S { game a\n game b\n mod c }')['s'].to_dict()
    assert data == {'game': ['a', 'b'], 'mod': 'c'}


# ---------------------------------------------------------------------- multi-value

def test_unquoted_vector_is_folded():
    # shipped: "$color2" .25 .25 .25
    body = loads('S { "$color2" .25 .25 .25 }').top()[1]
    assert body['$color2'] == '.25 .25 .25'
    assert body.get_vector('$color2') == (0.25, 0.25, 0.25)


def test_two_pairs_on_one_line_are_not_folded():
    # shipped: $model 1 "$basetexture" "tools/toolsblack"
    body = loads('S { $model 1 \t"$basetexture" "tools/toolsblack" }').top()[1]
    assert body['$model'] == '1'
    assert body['$basetexture'] == 'tools/toolsblack'


def test_folding_stops_at_end_of_line():
    body = loads('S {\n $a 1 2\n $b 3\n}').top()[1]
    assert body['$a'] == '1 2'
    assert body['$b'] == '3'


def test_bracketed_vector_value():
    # shipped: $color2 [1 1 1] unquoted, and "[.04 .048 .032]" quoted
    body = loads('S { $color2 [1 1 1]\n "$envmaptint" "[.04 .048 .032]" }').top()[1]
    assert body['$color2'] == '[1 1 1]'
    assert body.get_vector('$color2') == (1.0, 1.0, 1.0)
    assert body.get_vector('$envmaptint') == pytest.approx((0.04, 0.048, 0.032))


def test_brace_vector_is_0_255_syntax():
    from SourceIO.library.utils.kv1 import is_integer_syntax
    body = loads('S { "$color" "{255 128 0}" }').top()[1]
    assert body.get_vector('$color') == (255.0, 128.0, 0.0)
    assert is_integer_syntax(body['$color'])
    assert not is_integer_syntax('[1 1 1]')


# ----------------------------------------------------------------------- conditions

def test_condition_true_keeps_key():
    body = loads('S { "$a" "1" [$WIN32] }').top()[1]
    assert body['$a'] == '1'


def test_condition_false_drops_key():
    body = loads('S { "$a" "1" [$X360] }').top()[1]
    assert '$a' not in body


def test_negated_condition():
    # shipped: [!$ps3]
    body = loads('S { "$a" "1" [!$ps3]\n "$b" "2" [!$win32] }').top()[1]
    assert body['$a'] == '1'
    assert '$b' not in body


def test_compound_conditions():
    ctx = ConditionContext({'$one': True, '$zero': False})
    body = loads('''S {
        a 1 [$one || $zero]
        b 2 [$one && $zero]
        c 3 [$zero || $zero]
        d 4 [!$zero && $one]
        e 5 [($zero || $one) && !$zero]
    }''', conditions=ctx).top()[1]
    assert body.get('a') == '1'
    assert body.get('b') is None
    assert body.get('c') is None
    assert body.get('d') == '4'
    assert body.get('e') == '5'


def test_unknown_symbol_is_false_and_recorded():
    ctx = ConditionContext()
    body = loads('S { a 1 [$SOMETHING_NEW] }', conditions=ctx).top()[1]
    assert body.get('a') is None
    assert '$something_new' in ctx.unknown_seen


def test_unparseable_condition_keeps_the_key():
    # discarding data we merely failed to understand is the worse failure
    block = loads('S { a 1 [$win32 &&] }')
    assert block.top()[1]['a'] == '1'
    assert any('could not be evaluated' in d.message for d in errors(block))


def test_dx_level_conditions():
    body = loads('S { a 1 [>=dx90]\n b 2 [<dx90_20b]\n c 3 [gpu>=2] }').top()[1]
    assert body.get('a') == '1'
    assert body.get('b') is None
    assert body.get('c') == '3'


def test_condition_on_block():
    root = loads('S { sub [$X360] { a 1 }\n keep { b 2 } }').top()[1]
    assert 'sub' not in root
    assert root['keep']['b'] == '2'


def test_keep_disabled_retains_everything():
    body = loads('S { a 1 [$X360] }', keep_disabled=True).top()[1]
    assert body['a'] == '1'


def test_custom_condition_context_overrides_defaults():
    body = loads('S { a 1 [$X360] }',
                 conditions={'$X360': True}).top()[1]
    assert body['a'] == '1'


# ------------------------------------------------------------------------ tolerance

def test_unterminated_string_stops_at_end_of_line():
    # shipped: 6 materials have an unmatched quote. One stray quote must not swallow
    # the rest of the file.
    block = loads('S {\n "$a" "unclosed\n "$b" "2"\n}')
    body = block.top()[1]
    assert body['$a'] == 'unclosed'
    assert body['$b'] == '2'
    assert errors(block)


def test_missing_closing_brace_is_recovered():
    block = loads('S {\n "$a" "1"\n')
    assert block.top()[1]['$a'] == '1'
    assert any('missing "}"' in d.message for d in errors(block))


def test_stray_closing_brace_is_ignored():
    block = loads('S { "$a" "1" } }\nT { "$b" "2" }')
    assert block['s']['$a'] == '1'
    assert block['t']['$b'] == '2'
    assert any('stray "}"' in d.message for d in errors(block))


def test_key_without_value_is_dropped():
    block = loads('S { "$a" "1"\n "$dangling" }')
    body = block.top()[1]
    assert body['$a'] == '1'
    assert '$dangling' not in body
    assert any('no value' in d.message for d in errors(block))


def test_garbage_line_does_not_abort_the_rest():
    block = loads('''S {
    "$a" "1"
    $treeSway 2 (0 1 or 2 but 2 works best)
    "$b" "2"
}''')
    body = block.top()[1]
    # shipped: an unbracketed comment. The good keys on either side must survive.
    assert body['$a'] == '1'
    assert body['$b'] == '2'
    assert body['$treesway'] == '2'


def test_never_raises_on_fuzzed_input():
    samples = [
        '', '{', '}', '"', '[', ']', '//', '/*', 'a', 'a {', '{ }', '" "',
        'a b c d e', '[$win32]', 'a [', 'a ]', 'a { b', '}}}}', '{{{{',
        'a "b" [', '#base', '#base "x.vdf"', 'a\x00b "c"', '\x80\x81\x82',
        'a { b { c { d { e "1"', '"a" "b" "c" "d"', 'a 1 2 3 4 5',
    ]
    for sample in samples:
        block = loads(sample, name='fuzz')
        assert isinstance(block, KV1Block)


def test_deeply_nested_input_is_bounded():
    block = loads('a {' * 500 + 'k v' + '}' * 500)
    assert isinstance(block, KV1Block)
    assert any('nesting deeper' in d.message for d in block.diagnostics)


def test_unterminated_block_comment():
    block = loads('S { "$a" "1" /* never closed')
    assert block.top()[1]['$a'] == '1'
    assert any('/* comment' in d.message for d in errors(block))


# -------------------------------------------------------------------------- escapes

def test_backslash_paths_survive_by_default():
    # VMT paths must not be treated as escape sequences
    body = loads(r'S { "$basetexture" "models\props\metal_box" }').top()[1]
    assert body['$basetexture'] == r'models\props\metal_box'


def test_escapes_when_enabled():
    body = loads(r'S { "path" "C:\\Steam\\x" }', escapes=True).top()[1]
    assert body['path'] == r'C:\Steam\x'


# -------------------------------------------------------------------------- decoding

def test_non_utf8_bytes_are_decoded_not_fatal():
    block = load_bytes(b'S { "$a" "caf\xe9" }')
    assert block.top()[1]['$a'].startswith('caf')


def test_utf8_bom_is_stripped():
    block = load_bytes(b'\xef\xbb\xbfS { "$a" "1" }')
    assert block.top()[1]['$a'] == '1'


# ------------------------------------------------------------------- blocks / misc

def test_loads_blocks_for_entity_lump():
    blocks = loads_blocks('''{
"classname" "worldspawn"
"skyname" "sky_day01_01"
}
{
"classname" "info_player_start"
"origin" "0 0 64"
}''')
    assert len(blocks) == 2
    assert blocks[0]['classname'] == 'worldspawn'
    assert blocks[1]['origin'] == '0 0 64'


def test_entity_outputs_are_duplicate_keys():
    """An entity's I/O outputs are repeated keys -- the multi-key case in BSP data."""
    ent = loads_blocks('''{
"classname" "logic_relay"
"targetname" "relay_start"
"OnTrigger" "door_a,Open,,0,-1"
"OnTrigger" "door_b,Open,,0.5,-1"
"OnTrigger" "sound_a,PlaySound,,1,-1"
}''')[0]
    assert ent['classname'] == 'logic_relay'
    assert ent.get_all('ontrigger') == [
        'door_a,Open,,0,-1', 'door_b,Open,,0.5,-1', 'sound_a,PlaySound,,1,-1']
    # to_dict collapses them into a list, which is what the BSP importer consumes
    assert ent.to_dict()['ontrigger'] == [
        'door_a,Open,,0,-1', 'door_b,Open,,0.5,-1', 'sound_a,PlaySound,,1,-1']


def test_entity_value_that_is_a_lone_apostrophe():
    """shipped: Black Mesa's background06 has ``"floating_metal_barrel" "'"``.

    A single quote *inside* double quotes is ordinary data. The old parser treated it
    as a delimiter, desynced, and swallowed the following ``"fadescale" "1"`` line into
    a list -- losing that key entirely.
    """
    ent = loads_blocks('''{
"forcetoenablemotion" "0"
"floating_metal_barrel" "'"
"fadescale" "1"
"classname" "prop_physics"
}''')[0]
    assert ent['floating_metal_barrel'] == "'"
    assert ent['fadescale'] == '1'
    assert ent['classname'] == 'prop_physics'


def test_entity_value_may_contain_a_newline():
    """shipped: gm_br_pitfalls stores "sounds" "vo<LF>pc".

    Almost certainly a mangled ``vo/npc`` sound path, but the bytes really do hold a
    newline, and Valve's reader scans past it. Stopping at the newline salvaged only
    ``vo`` and shifted every remaining pair in the entity by one.
    """
    ent = loads_blocks('{\n"sounds" "vo\npc"\n"rendermode" "0"\n"classname" "func_button"\n}')[0]
    assert ent['sounds'] == 'vo\npc'
    assert ent['rendermode'] == '0'
    assert ent['classname'] == 'func_button'


def test_unmatched_quote_does_not_absorb_the_following_pair():
    """The counterpart: a later quote that *opens a key* must not be taken as closing.

    Both cases put a candidate closing quote on a later line; what follows the quote
    is what tells them apart.
    """
    block = loads('S {\n"$a" "unclosed\n"$b" "2"\n"$c" "3"\n}', 'm')
    body = block.top()[1]
    assert body['$a'] == 'unclosed'
    assert body['$b'] == '2'
    assert body['$c'] == '3'
    assert any('opens a key' in d.message for d in errors(block))


def test_entity_values_keep_backslashes_and_case():
    ent = loads_blocks(r'''{
"classname" "prop_static"
"model" "models\props_c17\FurnitureCouch001a.mdl"
"origin" "-1024 512.5 -63.96875"
}''')[0]
    assert ent['model'] == r'models\props_c17\FurnitureCouch001a.mdl'
    assert ent.get_vector('origin') == (-1024.0, 512.5, -63.96875)


def test_entity_value_with_embedded_quote_does_not_derail_the_lump():
    # user-authored captions occasionally contain a stray quote
    blocks = loads_blocks('''{
"classname" "point_message"
"message" "He said "hello" loudly"
}
{
"classname" "info_player_start"
"origin" "0 0 64"
}''')
    # the damaged entity must not consume the healthy one that follows
    assert len(blocks) == 2
    assert blocks[1]['classname'] == 'info_player_start'
    assert blocks[1]['origin'] == '0 0 64'


def test_entity_lump_with_trailing_nulls_and_blank_lines():
    blocks = loads_blocks('{\n"classname" "worldspawn"\n}\n\n\n')
    assert len(blocks) == 1
    assert blocks[0]['classname'] == 'worldspawn'


def test_entity_lump_unclosed_last_entity_is_recovered():
    blocks = loads_blocks('{\n"classname" "a"\n}\n{\n"classname" "b"\n')
    assert [b['classname'] for b in blocks] == ['a', 'b']


def test_base_directive_is_recorded():
    block = loads('#base "shared.vdf"\nS { a 1 }')
    assert block.includes == ['shared.vdf']
    assert block['s']['a'] == '1'


def test_merge_overlays_recursively():
    base = loads('S { a 1\n sub { x 1\n y 2 } }')['s']
    patch = loads('S { a 9\n sub { y 9 }\n new 3 }')['s']
    base.merge(patch)
    assert base['a'] == '9'
    assert base['sub']['x'] == '1'
    assert base['sub']['y'] == '9'
    assert base['new'] == '3'


def test_typed_accessors_tolerate_junk():
    body = loads('''S {
    "$f" "0.5"
    "$i" "7"
    "$bad" ".4`"
    "$vec" ".5 .5 .5"
    "$bool" "1"
    "$boolword" "true"
    "$empty" ""
}''').top()[1]
    assert body.get_float('$f') == 0.5
    assert body.get_int('$i') == 7
    # shipped typo: a stray backtick
    assert body.get_float('$bad') == 0.4
    assert body.get_float('$vec') == 0.5
    assert body.get_bool('$bool') and body.get_bool('$boolword')
    assert body.get_float('$missing', 3.0) == 3.0
    assert body.get_int('$missing', 5) == 5


def test_parse_vector_variants():
    assert parse_vector('[1 1 1]') == [1.0, 1.0, 1.0]
    assert parse_vector('{255 128 0}') == [255.0, 128.0, 0.0]
    assert parse_vector('.5') == [0.5]
    assert parse_vector('-1 2.5e2') == [-1.0, 250.0]
    assert parse_vector('nonsense') == []
    assert parse_vector('') == []


def test_mutation_and_deletion():
    body = loads('S { a 1\n b 2 }')['s']
    body['a'] = '9'
    assert body['a'] == '9'
    del body['b']
    assert 'b' not in body
    with pytest.raises(KeyError):
        del body['nope']
    body.append('c', '3')
    body.append('c', '4')
    assert body.get_all('c') == ['3', '4']


def test_empty_and_whitespace_documents():
    for text in ('', '   \n\t\n', '// only a comment\n'):
        block = loads(text)
        assert len(block) == 0
        assert block.top() == ('', KV1Block()) or block.top()[0] == ''


# ------------------------------------------------------------- malformed KV pairs
#
# Each case asserts the *specific* recovery, not merely "did not crash": a tolerant
# parser is only useful if what it salvages is predictable.

def body_of(text):
    """Parse one line inside a shader block and return the block."""
    return loads('S {\n' + text + '\n}', 'malformed').top()[1]


def test_malformed_double_bracket_in_quoted_value():
    # $key "[[1 1 1]" -- an extra bracket inside a quoted vector
    body = body_of('$key "[[1 1 1]"')
    assert body['$key'] == '[[1 1 1]'
    assert body.get_vector('$key') == (1.0, 1.0, 1.0)


def test_malformed_backtick_then_quote():
    body = body_of('$key `value"')
    assert body['$key'] == '`value'


def test_malformed_backtick_suffix_on_number():
    # shipped: Portal 2's nature/dirtfloor004d has `$detailblendfactor .4``
    body = body_of('$detailblendfactor .4`')
    assert body['$detailblendfactor'] == '.4`'
    assert body.get_float('$detailblendfactor') == 0.4


def test_quoted_condition_is_a_value_not_a_condition():
    # the distinction that matters most: quoting makes it data
    body = body_of('"$x" "[$WIN32]"')
    assert body['$x'] == '[$WIN32]'
    body = body_of('"$x" "[!$ps3]"')
    assert body['$x'] == '[!$ps3]'


def test_malformed_unterminated_bracket_vector_is_recovered():
    # closed at end of line, and normalised so downstream vector parsing still works
    body = body_of('$key [1 1 1')
    assert body['$key'] == '[1 1 1]'
    assert body.get_vector('$key') == (1.0, 1.0, 1.0)


def test_malformed_unterminated_bracket_condition_still_evaluates():
    assert body_of('$key value [$win32').get('$key') == 'value'
    assert body_of('$key value [$x360').get('$key') is None


def test_malformed_empty_condition_keeps_key():
    body = body_of('$key value []')
    assert body['$key'] == 'value'


def test_malformed_second_condition_is_reported_not_applied():
    block = loads('S {\n$key value [$win32] [$x360]\n}', 'm')
    assert block.top()[1]['$key'] == 'value'
    assert any('nothing to apply to' in d.message for d in errors(block))


def test_malformed_trailing_quote_after_bare_value():
    assert body_of('$key value"')['$key'] == 'value'


def test_malformed_quote_in_middle_of_bare_token():
    body = body_of('$key va"lue')
    assert body['$key'] == 'va'


def test_malformed_single_quoted_value():
    # shipped: props_xen/foliage/xen_creeper_glow_1 writes "$selfillum" '1'
    assert body_of("\"$selfillum\" '1'")['$selfillum'] == '1'


def test_malformed_empty_key_is_kept_with_its_value():
    # shipped: 16 maps contain `"" "tonemap,,,0,-1"` from Hammer outputs with no name.
    # Dropping the key would make the value look like the next key and lose both.
    body = body_of('"" "tonemap,,,0,-1"')
    assert body[''] == 'tonemap,,,0,-1'


def test_malformed_empty_value_is_kept():
    assert body_of('$key ""')['$key'] == ''


def test_malformed_brace_glued_to_value_and_key():
    assert body_of('$key "value"}').get('$key') == 'value'
    inner = body_of('$key{ $a 1 }')['$key']
    assert inner['$a'] == '1'


def test_malformed_junk_after_number():
    # the old parser raised ValueError on '42]'
    body = body_of('$alpha "42]"')
    assert body['$alpha'] == '42]'
    assert body.get_float('$alpha') == 42.0


def test_malformed_vector_with_punctuation():
    body = body_of('$c "[.5, .5; .5]"')
    assert body.get_vector('$c') == (0.5, 0.5, 0.5)


def test_dangling_key_absorbs_the_next_token():
    """KV1 is newline agnostic, so a value-less key pairs with whatever follows.

    This is Source's behaviour, not damage tolerance: ``KeyValues`` reads tokens and
    never treats a line break as a terminator. A key is only *dropped* when the block
    or the file ends before a value arrives.
    """
    block = loads('S {\n$key\n"$good" "1"\n}', 'm')
    body = block.top()[1]
    assert body['$key'] == '$good'
    # the leftover "1" then has no value of its own and is dropped at the `}`
    assert list(body.items()) == [('$key', '$good')]
    assert any("'1' has no value" in d.message for d in errors(block))

    body = loads('S {\n"$good" "1"\n$dangling\n}', 'm').top()[1]
    assert body['$good'] == '1'
    assert '$dangling' not in body


def test_malformed_stray_condition_alone():
    block = loads('S {\n[$win32]\n"$good" "1"\n}', 'm')
    assert block.top()[1]['$good'] == '1'


def test_malformed_control_bytes_in_value():
    body = body_of('$key \x01\x02\x03')
    assert body['$key'] == '\x01\x02\x03'


def test_malformed_inputs_never_lose_a_following_good_pair():
    """Self-contained damage must not reach the next pair.

    Lines whose key already got a value are listed here. A line whose key is left
    *without* a value legitimately pairs with the next token instead -- see
    :func:`test_dangling_key_absorbs_the_next_token` -- so those are excluded rather
    than asserted against.
    """
    self_contained = [
        '$key "[[1 1 1]"', '$key `value"', '$key [1 1 1', '$key value []',
        '$key value"', '$key ""', '"" "x"', '$alpha "42]"',
        '$key value [$win32] [$x360]', '$key value [',
        '$key "a" "b" "c"', '$key {}', '$key value [$x360]',
        '$key "unclosed', "$key 'unclosed", '$key \x01\x02', '[$win32]',
    ]
    for line in self_contained:
        block = loads('S {\n' + line + '\n"$sentinel" "ok"\n}', 'm')
        body = block.top()[1]
        assert body.get('$sentinel') == 'ok', f'sentinel lost after {line!r}'


def test_value_less_keys_absorb_but_never_crash():
    """The other half: keys with no value on the line. Data may shift, nothing dies."""
    absorbing = ['$key', '$key `', '$key [', '$key ]', '$key va"lue']
    for line in absorbing:
        block = loads('S {\n' + line + '\n"$sentinel" "ok"\n}', 'm')
        body = block.top()[1]
        assert isinstance(body, KV1Block)
        # the sentinel key becomes this key's value rather than vanishing silently
        assert '$sentinel' in list(body.values()) or body.get('$sentinel') == 'ok', \
            f'{line!r} -> {list(body.items())}'


def test_malformed_lines_do_not_leak_into_the_next_block():
    block = loads('''A {
    "$broken" "unclosed
}
B {
    "$good" "1"
}''', 'm')
    assert block['b']['$good'] == '1'


# ------------------------------------------------------------------- writing

def test_dumps_round_trips_a_block():
    src = '''Shader
{
    "$basetexture" "brick/wall"
    "proxies" { "sine" { "resultvar" "$alpha" } }
}'''
    block = loads(src, 'src')
    assert loads(dumps(block), 'out').to_dict() == block.to_dict()


def test_dumps_preserves_duplicates_and_order():
    block = loads('F { S { Game a\n Game b\n Mod c } }', 'src')
    out = loads(dumps(block), 'out')
    assert out['f']['s'].get_all('game') == ['a', 'b']
    assert [k for k, _ in out['f']['s'].items()] == ['game', 'game', 'mod']


def test_dumps_writes_blocks_as_a_bare_sequence():
    ents = loads_blocks('{\n"classname" "a"\n"OnTrigger" "x"\n"OnTrigger" "y"\n}\n'
                        '{\n"classname" "b"\n}')
    back = loads_blocks(dumps(ents), 'out')
    assert [e.to_dict() for e in back] == [e.to_dict() for e in ents]
    assert back[0].get_all('ontrigger') == ['x', 'y']


def test_dumps_quotes_values_that_need_it():
    block = loads('S { "$a" "has spaces" }', 'src')
    text = dumps(block)
    assert '"$a" "has spaces"' in text
    assert loads(text, 'out')['s']['$a'] == 'has spaces'


def test_dumps_keeps_a_recorded_condition():
    block = loads('S { "$a" "1" [$win32] }', 'src')
    text = dumps(block)
    assert '[$win32]' in text
    assert loads(text, 'out')['s']['$a'] == '1'


def test_dumps_escapes_when_asked():
    block = loads(r'S { "path" "C:\\Steam" }', 'src', escapes=True)
    text = dumps(block, escapes=True)
    assert loads(text, 'out', escapes=True)['s']['path'] == r'C:\Steam'


def test_dumps_backslash_paths_survive_without_escapes():
    block = loads(r'S { "$basetexture" "models\props\crate" }', 'src')
    out = loads(dumps(block), 'out')
    assert out['s']['$basetexture'] == r'models\props\crate'


def test_dumps_empty_block_and_empty_key():
    assert dumps(KV1Block()) == ''
    block = loads('S { "" "nameless" }', 'src')
    assert loads(dumps(block), 'out')['s'][''] == 'nameless'


def test_dumps_records_includes():
    block = loads('#base "shared.vdf"\nS { a 1 }', 'src')
    text = dumps(block)
    assert '#base "shared.vdf"' in text
    assert loads(text, 'out').includes == ['shared.vdf']


def test_dumps_indent_is_configurable():
    block = loads('S { a 1 }', 'src')
    assert '\n    "a" "1"' in dumps(block, indent='    ')


def test_dump_writes_a_file(tmp_path):
    block = loads('S { "$a" "1"\n"$b" "2" }', 'src')
    target = tmp_path / 'out.vmt'
    dump(block, target)
    assert loads(target.read_text(), 'out').to_dict() == block.to_dict()
