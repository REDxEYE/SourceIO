"""KeyValues2 DMX interchange through the standalone v2 codecs."""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from SourceIO.library.utils import datamodel as legacy
from SourceIO.library.utils import datamodel_v2 as dm


ROOT = '00000000-0000-0000-0000-000000000001'
CHILD = '00000000-0000-0000-0000-000000000002'
HEADER = '<!-- dmx encoding keyvalues2 1 format model 1 -->\n'
HANDWRITTEN = HEADER + f'''// Forward references and an inline element inside an array.
"$prefix_element$" {{ "author" "string" "SourceIO" }}
"DmElement"
{{
    "id" "elementid" "{ROOT}"
    "name" "string" "Root"
    "peer" "element" "{CHILD}"
    "children" "element_array" [
        "DmeChild" {{
            "id" "elementid" "{CHILD}"
            "name" "string" "Child"
            "back" "element" "{ROOT}"
        }},
        "element" "{CHILD}"
    ]
    "missing" "element" ""
    "flags" "bool_array" [ "1", "0" ]
    "positions" "vector3_array" [ "1 2 3", "4 5 6" ]
    "raw" "binary" "00ff"
    "time" "time" "1.25"
    "title" "string" "line\\nquote\\\" and slash\\\\ and ü"
}}
'''
FIXTURES = Path(__file__).parent / 'blender_tests/fixtures/source1_dmx/data'


class KeyValues2CodecTests(unittest.TestCase):
    def test_handwritten_nested_forward_and_cyclic_references(self):
        document = dm.parse(HANDWRITTEN)
        root = document.root
        child = root['children'][0]
        self.assertIs(root['peer'], child)
        self.assertIs(root['children'][1], child)
        self.assertIs(child['back'], root)
        self.assertEqual(document.prefix_attributes['author'], 'SourceIO')
        self.assertIsInstance(root['missing'], dm.NoneElement)
        self.assertIsInstance(root['flags'], dm.BoolArray)
        self.assertEqual(list(root['flags']), [True, False])
        np.testing.assert_array_equal(root['positions'], ((1, 2, 3), (4, 5, 6)))
        self.assertEqual(root['raw'], b'\x00\xff')
        self.assertEqual(root['time'], 1.25)
        self.assertEqual(root['title'], 'line\nquote" and slash\\ and ü')

        text = dm.dump(document)
        reread = dm.parse(text)
        self.assertEqual(dm.dump(reread), text)
        self.assertIs(reread.root['children'][0]['back'], reread.root)
        self.assertIs(reread.root['peer'], reread.root['children'][1])

    def test_scalar_and_empty_array_types_survive_text_roundtrip(self):
        document = dm.DataModel('keyvalues2', 4, 'model', 18)
        root = document.add_element('Types', uuid=ROOT)
        root['angle'] = dm.Angle((1, 2, 3))
        root['rotation'] = dm.Quaternion((0, 0, 0, 1))
        root['matrix'] = dm.Matrix(np.eye(4, dtype=np.float32))
        root['color'] = dm.Color((0, 127, 255, 3))
        root['long'] = np.uint64(2**63 + 7)
        root['byte'] = np.uint8(255)
        root['empty_strings'] = dm.StringArray()
        root['empty_vectors'] = dm.Vector3Array([])
        root['empty_elements'] = dm.ElementArray()
        root['times'] = dm.TimeArray([dm.Time(0.5), dm.Time(1.25)])
        root['quaternions'] = dm.QuaternionArray([(0, 0, 0, 1)])
        reread = dm.parse(dm.dump(document))
        self.assertIsInstance(reread.root['angle'], dm.Angle)
        self.assertIsInstance(reread.root['rotation'], dm.Quaternion)
        np.testing.assert_array_equal(reread.root['matrix'], np.eye(4))
        np.testing.assert_array_equal(reread.root['color'], (0, 127, 255, 3))
        self.assertIsInstance(reread.root['long'], np.uint64)
        self.assertEqual(reread.root['long'], 2**63 + 7)
        self.assertEqual(reread.root['byte'], 255)
        for key, expected_type in (('empty_strings', dm.StringArray),
                                   ('empty_vectors', dm.Vector3Array),
                                   ('empty_elements', dm.ElementArray),
                                   ('times', dm.TimeArray),
                                   ('quaternions', dm.QuaternionArray)):
            self.assertIsInstance(reread.root[key], expected_type)
        self.assertEqual(list(reread.root['times']), [0.5, 1.25])
        self.assertEqual(reread.root['quaternions'].shape, (1, 4))

    def test_binary_v9_unsigned_types_survive_ascii_conversion(self):
        document = dm.DataModel('binary', 9, 'model', 18)
        root = document.add_element('Unsigned', uuid=ROOT)
        root['long'] = np.uint64(7)
        root['byte'] = np.uint8(255)
        root['longs'] = dm.UInt64Array([2**63 + 7])
        for version in (1, 2, 3, 4):
            with self.subTest(keyvalues2_version=version):
                text = dm.dump(document, encoding='keyvalues2', encoding_ver=version)
                parsed = dm.parse(text)
                binary = dm.dump(parsed, encoding='binary', encoding_ver=9)
                restored = dm.load(file=io.BytesIO(binary))
                self.assertEqual(restored.encoding_ver, 9)
                self.assertIsInstance(restored.root['long'], np.uint64)
                self.assertIsInstance(restored.root['byte'], np.uint8)
                self.assertEqual(restored.root['long'], 7)
                self.assertEqual(restored.root['byte'], 255)
                self.assertEqual(restored.root['longs'][0], 2**63 + 7)

    def test_existing_binary_fixtures_convert_without_losing_model_graph(self):
        for name, binary_ver, model_ver in (('single', 2, 1), ('flex', 4, 15), ('vertex_animation', 5, 18)):
            with self.subTest(name=name):
                original = dm.load(filepath=FIXTURES / f'{name}_{binary_ver}_{model_ver}.dmx')
                text = dm.dump(original, encoding='keyvalues2', encoding_ver=1)
                parsed = dm.load(file=io.BytesIO(text.encode('utf-8')))
                self.assertEqual((parsed.encoding, parsed.format_ver), ('keyvalues2', model_ver))
                self.assertIs(parsed.root['skeleton'], parsed.root.get('model', parsed.root['skeleton']))
                binary = dm.dump(parsed, encoding='binary', encoding_ver=binary_ver)
                restored = dm.load(file=io.BytesIO(binary))
                self.assertEqual(len(restored.elements), len(original.elements))
                self.assertEqual([item.dm_type for item in restored.elements],
                                 [item.dm_type for item in original.elements])
                self.assertEqual([item.name for item in restored.elements], [item.name for item in original.elements])
                self.assertEqual([item.id for item in restored.elements], [item.id for item in original.elements])
                self.assertEqual(restored.root.name, original.root.name)

    def test_legacy_generated_ascii_is_readable_and_v2_output_is_interoperable(self):
        old = legacy.DataModel('model', 1)
        root = old.add_element('Root', id=ROOT)
        child = old.add_element('Child', 'DmeChild', id=CHILD)
        root['child'] = child
        root['label'] = 'sample'
        parsed = dm.parse(old.echo('keyvalues2', 1))
        self.assertEqual(parsed.root['child'].name, 'Child')
        echoed = legacy.parse(dm.dump(parsed, encoding='keyvalues2', encoding_ver=1))
        self.assertEqual(echoed.root['child'].name, 'Child')

    def test_file_and_stream_inputs_dispatch_without_closing_caller_streams(self):
        document = dm.parse(HANDWRITTEN)
        text = dm.dump(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.dmx'
            path.write_text(text, encoding='utf-8')
            self.assertEqual(dm.load(filepath=path).root.name, 'Root')
        for stream in (io.StringIO(text), io.BytesIO(text.encode('utf-8'))):
            self.assertEqual(dm.load(file=stream).root.name, 'Root')
            self.assertFalse(stream.closed)
        binary = io.BytesIO(dm.dump(document, encoding='binary', encoding_ver=3))
        loaded = dm.load(file=binary)
        self.assertEqual(loaded.root.name, 'Root')
        self.assertFalse(binary.closed)
        self.assertIsInstance(loaded.root['missing'], dm.NoneElement)

    def test_bad_ascii_fails_with_context_before_binary_decode(self):
        cases = (
            (HEADER + f'"DmElement" {{ "id" "elementid" "{ROOT}" '
             '"name" "string" "Root" "bad" "vector3" "1 2" }}', 'line'),
            (HEADER + f'"DmElement" {{ "id" "elementid" "{ROOT}" '
             f'"name" "string" "Root" "ref" "element" "{CHILD}" }}', 'Unresolved'),
            (HEADER + f'"DmElement" {{ "id" "elementid" "{ROOT}" '
             f'"name" "string" "Root" }} "DmElement" {{ "id" "elementid" "{ROOT}" }}', 'Duplicate'),
            (HEADER + f'"DmElement" {{ "id" "elementid" "{ROOT}" '
             '"name" "string" "Root" "bad" "unknown" "x" }}', 'Unsupported'),
            (HEADER + '"DmElement" { "name" "string" "unfinished', 'Unterminated'),
        )
        for source, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(dm.KeyValues2ParseError, message):
                dm.load(file=io.BytesIO(source.encode('utf-8')))
        with self.assertRaisesRegex(ValueError, 'Unsupported keyvalues2'):
            dm.parse(HEADER.replace('keyvalues2 1', 'keyvalues2 9') + '"DmElement" {}')
        with self.assertRaisesRegex(ValueError, 'Invalid DMX header'):
            dm.load(file=io.BytesIO(b'<!-- broken -->\nnot binary'))


if __name__ == '__main__':
    unittest.main()
