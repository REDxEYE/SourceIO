"""Source 1 SMD mesh, animation, and VTA interchange contracts."""

import io
import unittest

import numpy as np

from SourceIO.library.models.smd import (
    SMDBone, SMDDocument, SMDNode, SMDVertexAnimationVertex, dump, load, parse, triangle_array,
)


class SMDIOTests(unittest.TestCase):
    def test_mesh_preserves_names_uvs_material_and_explicit_links(self):
        triangles = triangle_array(1, 1)
        corners = triangles['corners'][0]
        corners['bone'] = (0, 1, 0)
        corners['position'] = ((1., 2., 3.), (2., 3., 4.), (3., 4., 5.))
        corners['normal'] = ((0., 0., 1.), (0., 0., 1.), (0., 1., 0.))
        corners['uv'] = ((.25, .75), (1., 0.), (0., 1.))
        corners['link_count'][[0, 2]] = (1, 0)  # the middle corner omits the optional count
        corners['links'][0, 0] = (1, .375)
        document = SMDDocument(
            nodes=[SMDNode(0, 'root // bone', -1), SMDNode(1, 'child\\bone', 0)],
            skeleton={0: [SMDBone(0, (0., 0., 0.), (0., 0., 0.)), SMDBone(1, (1., 0., 0.), (0., .5, 0.))]},
            materials=['materials/panel face'],
            triangles=triangles,
        )
        text = dump(document)
        self.assertEqual(parse(text), document)
        self.assertEqual(parse(dump(parse(text))), document)
        restored = parse(text).triangles['corners'][0]
        self.assertEqual(restored['link_count'].tolist(), [1, -1, 0])
        self.assertEqual(restored['links'][0, 0].tolist(), (1, .375))
        self.assertEqual(parse(text).materials, ['materials/panel face'])

    def test_mixed_link_counts_text_and_validation(self):
        def document():
            triangles = triangle_array(2, 2)
            triangles['corners']['position'] = np.arange(18.).reshape(2, 3, 3) / 8
            triangles['corners']['link_count'] = ((0, 2, -1), (1, 1, 2))
            triangles['corners']['links']['bone'] = 1
            triangles['corners']['links']['weight'] = .25
            triangles['material'] = (0, 1)
            return SMDDocument(nodes=[SMDNode(0, 'root', -1), SMDNode(1, 'tip', 0)], skeleton={0: []},
                               materials=['a', 'b'], triangles=triangles)

        text = dump(document())
        self.assertIn('a\n0 0.0 0.125 0.25 0.0 0.0 0.0 0.0 0.0 0\n'
                      '0 0.375 0.5 0.625 0.0 0.0 0.0 0.0 0.0 2 1 0.25 1 0.25\n'
                      '0 0.75 0.875 1.0 0.0 0.0 0.0 0.0 0.0\n'
                      'b\n', text)
        self.assertEqual(dump(parse(text)), text)  # padding beyond link_count is not significant
        for label, change in (
                ('weight', lambda d: d.triangles['corners']['links']['weight'].__setitem__((0, 1, 0), 1.5)),
                ('sum', lambda d: d.triangles['corners']['links']['weight'].__setitem__((1, 2), .75)),
                ('bone', lambda d: d.triangles['corners']['bone'].__setitem__((1, 0), 7)),
                ('link bone', lambda d: d.triangles['corners']['links']['bone'].__setitem__((0, 1, 1), 7)),
                ('nan', lambda d: d.triangles['corners']['uv'].__setitem__((1, 1, 0), np.nan)),
                ('count', lambda d: d.triangles['corners']['link_count'].__setitem__((0, 0), 3)),
                ('material', lambda d: d.triangles['material'].__setitem__(1, 2)),
                ('name', lambda d: d.materials.__setitem__(1, 'end'))):
            with self.subTest(label), self.assertRaises(ValueError):
                broken = document()
                change(broken)
                dump(broken)

    def test_literal_backslashes_and_unrepresentable_quotes(self):
        for name in ('bone\\', r'child\\bone'):
            with self.subTest(name=name):
                text = f'version 1\nnodes\n0 "{name}" -1 // comment\nend\n'
                document = parse(text)
                self.assertEqual(document.nodes[0].name, name)
                self.assertIn(f'0 "{name}" -1\n', dump(document))
        with self.assertRaises(ValueError):
            dump(SMDDocument(nodes=[SMDNode(0, 'bad"bone', -1)]))

    def test_sparse_animation_frames_and_caller_owned_streams(self):
        document = SMDDocument(nodes=[SMDNode(0, 'root', -1)],
                               skeleton={7: [SMDBone(0, (1., 2., 3.), (0., 0., 0.))],
                                         0: [SMDBone(0, (0., 0., 0.), (0., 0., 0.))]})
        text = dump(document)
        self.assertLess(text.index('time 0'), text.index('time 7'))
        self.assertEqual(parse(text), document)
        output = io.StringIO()
        self.assertIsNone(dump(document, output))
        self.assertFalse(output.closed)
        stream = io.BytesIO(output.getvalue().replace('\n', '\r\n').encode('utf-8'))
        self.assertEqual(load(file=stream), document)
        self.assertFalse(stream.closed)

    def test_vta_frames_and_comments(self):
        text = ('\ufeffversion 1 // VTA\r\n'
                'nodes\r\n0 "root" -1\r\nend\r\n'
                'skeleton\r\ntime 0\r\n0 0 0 0 0 0 0\r\nend\r\n'
                'vertexanimation\r\ntime 5\r\n'
                '2 1 2 3 0 0 1 // changed vertex\r\n'
                'time 0\r\n2 0 0 0 0 0 1\r\nend\r\n')
        document = parse(text)
        self.assertEqual(document.vertex_animation[5][0], SMDVertexAnimationVertex(2, (1., 2., 3.), (0., 0., 1.)))
        self.assertEqual(parse(dump(document)), document)

    def test_text_vectors_preserve_decimal_precision(self):
        value = 0.123456789012345
        document = SMDDocument(
            nodes=[SMDNode(0, 'root', -1)],
            skeleton={0: [SMDBone(0, (value, 0., 0.), (0., 0., 0.))]},
            vertex_animation={1: [SMDVertexAnimationVertex(0, (value, 0., 0.), (0., 0., 1.))]},
        )
        restored = parse(dump(document))
        self.assertEqual(restored, document)
        self.assertEqual(restored.vertex_animation[1][0].position[0], value)


    def test_empty_sections_stay_empty(self):
        text = 'version 1\nnodes\nend\nskeleton\nend\ntriangles\nend\nvertexanimation\nend\n'
        self.assertEqual(parse(text), SMDDocument())
        self.assertEqual(parse(dump(parse(text))), SMDDocument())


    def test_rejects_truncated_or_invalid_sections_and_weights(self):
        root = 'version 1\nnodes\n0 "root" -1\nend\nskeleton\ntime 0\n0 0 0 0 0 0 0\nend\n'
        cases = (
            'version 2\nnodes\nend\n',
            root + 'nodes\nend\n',
            'version 1\nnodes\n0 "root" -1\n',
            'version 1\nnodes\n0 "root" 1\nend\n',
            'version 1\nnodes\n0 "a" 1\n1 "b" 0\nend\n',
            root + 'triangles\nmaterial\n0 0 0 0 0 0 1 0 0\nend\n',
            root + 'triangles\nmaterial\n' + '0 0 0 0 0 0 1 0 0 1 0 1.1\n' * 3 + 'end\n',
            root + 'vertexanimation\n0 1 2 3 0 0 1\nend\n',
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse(text)
        with self.assertRaises(ValueError):
            load()
        with self.assertRaises(ValueError):
            load(file=io.StringIO(root), filepath='other.smd')


if __name__ == '__main__':
    unittest.main()
