"""Source 1 SMD mesh, animation, and VTA interchange contracts."""

import io
import unittest

from SourceIO.library.models.smd import (
    SMDBone, SMDDocument, SMDNode, SMDTriangle, SMDVertex, SMDVertexAnimationVertex, SMDWeight, dump, load, parse,
)


class SMDIOTests(unittest.TestCase):
    def test_mesh_preserves_names_uvs_material_and_explicit_links(self):
        vertex = SMDVertex(0, (1., 2., 3.), (0., 0., 1.), (.25, .75), (SMDWeight(1, .375),))
        rigid = SMDVertex(1, (2., 3., 4.), (0., 0., 1.), (1., 0.))
        explicit_zero = SMDVertex(0, (3., 4., 5.), (0., 1., 0.), (0., 1.), ())
        document = SMDDocument(
            nodes=[SMDNode(0, 'root // "bone"', -1), SMDNode(1, 'child\\bone', 0)],
            skeleton={0: [SMDBone(0, (0., 0., 0.), (0., 0., 0.)), SMDBone(1, (1., 0., 0.), (0., .5, 0.))]},
            triangles=[SMDTriangle('materials/panel face', (vertex, rigid, explicit_zero))],
        )
        text = dump(document)
        self.assertEqual(parse(text), document)
        self.assertEqual(parse(dump(parse(text))), document)
        self.assertEqual(parse(text).triangles[0].vertices[0].weights, (SMDWeight(1, .375),))
        self.assertIsNone(parse(text).triangles[0].vertices[1].weights)
        self.assertEqual(parse(text).triangles[0].vertices[2].weights, ())

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
