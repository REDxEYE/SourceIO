import struct
from enum import IntEnum
from pathlib import Path

import numpy as np
from typing import Dict, Any

from .mgr import ContentManager
from ..wad import make_texture, flip_texture


class BspLumpType(IntEnum):
    LUMP_ENTITIES = 0
    LUMP_PLANES = 1
    LUMP_TEXTURES_DATA = 2
    LUMP_VERTICES = 3
    LUMP_VISIBILITY = 4
    LUMP_NODES = 5
    LUMP_TEXTURES_INFO = 6
    LUMP_FACES = 7
    LUMP_LIGHTING = 8
    LUMP_CLIP_NODES = 9
    LUMP_LEAVES = 10
    LUMP_MARK_SURFACES = 11
    LUMP_EDGES = 12
    LUMP_SURFACE_EDGES = 13
    LUMP_MODELS = 14


class BspLump:
    def __init__(self, file: 'BspFile', type: BspLumpType):
        self.file = file
        self.type = type
        self.offset, self.length = struct.unpack('II', file.handle.read(8))

    def parse(self):
        raise NotImplementedError

    def get_contents(self):
        handle = self.file.handle
        position = handle.tell()
        handle.seek(self.offset)
        contents = handle.read(self.length)
        handle.seek(position)
        return contents

    @staticmethod
    def get_handler(type: BspLumpType):
        if type == BspLumpType.LUMP_ENTITIES:
            return BspEntitiesLump
        if type == BspLumpType.LUMP_TEXTURES_DATA:
            return BspTexturesDataLump
        if type == BspLumpType.LUMP_VERTICES:
            return BspVerticesLump
        if type == BspLumpType.LUMP_TEXTURES_INFO:
            return BspTexturesInfoLump
        if type == BspLumpType.LUMP_FACES:
            return BspFaceLump
        if type == BspLumpType.LUMP_EDGES:
            return BspEdgeLump
        if type == BspLumpType.LUMP_SURFACE_EDGES:
            return BspSurfaceEdgeLump
        if type == BspLumpType.LUMP_MODELS:
            return BspModelsLump
        return lambda file: BspLump(file, type)

    def __repr__(self):
        return f'<BspLump {self.type.name} at {self.offset}:{self.length}>'


class BspEntitiesLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_ENTITIES)
        self.values = []

    def parse(self):
        handle = self.file.handle
        handle.seek(self.offset)

        entities = handle.read(self.length)
        entities = entities[:entities.index(b'\x00')].decode()
        entity = {}
        for line in entities.splitlines():
            if line == '{' or len(line) == 0:
                continue
            elif line == '}':
                self.values.append(entity)
                entity = {}
            else:
                entity_key_start = line.index('"') + 1
                entity_key_end = line.index('"', entity_key_start)
                entity_value_start = line.index('"', entity_key_end + 1) + 1
                entity_value_end = line.index('"', entity_value_start)
                entity[line[entity_key_start:entity_key_end]] = line[entity_value_start:entity_value_end]
        return self


class BspTexturesDataLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_TEXTURES_DATA)
        self.values = []

    def parse(self):
        handle = self.file.handle
        handle.seek(self.offset)
        textures_count = struct.unpack('I', handle.read(4))[0]
        textures_offset = struct.unpack('I' * textures_count, handle.read(4 * textures_count))
        for texture_offset in textures_offset:
            if texture_offset == 0xffffffff:
                continue
            handle.seek(self.offset + texture_offset)
            texture_name = handle.read(16)
            texture_name = texture_name[:texture_name.index(b'\x00')].decode().upper()
            texture_width, texture_height = struct.unpack('II', handle.read(8))
            texture_offsets = struct.unpack('4I', handle.read(16))

            texture_indices = []
            texture_palette = []

            if any(texture_offsets):
                for index, offset in enumerate(texture_offsets):
                    handle.seek(self.offset + texture_offset + offset)
                    texture_size = (texture_width * texture_height) >> (index * 2)
                    texture_indices.append(struct.unpack('B' * texture_size, handle.read(texture_size)))

                assert handle.read(2) == b'\x00\x01', 'Invalid palette start anchor'

                for _ in range(256):
                    texture_palette.append(struct.unpack('BBB', handle.read(3)))

                assert handle.read(2) == b'\x00\x00', 'Invalid palette end anchor'

                texture_data = make_texture(texture_indices[0], texture_palette, use_alpha=texture_name.startswith('{'))
                texture_data = flip_texture(texture_data, texture_width, texture_height)
                texture_data = texture_data.flatten().tolist()
            else:
                texture_resource = self.file.manager.get_game_resource(texture_name)
                if not texture_resource:
                    print(f'Could not find texture resource: {texture_name}')
                    texture_data = [0.5 for _ in range(texture_width * texture_height * 4)]
                else:
                    texture_data = texture_resource.read_texture()[0]

            self.values.append({
                'name': texture_name,
                'width': texture_width,
                'height': texture_height,
                'data': texture_data
            })
        return self


class BspVerticesLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_VERTICES)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.float32)
        self.values = self.values.reshape((-1, 3))
        return self


class BspTexturesInfoLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_TEXTURES_INFO)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 40):
            s = struct.unpack('ffff', self.file.handle.read(16))
            t = struct.unpack('ffff', self.file.handle.read(16))
            texture, flags = struct.unpack('II', self.file.handle.read(8))
            self.values.append({
                's': s,
                't': t,
                'texture': texture,
                'flags': flags
            })
        return self


class BspFaceLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_FACES)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 20):
            (plane,
             plane_side,
             first_edge,
             edges,
             texture_info,
             styles,
             light_map_offset) = struct.unpack('HHIHHII', self.file.handle.read(20))

            self.values.append({
                'plane': plane,
                'plane_side': plane_side,
                'first_edge': first_edge,
                'edges': edges,
                'texture_info': texture_info,
                'styles': struct.unpack('BBBB', struct.pack('I', styles)),
                'light_map_offset': light_map_offset
            })
        return self


class BspEdgeLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_EDGES)
        self.values = np.array([])

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.int16)
        self.values = self.values.reshape((-1, 2))
        return self


class BspSurfaceEdgeLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_SURFACE_EDGES)
        self.values = np.array([])

    def parse(self):
        self.file.handle.seek(self.offset)
        self.values = np.frombuffer(self.file.handle.read(self.length), np.int32)
        return self


class BspModelsLump(BspLump):
    def __init__(self, file: 'BspFile'):
        super().__init__(file, BspLumpType.LUMP_MODELS)
        self.values = []

    def parse(self):
        self.file.handle.seek(self.offset)
        for _ in range(self.length // 60):
            mins = struct.unpack('3f', self.file.handle.read(12))
            maxs = struct.unpack('3f', self.file.handle.read(12))
            origin = struct.unpack('3f', self.file.handle.read(12))
            head_nodes = struct.unpack('4I', self.file.handle.read(16))
            vis_leafs, first_face, faces = struct.unpack('III', self.file.handle.read(12))
            self.values.append({
                'mins': mins,
                'maxs': maxs,
                'origin': origin,
                'head_nodes': head_nodes,
                'vis_leafs': vis_leafs,
                'first_face': first_face,
                'faces': faces,
            })
        return self


class BspFile:
    def __init__(self, file: Path):
        self.manager = ContentManager(file)
        self.handle = file.open('rb')
        self.version = struct.unpack('<I', self.handle.read(4))[0]
        self.lumps = [BspLump.get_handler(type)(self) for type in BspLumpType]
        assert self.version in (29, 30), 'Not a GoldSRC map file (BSP29, BSP30)'


