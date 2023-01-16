from .resource import CompiledResource


class CompiledMeshResource(CompiledResource):

    def get_name(self):
        data, = self.get_data_block(block_name='DATA')
        return data['m_name']

    def get_index_buffer(self):
        data_block: DATA = self.get_data_block(block_name='DATA')[0]
        data = data_block.data
        if 'm_indexBuffers' in data:
            info_block = InfoBlock("VBIB", 0, 0)
            data_index_buffers = data['m_indexBuffers']
            vbib = VBIB(self, None)
            vbib.parsed = True
            vbib.info_block = info_block
            for data_index_buffer in data_index_buffers:
                index_buffer = IndexBuffer()
                index_buffer.index_count = data_index_buffer['m_nElementCount']
                vbib.index_count += index_buffer.index_count
                index_buffer.index_size = data_index_buffer['m_nElementSizeInBytes']
                index_dtype = np.uint16 if index_buffer.index_size == 2 else np.uint32
                index_buffer.indices = data_index_buffer['m_pData'].view(index_dtype).copy().reshape((-1, 3))
                vbib.index_buffer.append(index_buffer)

    def get_vertex_buffer(self):
        data_block: DATA = self.get_data_block(block_name='DATA')[0]
        data = data_block.data
        if 'm_vertexBuffers' in data:
            info_block = InfoBlock("VBIB", 0, 0)
            data_vertex_buffers = data['m_vertexBuffers']
            vbib = VBIB(self, None)
            vbib.parsed = True
            vbib.info_block = info_block
            for data_vertex_buffer in data_vertex_buffers:
                vertex_buffer = VertexBuffer()
                vertex_buffer.vertex_count = data_vertex_buffer['m_nElementCount']
                vbib.vertex_count += vertex_buffer.vertex_count
                vertex_buffer.vertex_size = data_vertex_buffer['m_nElementSizeInBytes']
                vertex_buffer.buffer = ByteIO(data_vertex_buffer['m_pData'].tobytes())
                seen_blend_indices = False
                for attribute in data_vertex_buffer['m_inputLayoutFields']:
                    vertex_attribute = VertexAttribute()
                    vertex_attribute.name = semantic_id_to_name[attribute['m_pSemanticName']]
                    if vertex_attribute.name == 'BLENDINDICES' and seen_blend_indices:
                        vertex_attribute.name = 'BLENDWEIGHT'
                    elif vertex_attribute.name == 'BLENDINDICES':
                        seen_blend_indices = True
                    vertex_attribute.offset = attribute['m_nOffset']
                    vertex_attribute.format = DxgiFormat(attribute['m_Format'])
                    vertex_buffer.attribute_names.append(vertex_attribute.name)
                    vertex_buffer.attributes.append(vertex_attribute)
                vertex_buffer.attribute_names = list(set(vertex_buffer.attribute_names))
                vertex_buffer.read_buffer()
                vbib.vertex_buffer.append(vertex_buffer)