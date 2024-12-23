from .resource import CompiledResource


class CompiledMeshResource(CompiledResource):

    def get_name(self):
        data, = self.get_data_block(block_name='DATA')
        return data['m_name']
