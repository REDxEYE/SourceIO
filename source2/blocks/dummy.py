from ...byte_io_mdl import ByteIO


class Dummy:

    def __init__(self):
        self.empty = True

    def read(self, reader: ByteIO):
        raise NotImplementedError()

    def __repr__(self):
        template = '<{} {}>'
        member_template = '{}:{}'
        members = []
        for key, item in self.__dict__.items():
            members.append(member_template.format(key, item))
        return template.format(type(self).__name__, ' '.join(members))
