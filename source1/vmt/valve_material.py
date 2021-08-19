from ...content_providers.content_manager import ContentManager
from ...bpy_utilities.logging import BPYLoggingManager
from .vmt_parser import VMTParser

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('valve_material')


# TODO: remove this and just use VMTParser with ported PATCH material support
class VMT:
    def __init__(self, file_object):

        try:
            self.material: VMTParser = VMTParser(file_object)
        except ValueError as e:
            self.material: VMTParser = VMTParser("invalid{}")
            logger.warn(f'Cannot parse material file due to: {e}')

    @property
    def shader(self):
        return self.material.header

    def get_param(self, name, default):
        return self.material.get_raw_data().get(name.lower(), default)

    def get_raw_data(self):
        return self.material.get_raw_data()

    def parse(self) -> VMTParser:
        content_manager = ContentManager()
        if self.material.header == 'patch':
            original_material = content_manager.find_file(self.material.get_string('include'))
            logger.info(f'Got "Patch" material, applying patch to {self.material.get_string("include")}')
            if original_material:
                new_vmt = VMT(original_material)
                new_vmt.parse()
                new_material = new_vmt.material
                if 'insert' in self.material.get_raw_data():
                    patch_data = self.material.get_subblock('insert', {})
                    new_material.apply_patch(patch_data)
                if 'replace' in self.material.get_raw_data():
                    patch_data = self.material.get_subblock('replace', {})
                    new_material.apply_patch(patch_data)
                self.material = new_material
            else:
                logger.error('Failed to find original material')
        return self.material
