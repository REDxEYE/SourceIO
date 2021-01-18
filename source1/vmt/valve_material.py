from ...source_shared.content_manager import ContentManager
from ...bpy_utilities.logging import BPYLoggingManager
from ...utilities.keyvalues import KVParser

log_manager = BPYLoggingManager()
logger = log_manager.get_logger('valve_material')


class VMT:
    def __init__(self, file_object):
        KVParser.set_strict_parsing_mode(True)
        kv_parser = KVParser('VMT', file_object.read(-1).decode('latin', errors='replace'))
        try:
            self.shader, self.material_data = kv_parser.parse()
            self.shader = self.shader.lower()
        except ValueError as e:
            self.shader = '<invalid>'
            self.material_data = {}
            logger.warn(f'Cannot parse material file: {e}')
        KVParser.set_strict_parsing_mode(False)

    def get_param(self, name, default):
        return self.material_data.get(name, default)

    def parse(self):
        content_manager = ContentManager()
        if self.shader == 'patch':
            original_material = content_manager.find_file(self.material_data['include'])
            if original_material:
                KVParser.set_strict_parsing_mode(True)
                kv_parser = KVParser('VMT', original_material.read(-1).decode())
                KVParser.set_strict_parsing_mode(False)
                old_data = self.material_data
                self.shader, self.material_data = kv_parser.parse()
                self.shader = self.shader.lower()
                if 'insert' in old_data:
                    patch_data = old_data['insert']
                    self.material_data.update(patch_data)
                if 'replace' in old_data:
                    patch_data = old_data['replace']
                    self.material_data.update(patch_data)
            else:
                print('Failed to find original material')
                return
