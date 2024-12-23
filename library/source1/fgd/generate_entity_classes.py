import os
from valvefgd import Fgd, FgdEntity, FgdParse

from SourceIO.library.utils.tiny_path import TinyPath

os.environ['NO_BPY'] = '1'
from ...content_manager import ContentManager


def parse_int_vector(string):
    return [int(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    if string is None:
        return [0.0, 0.0, 0.0]
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


def collect_parents(parent: FgdEntity):
    parents = []
    for pparent in parent.parents:
        parents += collect_parents(pparent)
    return [parent] + parents


def main():
    # fgd_path = r"F:\SteamLibrary\steamapps\common\Black Mesa\bin\bms.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\Team Fortress 2\bin\base.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\Team Fortress 2\bin\tf.fgd"
    # fgd_path = r"D:\SteamLibrary\steamapps\common\Portal\bin\portal.fgd"
    # fgd_path = r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\bin\csgo.fgd"
    # fgd_path = r"D:\SteamLibrary\steamapps\common\Portal 2\bin\portal2.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\Left 4 Dead 2\bin\left4dead2.fgd"
    # fgd_path = r"F:\SteamLibrary\steamapps\common\Half-Life 2\bin\halflife2.fgd"
    fgd_path = r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\bin\swarm.fgd"
    ContentManager().scan_for_content(fgd_path)
    fgd: Fgd = FgdParse(fgd_path)
    processed_classes = []
    buffer = ''

    buffer += """
def parse_source_value(value):
    if type(value) is str:
        value: str
        if value.replace('.', '', 1).replace('-', '', 1).isdecimal():
            return float(value) if '.' in value else int(value)
        return 0
    else:
        return value


def parse_int_vector(string):
    return [parse_source_value(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    if string is None:
        return [0.0, 0.0, 0.0]
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


class Base:
    hammer_id_counter = 0

    def __init__(self, entity_data: dict):
        self.__hammer_id = -1
        self._raw_data = entity_data

    @classmethod
    def new_hammer_id(cls):
        new_id = cls.hammer_id_counter
        cls.hammer_id_counter += 1
        return new_id

    @property
    def class_name(self):
        return self._raw_data.get('classname')
        
    @property
    def hammer_id(self):
        if self.__hammer_id == -1:
            if 'hammerid' in self._raw_data:
                self.__hammer_id = int(self._raw_data.get('hammerid'))
            else:  # Titanfall
                self.__hammer_id = Base.new_hammer_id()
        return self.__hammer_id
\n\n"""

    entity_classes = fgd.entities
    include: Fgd
    # for include in fgd.includes:
    #     for entity in include.entities:
    #         if entity in entity_classes:
    #             entity_classes.pop(entity_classes.index(entity))
    entity_class: FgdEntity
    for entity_class in entity_classes:
        print(entity_class.name)
        if entity_class.name in processed_classes:
            continue
        buffer += f'class {entity_class.name}'

        all_parents = []
        for parent in entity_class.parents:
            all_parents += collect_parents(parent)
            all_parents.pop(all_parents.index(parent))
        all_parents = set(all_parents)

        if entity_class.parents:
            buffer += '('
            buffer += ', '.join(parent.name for parent in set(entity_class.parents) if parent not in all_parents)
            buffer += ')'
        else:
            buffer += '(Base)'
        buffer += ':\n'
        for definition in entity_class.definitions:
            if definition['name'] == 'iconsprite':
                buffer += f'    icon_sprite = {definition["args"][0]}\n'
            elif definition['name'] == 'studio' and definition['args']:
                buffer += f'    model = {definition["args"][0]}\n'
            elif definition['name'] == 'studioprop' and definition['args']:
                buffer += f'    viewport_model = {definition["args"][0]}\n'

        prop_cache = []
        for parent in list(all_parents) + entity_class.parents:
            for prop in parent.properties:
                if prop.name not in prop_cache:
                    prop_cache.append(prop.name)
        if entity_class.class_type == 'PointClass':
            buffer += f'''    @property\n    def origin(self):
        return parse_int_vector(self._raw_data.get('origin',"0 0 0"))
'''
        prop_cound = 0
        for prop in entity_class.properties:
            if prop.name not in prop_cache:
                buffer += f'\n    @property\n    def {prop.name}(self):\n        '
                try:
                    if prop.value_type == 'color255':
                        def_value = f'"{prop.default_value}"' if prop.default_value is not None else None
                        buffer += f'return parse_int_vector(self._raw_data.get(\'{prop.name.lower()}\', {def_value}))'
                    elif prop.value_type in ['angle', 'vector', 'color1', 'origin']:
                        def_value = f'"{prop.default_value}"' if prop.default_value is not None else None
                        buffer += f'return parse_float_vector(self._raw_data.get(\'{prop.name.lower()}\', {def_value}))'
                    elif prop.value_type in ['integer', 'float', 'node_dest', 'angle_negative_pitch', 'node_dest']:
                        buffer += f'return parse_source_value(self._raw_data.get(\'{prop.name.lower()}\', {prop.default_value}))'
                    elif prop.value_type == 'choices':
                        def_value = f'"{prop.default_value}"' if prop.default_value is not None else None
                        buffer += f'return self._raw_data.get(\'{prop.name.lower()}\', {def_value})'
                    elif prop.value_type in ['string', 'studio', 'material', 'sprite', 'sound']:
                        def_value = f'"{prop.default_value}"' if prop.default_value is not None else None
                        buffer += f'return self._raw_data.get(\'{prop.name.lower()}\', {def_value})'
                    elif prop.value_type == 'target_destination' and prop.default_value == 'Name of the entity to set navigation properties on.':
                        buffer += f'return self._raw_data.get(\'{prop.name.lower()}\', None)  # Set to none due to bug in BlackMesa base.fgd file'
                    elif prop.value_type == 'vecline' and prop.default_value == 'The position the rope attaches to object 2':
                        buffer += f'return self._raw_data.get(\'{prop.name.lower()}\', None)  # Set to none due to bug in BlackMesa base.fgd file'
                    else:
                        def_value = f'"{prop.default_value}"' if prop.default_value is not None else None
                        buffer += f'return self._raw_data.get(\'{prop.name.lower()}\', {def_value})'
                except ValueError as ex:
                    buffer += f'        # Failed to parse value type due to {ex}'
                prop_cound += 1
                buffer += '\n'

        if prop_cound == 0:
            buffer += '    pass'
        buffer += '\n\n\n'

        processed_classes.append(entity_class.name)
    buffer += '\nentity_class_handle = {'
    for entity_class in processed_classes:
        buffer += f'\n    \'{entity_class}\': {entity_class},'
    buffer += '\n}'
    # print(StandaloneContentManager().get_content_provider_from_path(fgd_path))
    output_name = TinyPath(fgd_path).stem
    with open(f'../../../blender_bindings/source1/bsp/entities/{output_name}_entity_classes.py', 'w') as f:
        f.write(buffer)


if __name__ == '__main__':
    main()
