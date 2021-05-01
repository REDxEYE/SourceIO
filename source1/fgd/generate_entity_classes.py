from pathlib import Path

from valvefgd import FgdEntity, FgdParse, FgdEntityProperty, Fgd
import os

os.environ['NO_BPY'] = '1'
from SourceIO.source_shared.content_manager import ContentManager


def parse_int_vector(string):
    return [int(val) for val in string.replace('  ', ' ').split(' ')]


def parse_float_vector(string):
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


def collect_parents(parent: FgdEntity):
    parents = []
    for pparent in parent.parents:
        parents += collect_parents(pparent)
    return [parent] + parents


def main():
    # fgd_path = r"F:\SteamLibrary\steamapps\common\Black Mesa\bin\bms.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\Team Fortress 2\bin\base.fgd"
    # fgd_path = r"D:\SteamLibrary\steamapps\common\Portal\bin\portal.fgd"
    fgd_path = r"D:\SteamLibrary\steamapps\common\Counter-Strike Global Offensive\bin\csgo.fgd"
    # fgd_path = r"H:\SteamLibrary\steamapps\common\Portal 2\bin\portal2.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\Left 4 Dead 2\bin\left4dead2.fgd"
    # fgd_path = r"F:\SteamLibrary\steamapps\common\Half-Life 2\bin\halflife2.fgd"
    # fgd_path = r"H:\SteamLibrary\SteamApps\common\SourceFilmmaker\game\bin\swarm.fgd"
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
    return [float(val) for val in string.replace('  ', ' ').split(' ')]


class Base:
    hammer_id_counter = 0

    def __init__(self):
        self.hammer_id = 0
        self.class_name = 'ANY'

    @classmethod
    def new_hammer_id(cls):
        new_id = cls.hammer_id_counter
        cls.hammer_id_counter += 1
        return new_id

    @staticmethod
    def from_dict(instance, entity_data: dict):
        if 'hammerid' in entity_data:
            instance.hammer_id = int(entity_data.get('hammerid'))
        else:  # Titanfall
            instance.hammer_id = Base.new_hammer_id()
        instance.class_name = entity_data.get('classname')
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

        buffer += '    def __init__(self):'

        if entity_class.parents:
            for parent in sorted(set(entity_class.parents), key=lambda a: len(a.parents), reverse=True):
                if parent.name not in all_parents:
                    buffer += f'\n        super({parent.name}).__init__()'
        else:
            buffer += '\n        super().__init__()'

        prop_cache = []
        for parent in list(all_parents) + entity_class.parents:
            for prop in parent.properties:
                if prop.name not in prop_cache:
                    prop_cache.append(prop.name)
        default_prop_cache = {}
        if entity_class.class_type == 'PointClass':
            buffer += f'\n        self.origin = [0, 0, 0]'
        for prop in entity_class.properties:
            if prop.name not in prop_cache:
                buffer += f'\n        self.{prop.name} = '
                if prop.default_value:  # and prop.description is not None:
                    try:
                        if prop.value_type == 'color255':
                            def_value = f'[{", ".join(map(str, parse_int_vector(prop.default_value)))}]'
                            default_prop_cache[prop.name] = def_value
                            buffer += def_value
                        elif prop.value_type in ['angle', 'vector', 'color1']:
                            def_value = f'[{", ".join(map(str, parse_float_vector(prop.default_value)))}]'
                            default_prop_cache[prop.name] = def_value
                            buffer += def_value
                        elif prop.value_type in ['integer', 'float', 'node_dest', 'angle_negative_pitch']:
                            def_value = f'{prop.default_value}'
                            default_prop_cache[prop.name] = def_value
                            buffer += def_value
                        elif prop.value_type == 'choices':
                            def_value = '"CHOICES NOT SUPPORTED"'
                            default_prop_cache[prop.name] = def_value
                            buffer += def_value
                        elif prop.value_type in ['string', 'studio', 'material', 'sprite', 'sound']:
                            def_value = f'"{prop.default_value}"'
                            default_prop_cache[prop.name] = def_value
                            buffer += def_value
                        elif prop.value_type == 'target_destination' and prop.default_value == 'Name of the entity to set navigation properties on.':
                            buffer += 'None  # Set to none due to bug in BlackMesa base.fgd file'
                        elif prop.value_type == 'vecline' and prop.default_value == 'The position the rope attaches to object 2':
                            buffer += 'None  # Set to none due to bug in BlackMesa base.fgd file'
                        else:
                            buffer += str(prop.default_value)
                    except ValueError as ex:
                        buffer += f'None  # Failed to parse value type due to {ex}'
                    buffer += f'  # Type: {prop.value_type}'
                else:
                    buffer += f'None  # Type: {prop.value_type}'

        if not entity_class.properties:
            buffer += '\n        pass'

        buffer += '\n\n    @staticmethod'
        buffer += '\n    def from_dict(instance, entity_data: dict):'
        if entity_class.parents:
            for parent in set(entity_class.parents):
                if parent.name not in all_parents:
                    buffer += f'\n        {parent.name}.from_dict(instance, entity_data)'
        else:
            buffer += f'\n        Base.from_dict(instance, entity_data)'

        if entity_class.class_type == 'PointClass':
            buffer += f'\n        instance.origin = parse_float_vector(entity_data.get(\'origin\', "0 0 0"))'

        for prop in entity_class.properties:
            if prop.name not in prop_cache:
                prefix = '\n        '
                assigment = prefix + f'instance.{prop.name} = '
                try:
                    if prop.value_type == 'color255':
                        buffer += assigment + f'parse_int_vector(entity_data.get(\'{prop.name.lower()}\', "{prop.default_value or "0 0 0"}"))'
                    elif prop.value_type in ['angle', 'vector', 'color1', 'origin']:
                        buffer += assigment + f'parse_float_vector(entity_data.get(\'{prop.name.lower()}\', "{prop.default_value or "0 0 0"}"))'
                    elif prop.value_type in ['integer', 'node_dest']:
                        buffer += assigment + f'parse_source_value(entity_data.get(\'{prop.name.lower()}\', {default_prop_cache.get(prop.name, 0)}))'
                    elif prop.value_type in ['float', 'angle_negative_pitch']:
                        buffer += assigment + f'float(entity_data.get(\'{prop.name.lower()}\', {default_prop_cache.get(prop.name, 0)}))'
                    elif prop.value_type == 'choices':
                        buffer += assigment + f'entity_data.get(\'{prop.name.lower()}\', {default_prop_cache.get(prop.name, "None")})'
                    elif prop.value_type in ['string', 'studio', 'material', 'sprite', 'sound']:
                        buffer += assigment + f'entity_data.get(\'{prop.name.lower()}\', {default_prop_cache.get(prop.name, "None")})'
                    elif prop.value_type == 'target_destination' and prop.default_value == 'Name of the entity to set navigation properties on.':
                        buffer += assigment + f'entity_data.get(\'{prop.name.lower()}\', None)  # Set to none due to bug in BlackMesa base.fgd file'
                    elif prop.value_type == 'vecline' and prop.default_value == 'The position the rope attaches to object 2':
                        buffer += assigment + f'entity_data.get(\'{prop.name.lower()}\', None)  # Set to none due to bug in BlackMesa base.fgd file'
                    else:
                        buffer += assigment + f'entity_data.get(\'{prop.name.lower()}\', {default_prop_cache.get(prop.name, "None")})'
                except ValueError as ex:
                    buffer += f'None  # Failed to parse value type due to {ex}'
                buffer += f'  # Type: {prop.value_type}'
        buffer += '\n\n\n'

        processed_classes.append(entity_class.name)
    buffer += '\nentity_class_handle = {'
    for entity_class in processed_classes:
        buffer += f'\n    \'{entity_class}\': {entity_class},'
    buffer += '\n}'
    # print(ContentManager().get_content_provider_from_path(fgd_path))
    output_name = Path(fgd_path).stem
    with open(f'../bsp/entities/{output_name}_entity_classes.py', 'w') as f:
        f.write(buffer)


if __name__ == '__main__':
    main()
