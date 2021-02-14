from fgdtools import FgdParse, Fgd, FgdEntity, FgdEntityProperty

base_class = """from typing import Type

def parse_int_vector(value:str):
    return [int(val) for val in value.split(' ')]

def parse_float_vector(value:str):
    return [float(val) for val in value.split(' ')]


class Base:
    def __init__(self):
        self.hammer_id = '0'
        self.class_name = '0'
        
    def from_dict(self,entity_props):
        assert 'classname' in entity_props, 'Missing "classname" property'
        assert 'hammerid' in entity_props, 'Missing "hammerid" property'
        self.class_name = entity_props.get('classname')
        self.hammer_id = entity_props.get('hammerid')
        
        
"""
class_template = """class {class_name}({bases}):
    def __init__(self):
        super().__init__()
        for base in self.__class__.mro()[1:-1]:
            base.__init__(self)
        {properties}
    
    def from_dict(self, entity_props):
        base: Type[Base]
        for base in self.__class__.mro()[1:-1]:
            base.from_dict(self, entity_props)
        {property_init}
        
        
"""

property_template = '\n\t\t"""{display_name}\n\t\t{description}\n\t\tType:{type}"""\n\t\tself.{name} = {default_value}'
property_init_template = 'self.{name} = entity_props.get(\'{name2}\', {default_value})'

classname_remap = {}


def collect_parents(parent: FgdEntity):
    parents = []
    for pparent in parent.parents:
        parents += collect_parents(pparent)
    return [classname_remap[parent.name]] + parents


if __name__ == '__main__':
    fgd_data: Fgd = FgdParse(r"H:\SteamLibrary\SteamApps\common\Team Fortress 2\bin\tf.fgd")
    classes_data = base_class
    entity_class: FgdEntity

    all_entities = fgd_data.entities
    for inc in fgd_data.includes:
        all_entities += inc.entities

    for entity_class in fgd_data.entities:
        class_name = entity_class.name  # ''.join([p.capitalize() for p in entity_class.name.split('_')])
        classname_remap[entity_class.name] = class_name
        prop: FgdEntityProperty
        class_props = []
        class_props_init = []
        for prop in entity_class.properties:
            parent: FgdEntity
            skip = False
            for parent in entity_class.parents:
                for pprop in parent.properties:
                    if prop.name == pprop.name:
                        skip = True
                        break
                if skip:
                    break
            if skip:
                continue
            default_value = prop.default_value
            if prop.description is None:
                default_value = None
            elif prop.value_type in ['angle', 'origin', 'color255', 'vector']:
                default_value = '[{}, {}, {}]'.format(*(prop.default_value or '0 0 0').split(' '))
            elif prop.value_type in ['string', 'sprite', 'choices', 'sound', 'material', 'studio']:
                default_value = f'"{prop.default_value}"'
            else:
                print(prop.value_type)
            prop_name = prop.name
            if prop_name == 'class':
                prop_name = 'class_'
            prop_data = property_template.format(display_name=prop.display_name, description=prop.description,
                                                 name=prop_name, default_value=default_value or 'None',
                                                 type=prop.value_type)
            class_props_init_data = property_init_template.format(name=prop_name, name2=prop.name,
                                                                  default_value=default_value or 'None')
            class_props.append(prop_data)
            class_props_init.append(class_props_init_data)

        if entity_class.parents:
            skip = False
            parents = []
            for parent in entity_class.parents:
                parents += collect_parents(parent)

            bases = []
            for parent in entity_class.parents:
                if classname_remap[parent.name] not in parents and classname_remap[parent.name] not in bases:
                    bases.append(classname_remap[parent.name])

        else:
            bases = ['Base']

        class_data = class_template.format(class_name=class_name,
                                           bases=', '.join(bases),
                                           properties='\n\t\t'.join(class_props),
                                           property_init='\n\t\t'.join(class_props_init))
        classes_data += class_data

    class_map = """\nclass_map = {{
{}
}}""".format(',\n'.join([f'\t"{ent.name}": {ent.name}' for ent in fgd_data.entities]))
    classes_data += class_map
    with open('./tf2_class_dump.py', 'w') as f:
        f.write(classes_data.replace('\t', ' ' * 4))
