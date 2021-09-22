from typing import List


class FGDProperty:
    def __init__(self, name, value_type, display_name=None, default_value=None, description=None, meta=None):
        self._name = name
        self.value_type = value_type
        self._display_name = display_name
        self._default_value = default_value
        self._description = description
        self._meta = meta

    @property
    def name(self):
        return self._name.replace('.', '_')

    def __str__(self):
        return f"{self.name}({self.value_type})"

    @property
    def meta_data(self):
        return self._meta or {}

    @property
    def readonly(self):
        if self._meta:
            return self._meta.get('readonly', False)
        return False

    @property
    def description(self):
        return self._description or ""

    @property
    def display_name(self):
        return self._display_name or ""

    @property
    def default_value(self):
        return self._default_value

    def parser_code(self):
        buffer: List[str] = ['@property', f'def {self.name}(self):', f'\tif \"{self.name}\" in self._entity_data:']
        value_type = self.value_type.lower()
        if value_type == 'float':
            buffer.append(f"\t\treturn float(self._entity_data.get('{self.name}'))")
            buffer.append(f'\treturn float({self.default_value})')
        elif value_type in ['integer', 'int', 'node_id', 'node_dest', 'lod_level']:
            buffer.append(f"\t\treturn int(self._entity_data.get('{self.name}'))")
            buffer.append(f'\treturn int({self.default_value})')
        elif value_type in ['string', 'parentAttachment', 'target_source',
                            'target_destination', 'choices', 'studio',
                            'materialgroup', 'bodygroupchoices', 'parentattachment',
                            'sequence', 'filterclass', 'sound', 'sprite',
                            'material', 'particle', 'resource:particle', 'decal',
                            'particlesystem', 'npcclass', 'localaxis',
                            'resource:model', 'instance_file', 'instance_variable',
                            'text_block', 'resource:material', 'resource:texture',
                            'surface_properties', 'target_name_or_class', 'scene',
                            'modelstatechoices', 'resource_choices:model', 'resource:postprocessing',
                            'remove_key', ]:
            buffer.append(f"\t\treturn self._entity_data.get('{self.name}')")
            if self.default_value is None:
                buffer.append(f'\treturn None')
            else:
                buffer.append(f'\treturn "{self.default_value}"')
        elif value_type in ['boolean', 'bool']:
            buffer.append(f"\t\treturn bool(self._entity_data.get('{self.name}'))")
            if type(self.default_value) is str:
                default = self.default_value.replace("No", "False").replace("Yes", "True")
                buffer.append(f'\treturn bool({default})')
            else:
                buffer.append(f'\treturn bool({self.default_value})')
        elif value_type == 'scriptlist':
            buffer.append(f"\t\treturn self._entity_data.get('{self.name}')")
            buffer.append(f'\treturn "{self.default_value}"')
        elif value_type in ['vector', 'angle', 'color255', 'local_point', 'vecline', 'sidelist', 'axis',
                            'node_id_list', 'world_point', 'curve']:
            buffer.append(f"\t\treturn parse_int_vector(self._entity_data.get('{self.name}'))")
            buffer.append(f'\treturn parse_int_vector("{self.default_value}")')
        else:
            raise NotImplementedError(f"Unsupported type:{value_type}")
            # buffer.append(f"\t\treturn entity_data.get('{self.name}')")
            # buffer.append(f'\treturn "{self.default_value}"')
        return buffer


class FGDChoiceProperty(FGDProperty):
    def __init__(self, name, value_type, display_name=None, default_value=None, description=None, meta=None,
                 choices=None):
        super().__init__(name, value_type, display_name, default_value, description, meta)
        self._choices = choices

    @property
    def choices(self):
        return self._choices


class FGDFlagProperty(FGDProperty):
    def __init__(self, name, value_type, display_name=None, default_value=None, description=None, meta=None,
                 flags=None):
        super().__init__(name, value_type, display_name, default_value, description, meta)
        self._flags = flags

    def parser_code(self):
        buffer: List[str] = ['@property',
                             f'def {self.name}(self):',
                             "\tflags = []",
                             f'\tif \"{self.name}\" in self._entity_data:',
                             f"\t\tvalue = self._entity_data.get(\"{self.name}\",{self.default_value})"]
        keys = self._flags
        buffer.append(f"\t\tfor name,(key,_) in {keys}.items():")
        buffer.append(f'\t\t\tif value&key>0:')
        buffer.append(f'\t\t\t\tflags.append(name)')
        buffer.append(f'\treturn flags')
        return buffer

    @property
    def flags(self):
        return self._flags


class FGDTagProperty(FGDFlagProperty):
    def __init__(self, name, value_type, display_name=None, default_value=None, description=None, meta=None,
                 flags=None):
        super().__init__(name, value_type, display_name, default_value, description, meta)
        self._flags = flags

    @property
    def flags(self):
        return self._flags


class FGDFunction:
    def __init__(self, name, func_type, args, doc):
        self.name = name
        self.type = func_type
        self.args = args
        self.doc = doc

    def __str__(self):
        return f"{self.type} {self.name}({','.join(self.args)}) : \"{self.doc}\""


class FGDEntity:

    def __init__(self, class_type, name, definitions=None, description=None,
                 properties=None, io=None):
        self.name: str = name
        self.class_type: str = class_type
        self._definitions: List = definitions
        self._description: str = description
        self._properties = properties
        self._io: List = io

    def __str__(self):
        return f"{self.class_type}({self.name})"

    def _find_parent_class(self, class_name, list_of_classes: List['FGDEntity']):
        for c in list_of_classes:
            if c.name == class_name:
                return c
        return None

    def _gather_all_bases(self, base_name, list_of_classes):
        bases = []
        cls = self._find_parent_class(base_name, list_of_classes)
        if cls:
            for base in cls.bases:
                bases.extend(self._gather_all_bases(base, list_of_classes))
                bases.append(base)
        return bases

    def parser_code(self, list_of_clases: List['FGDEntity']):
        buffer = ''
        if self.bases:
            existing_bases = []
            for base in self.bases:
                existing_bases += self._gather_all_bases(base, list_of_clases)

            buffer += f'class {self.name}({", ".join([base for base in self.bases if base not in existing_bases])}):\n'
        else:
            buffer += f'class {self.name}:\n'
        buffer += '\tpass\n\n'

        for name, definition in self.definitions:
            if not definition:
                continue
            if name == 'iconsprite':
                buffer += f'\ticon_sprite = "{definition[0]}"\n'
            elif name == 'studio':
                buffer += f'\t_model = "{definition[0]}"\n'
        if not self.bases:
            buffer += f'\tdef __init__(self, entity_data:dict):\n'
            buffer += f'\t\tself._entity_data = entity_data\n\n'
        for prop in self.properties:
            for line in prop.parser_code():
                buffer += '\t' + line + '\n'
            buffer += '\n'
        buffer += '\n'
        return buffer

    @property
    def definitions(self):
        return self._definitions or []

    @property
    def bases(self):
        for name, definition in self._definitions:
            if name == 'base':
                return definition
        return []

    @property
    def metadata(self):
        for name, definition in self._definitions:
            if name == 'metadata':
                return definition
        return []

    @property
    def description(self):
        return self._description or ""

    @property
    def properties(self):
        props = []
        if self._properties:
            for prop_data in self._properties:
                prop_name = prop_data['name']
                prop_type = prop_data['type']
                prop_meta = prop_data['meta']
                prop_dname = prop_data.get('display_name', None)
                prop_default = prop_data.get('default', None)
                prop_doc = prop_data.get('doc', None)
                if 'choices' in prop_type.lower():
                    prop = FGDChoiceProperty(prop_name, prop_type, prop_dname,
                                             prop_default, prop_doc, prop_meta,
                                             prop_data.get('choices', []))
                    props.append(prop)
                elif 'flags' in prop_type.lower():
                    prop = FGDFlagProperty(prop_name, prop_type, prop_dname,
                                           prop_default, prop_doc, prop_meta,
                                           prop_data['flags'])
                    props.append(prop)
                elif 'tag_list' in prop_type.lower():
                    prop = FGDTagProperty(prop_name, prop_type, prop_dname,
                                          prop_default, prop_doc, prop_meta,
                                          prop_data['tag_list'])
                    props.append(prop)
                else:
                    prop = FGDProperty(prop_name, prop_type, prop_dname,
                                       prop_default, prop_doc, prop_meta)
                    props.append(prop)
                pass
        return props

    @property
    def inputs(self):
        inputs = []
        if self._io:
            for io in self._io:
                if io['type'] != 'input':
                    continue
                inputs.append(FGDFunction(io['name'], io['type'], io['args'], io['doc']))
                pass
        return inputs

    @property
    def output(self):
        inputs = []
        if self._io:
            for io in self._io:
                if io['type'] != 'output':
                    continue
                inputs.append(FGDFunction(io['name'], io['type'], io['args'], io['doc']))
        return inputs

    def override(self, definitions, doc, props, io):
        for new_def in definitions:
            for n, old_def in enumerate(self._definitions):
                if old_def[0] == new_def[0]:
                    self._definitions[n] = new_def
        for new_io in io:
            for n, old_io in enumerate(self._io):
                if new_io['name'] == old_io['name']:
                    self._io[n].update(new_io)
        for new_prop in props:
            for n, old_prop in enumerate(self._properties):
                if new_prop['name'] == old_prop['name']:
                    self._properties[n].update(new_prop)

        self._description = doc
