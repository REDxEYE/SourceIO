# -*- coding: utf-8 -*-
import collections
import functools
import pickle
try:
    from dictionaries import ReadonlyDictProxy
except ImportError:
    from .dictionaries import ReadonlyDictProxy
__all__ = ['Flags', 'FlagsMeta', 'FlagData', 'UNDEFINED', 'unique', 'unique_bits']


# version_info[0]: Increase in case of large milestones/releases.
# version_info[1]: Increase this and zero out version_info[2] if you have explicitly modified
#                  a previously existing behavior/interface.
#                  If the behavior of an existing feature changes as a result of a bugfix
#                  and the new (bugfixed) behavior is that meets the expectations of the
#                  previous interface documentation then you shouldn't increase this, in that
#                  case increase only version_info[2].
# version_info[2]: Increase in case of bugfixes. Also use this if you added new features
#                  without modifying the behavior of the previously existing ones.
version_info = (1, 1, 2)
__version__ = '.'.join(str(n) for n in version_info)
__author__ = 'István Pásztor'
__license__ = 'MIT'


def unique(flags_class):
    """ A decorator for flags classes to forbid flag aliases. """
    if not is_flags_class_final(flags_class):
        raise TypeError('unique check can be applied only to flags classes that have members')
    if not flags_class.__member_aliases__:
        return flags_class
    aliases = ', '.join('%s -> %s' % (alias, name) for alias, name in flags_class.__member_aliases__.items())
    raise ValueError('duplicate values found in %r: %s' % (flags_class, aliases))


def unique_bits(flags_class):
    """ A decorator for flags classes to forbid declaring flags with overlapping bits. """
    flags_class = unique(flags_class)
    other_bits = 0
    for name, member in flags_class.__members_without_aliases__.items():
        bits = int(member)
        if other_bits & bits:
            for other_name, other_member in flags_class.__members_without_aliases__.items():
                if int(other_member) & bits:
                    raise ValueError("%r: '%s' and '%s' have overlapping bits" % (flags_class, other_name, name))
        else:
            other_bits |= bits


def is_descriptor(obj):
    return hasattr(obj, '__get__') or hasattr(obj, '__set__') or hasattr(obj, '__delete__')


class Const:
    def __init__(self, name):
        self.__name = name

    def __repr__(self):
        return self.__name


# "singleton" to be used as a const value with identity checks
UNDEFINED = Const('UNDEFINED')


def create_flags_subclass(base_enum_class, class_name, flags, *, mixins=(), module=None, qualname=None,
                          no_flags_name=UNDEFINED, all_flags_name=UNDEFINED):
    meta_class = type(base_enum_class)
    bases = tuple(mixins) + (base_enum_class,)
    class_dict = {'__members__': flags}
    if no_flags_name is not UNDEFINED:
        class_dict['__no_flags_name__'] = no_flags_name
    if all_flags_name is not UNDEFINED:
        class_dict['__all_flags_name__'] = all_flags_name
    flags_class = meta_class(class_name, bases, class_dict)

    # disabling on enabling pickle on the new class based on our module parameter
    if module is None:
        # Making the class unpicklable.
        def disabled_reduce_ex(self, proto):
            raise pickle.PicklingError("'%s' is unpicklable" % (type(self).__name__,))
        flags_class.__reduce_ex__ = disabled_reduce_ex

        # For pickle module==None means the __main__ module so let's change it to a non-existing name.
        # This will cause a failure while trying to pickle the class.
        module = '<unknown>'
    flags_class.__module__ = module

    if qualname is not None:
        flags_class.__qualname__ = qualname

    return flags_class


def process_inline_members_definition(members):
    """
    :param members: this can be any of the following:
    - a string containing a space and/or comma separated list of names: e.g.:
      "item1 item2 item3" OR "item1,item2,item3" OR "item1, item2, item3"
    - tuple/list/Set of strings (names)
    - Mapping of (name, data) pairs
    - any kind of iterable that yields (name, data) pairs
    :return: An iterable of (name, data) pairs.
    """
    if isinstance(members, str):
        members = ((name, UNDEFINED) for name in members.replace(',', ' ').split())
    elif isinstance(members, (tuple, list, collections.Set)):
        if members and isinstance(next(iter(members)), str):
            members = ((name, UNDEFINED) for name in members)
    elif isinstance(members, collections.Mapping):
        members = members.items()
    return members


def is_member_definition_class_attribute(name, value):
    """ Returns True if the given class attribute with the specified
    name and value should be treated as a flag member definition. """
    return not name.startswith('_') and not is_descriptor(value)


def extract_member_definitions_from_class_attributes(class_dict):
    members = [(name, value) for name, value in class_dict.items()
               if is_member_definition_class_attribute(name, value)]
    for name, _ in members:
        del class_dict[name]

    members.extend(process_inline_members_definition(class_dict.pop('__members__', ())))
    return members


class ReadonlyzerMixin:
    """ Makes instance attributes readonly after setting readonly=True. """
    __slots__ = ('__readonly',)

    def __init__(self, *args, readonly=False, **kwargs):
        # Calling super() before setting readonly.
        # This way super().__init__ can set attributes even if readonly==True
        super().__init__(*args, **kwargs)
        self.__readonly = readonly

    @property
    def readonly(self):
        try:
            return self.__readonly
        except AttributeError:
            return False

    @readonly.setter
    def readonly(self, value):
        self.__readonly = value

    def __setattr__(self, key, value):
        if self.readonly:
            raise AttributeError("Can't set attribute '%s' of readonly '%s' object" % (key, type(self).__name__))
        super().__setattr__(key, value)

    def __delattr__(self, key):
        if self.readonly:
            raise AttributeError("Can't delete attribute '%s' of readonly '%s' object" % (key, type(self).__name__))
        super().__delattr__(key)


class FlagProperties(ReadonlyzerMixin):
    __slots__ = ('name', 'data', 'bits', 'index', 'index_without_aliases')

    def __init__(self, *, name, bits, data=None, index=None, index_without_aliases=None):
        self.name = name
        self.data = data
        self.bits = bits
        self.index = index
        self.index_without_aliases = index_without_aliases
        super().__init__()


READONLY_PROTECTED_FLAGS_CLASS_ATTRIBUTES = frozenset([
    '__writable_protected_flags_class_attributes__', '__all_members__', '__members__', '__members_without_aliases__',
    '__member_aliases__', '__bits_to_properties__', '__bits_to_instance__', '__pickle_int_flags__',
])

# these attributes are writable when __writable_protected_flags_class_attributes__ is set to True on the class.
TEMPORARILY_WRITABLE_PROTECTED_FLAGS_CLASS_ATTRIBUTES = frozenset([
    '__all_bits__', '__no_flags__', '__all_flags__', '__no_flags_name__', '__all_flags_name__',
])

PROTECTED_FLAGS_CLASS_ATTRIBUTES = READONLY_PROTECTED_FLAGS_CLASS_ATTRIBUTES | \
                                   TEMPORARILY_WRITABLE_PROTECTED_FLAGS_CLASS_ATTRIBUTES


def is_valid_bits_value(bits):
    return isinstance(bits, int) and not isinstance(bits, bool)


def initialize_class_dict_and_create_flags_class(class_dict, class_name, create_flags_class):
    # all_members is used by __getattribute__ and __setattr__. It contains all items
    # from members and also the no_flags and all_flags special members if they are defined.
    all_members = collections.OrderedDict()
    members = collections.OrderedDict()
    members_without_aliases = collections.OrderedDict()
    bits_to_properties = collections.OrderedDict()
    bits_to_instance = collections.OrderedDict()
    member_aliases = collections.OrderedDict()
    class_dict['__all_members__'] = ReadonlyDictProxy(all_members)
    class_dict['__members__'] = ReadonlyDictProxy(members)
    class_dict['__members_without_aliases__'] = ReadonlyDictProxy(members_without_aliases)
    class_dict['__bits_to_properties__'] = ReadonlyDictProxy(bits_to_properties)
    class_dict['__bits_to_instance__'] = ReadonlyDictProxy(bits_to_instance)
    class_dict['__member_aliases__'] = ReadonlyDictProxy(member_aliases)

    flags_class = create_flags_class(class_dict)

    def instantiate_member(name, bits, special):
        if not isinstance(name, str):
            raise TypeError('Flag name should be an str but it is %r' % (name,))
        if not is_valid_bits_value(bits):
            raise TypeError("Bits for flag '%s' should be an int but it is %r" % (name, bits))
        if not special and bits == 0:
            raise ValueError("Flag '%s' has the invalid value of zero" % name)
        member = flags_class(bits)
        if int(member) != bits:
            raise RuntimeError("%s has altered the assigned bits of member '%s' from %r to %r" % (
                class_name, name, bits, int(member)))
        return member

    def register_member(member, name, bits, data, special):
        # special members (like no_flags, and all_flags) have no index
        # and they appear only in the __all_members__ collection.
        if all_members.setdefault(name, member) is not member:
            raise ValueError('Duplicate flag name: %r' % name)

        # It isn't a problem if an instance with the same bits already exists in bits_to_instance because
        # a member contains only the bits so our new member is equivalent with the replaced one.
        bits_to_instance[bits] = member

        if special:
            return

        members[name] = member
        properties = FlagProperties(name=name, bits=bits, data=data, index=len(members))
        properties_for_bits = bits_to_properties.setdefault(bits, properties)
        is_alias = properties_for_bits is not properties
        if is_alias:
            if data is not UNDEFINED:
                raise ValueError("You aren't allowed to associate data with alias '%s'" % name)
            member_aliases[name] = properties_for_bits.name
        else:
            properties.index_without_aliases = len(members_without_aliases)
            members_without_aliases[name] = member
        properties.readonly = True

    def instantiate_and_register_member(*, name, bits, data=None, special_member=False):
        member = instantiate_member(name, bits, special_member)
        register_member(member, name, bits, data, special_member)
        return member

    return flags_class, instantiate_and_register_member


def create_flags_class_with_members(class_name, class_dict, member_definitions, create_flags_class):
    class_dict['__writable_protected_flags_class_attributes__'] = True

    flags_class, instantiate_and_register_member = initialize_class_dict_and_create_flags_class(
        class_dict, class_name, create_flags_class)

    member_definitions = [(name, data) for name, data in member_definitions]
    member_definitions = flags_class.process_member_definitions(member_definitions)
    # member_definitions has to be an iterable of iterables yielding (name, bits, data)

    all_bits = 0
    for name, bits, data in member_definitions:
        instantiate_and_register_member(name=name, bits=bits, data=data)
        all_bits |= bits

    if len(flags_class) == 0:
        # In this case process_member_definitions() returned an empty iterable which isn't allowed.
        raise RuntimeError("%s.%s returned an empty iterable" %
                           (flags_class.__name__, flags_class.process_member_definitions.__name__))

    def instantiate_special_member(name, default_name, bits):
        name = default_name if name is None else name
        return instantiate_and_register_member(name=name, bits=bits, special_member=True)

    flags_class.__no_flags__ = instantiate_special_member(flags_class.__no_flags_name__, '__no_flags__', 0)
    flags_class.__all_flags__ = instantiate_special_member(flags_class.__all_flags_name__, '__all_flags__', all_bits)

    flags_class.__all_bits__ = all_bits

    del flags_class.__writable_protected_flags_class_attributes__
    return flags_class


class FlagData:
    pass


def is_flags_class_final(flags_class):
    return hasattr(flags_class, '__members__')


class FlagsMeta(type):
    def __new__(mcs, class_name, bases, class_dict):
        if '__slots__' in class_dict:
            raise RuntimeError("You aren't allowed to use __slots__ in your Flags subclasses")
        class_dict['__slots__'] = ()

        def create_flags_class(custom_class_dict=None):
            return super(FlagsMeta, mcs).__new__(mcs, class_name, bases, custom_class_dict or class_dict)

        if Flags is None:
            # This __new__ call is creating the Flags class of this module.
            return create_flags_class()

        flags_bases = [base for base in bases if issubclass(base, Flags)]
        for base in flags_bases:
            # pylint: disable=protected-access
            if is_flags_class_final(base):
                raise RuntimeError("You can't subclass '%s' because it has already defined flag members" %
                                   (base.__name__,))

        member_definitions = extract_member_definitions_from_class_attributes(class_dict)
        if not member_definitions:
            return create_flags_class()
        return create_flags_class_with_members(class_name, class_dict, member_definitions, create_flags_class)

    def __call__(cls, *args, **kwargs):
        if kwargs or len(args) >= 2:
            # The Flags class or one of its subclasses was "called" as a
            # utility function to create a subclass of the called class.
            return create_flags_subclass(cls, *args, **kwargs)

        # We have zero or one positional argument and we have to create and/or return an exact instance of cls.
        # 1. Zero argument means we have to return a zero flag.
        # 2. A single positional argument can be one of the following cases:
        #    1. An object whose class is exactly cls.
        #    2. An str object that comes from Flags.__str__() or Flags.to_simple_str()
        #    3. An int object that specifies the bits of the Flags instance to be created.

        if not is_flags_class_final(cls):
            raise RuntimeError("Instantiation of abstract flags class '%s.%s' isn't allowed." % (
                cls.__module__, cls.__name__))

        if not args:
            # case 1 - zero positional arguments, we have to return a zero flag
            return cls.__no_flags__

        value = args[0]

        if type(value) is cls:
            # case 2.1
            return value

        if isinstance(value, str):
            # case 2.2
            bits = cls.bits_from_str(value)
        elif is_valid_bits_value(value):
            # case 2.3
            bits = cls.__all_bits__ & value
        else:
            raise TypeError("Can't instantiate flags class '%s' from value %r" % (cls.__name__, value))

        instance = cls.__bits_to_instance__.get(bits)
        if instance:
            return instance
        return super().__call__(bits)

    @classmethod
    def __prepare__(cls, class_name, bases):
        return collections.OrderedDict()

    def __delattr__(cls, name):
        if (name in PROTECTED_FLAGS_CLASS_ATTRIBUTES and name != '__writable_protected_flags_class_attributes__') or\
                (name in getattr(cls, '__all_members__', {})):
            raise AttributeError("Can't delete protected attribute '%s'" % name)
        super().__delattr__(name)

    def __setattr__(cls, name, value):
        if name in PROTECTED_FLAGS_CLASS_ATTRIBUTES:
            if name in READONLY_PROTECTED_FLAGS_CLASS_ATTRIBUTES or\
                    not getattr(cls, '__writable_protected_flags_class_attributes__', False):
                raise AttributeError("Can't assign protected attribute '%s'" % name)
        elif name in getattr(cls, '__all_members__', {}):
            raise AttributeError("Can't assign protected attribute '%s'" % name)
        super().__setattr__(name, value)

    def __getattr__(cls, name):
        try:
            return super().__getattribute__('__all_members__')[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(cls, name):
        return cls.__all_members__[name]

    def __iter__(cls):
        return iter(cls.__members_without_aliases__.values())

    def __reversed__(cls):
        return reversed(list(cls.__members_without_aliases__.values()))

    def __bool__(cls):
        return True

    def __len__(cls):
        members = getattr(cls, '__members_without_aliases__', ())
        return len(members)

    def flag_attribute_value_to_bits_and_data(cls, name, value):
        if value is UNDEFINED:
            return UNDEFINED, UNDEFINED
        elif isinstance(value, FlagData):
            return UNDEFINED, value
        elif is_valid_bits_value(value):
            return value, UNDEFINED
        elif isinstance(value, collections.Iterable):
            arr = tuple(value)
            if len(arr) == 0:
                return UNDEFINED, UNDEFINED
            if len(arr) == 1:
                return UNDEFINED, arr[0]
            if len(arr) == 2:
                return arr
            raise ValueError("Iterable is expected to have at most 2 items instead of %s "
                             "for flag '%s', iterable: %r" % (len(arr), name, value))
        raise TypeError("Expected an int or an iterable of at most 2 items "
                        "for flag '%s', received %r" % (name, value))

    def process_member_definitions(cls, member_definitions):
        """
        The incoming member_definitions contains the class attributes (with their values) that are
        used to define the flag members. This method can do anything to the incoming list and has to
        return a final set of flag definitions that assigns bits to the members. The returned member
        definitions can be completely different or unrelated to the incoming ones.
        :param member_definitions: A list of (name, data) tuples.
        :return: An iterable of iterables yielding 3 items: name, bits, data
        """
        members = []
        auto_flags = []
        all_bits = 0
        for name, data in member_definitions:
            bits, data = cls.flag_attribute_value_to_bits_and_data(name, data)
            if bits is UNDEFINED:
                auto_flags.append(len(members))
                members.append((name, data))
            elif is_valid_bits_value(bits):
                all_bits |= bits
                members.append((name, bits, data))
            else:
                raise TypeError("Expected an int value as the bits of flag '%s', received %r" % (name, bits))

        # auto-assigning unused bits to members without custom defined bits
        bit = 1
        for index in auto_flags:
            while bit & all_bits:
                bit <<= 1
            name, data = members[index]
            members[index] = name, bit, data
            bit <<= 1

        return members

    def __repr__(cls):
        return "<flags %s>" % cls.__name__

    __no_flags_name__ = 'no_flags'
    __all_flags_name__ = 'all_flags'
    __dotted_single_flag_str__ = True
    __pickle_int_flags__ = False
    __all_bits__ = -1

    # TODO: utility method to fill the flag members to a namespace, and another utility that can fill
    # them to a module (a specific case of namespaces)


def operator_requires_type_identity(wrapped):
    @functools.wraps(wrapped)
    def wrapper(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return wrapped(self, other)
    return wrapper


class FlagsArithmeticMixin:
    __slots__ = ('__bits',)

    def __new__(cls, bits):
        instance = super().__new__(cls)
        # pylint: disable=protected-access
        instance.__bits = bits & cls.__all_bits__
        return instance

    def __int__(self):
        return self.__bits

    def __bool__(self):
        return self.__bits != 0

    def __contains__(self, item):
        if type(item) is not type(self):
            return False
        # this logic is equivalent to that of __ge__(self, item) and __le__(item, self)
        # pylint: disable=protected-access
        return item.__bits == (self.__bits & item.__bits)

    def is_disjoint(self, *flags_instances):
        for flags in flags_instances:
            if self & flags:
                return False
        return True

    def __create_flags_instance(self, bits):
        # optimization, exploiting immutability
        if bits == self.__bits:
            return self
        return type(self)(bits)

    @operator_requires_type_identity
    def __or__(self, other):
        # pylint: disable=protected-access
        return self.__create_flags_instance(self.__bits | other.__bits)

    @operator_requires_type_identity
    def __xor__(self, other):
        # pylint: disable=protected-access
        return self.__create_flags_instance(self.__bits ^ other.__bits)

    @operator_requires_type_identity
    def __and__(self, other):
        # pylint: disable=protected-access
        return self.__create_flags_instance(self.__bits & other.__bits)

    @operator_requires_type_identity
    def __sub__(self, other):
        # pylint: disable=protected-access
        bits = self.__bits ^ (self.__bits & other.__bits)
        return self.__create_flags_instance(bits)

    @operator_requires_type_identity
    def __eq__(self, other):
        # pylint: disable=protected-access
        return self.__bits == other.__bits

    @operator_requires_type_identity
    def __ne__(self, other):
        # pylint: disable=protected-access
        return self.__bits != other.__bits

    @operator_requires_type_identity
    def __ge__(self, other):
        # pylint: disable=protected-access
        return other.__bits == (self.__bits & other.__bits)

    @operator_requires_type_identity
    def __gt__(self, other):
        # pylint: disable=protected-access
        return (self.__bits != other.__bits) and (other.__bits == (self.__bits & other.__bits))

    @operator_requires_type_identity
    def __le__(self, other):
        # pylint: disable=protected-access
        return self.__bits == (self.__bits & other.__bits)

    @operator_requires_type_identity
    def __lt__(self, other):
        # pylint: disable=protected-access
        return (self.__bits != other.__bits) and (self.__bits == (self.__bits & other.__bits))

    def __invert__(self):
        return self.__create_flags_instance(self.__bits ^ type(self).__all_bits__)


# This is used by FlagsMeta to detect whether the flags class currently being created is Flags.
Flags = None


class Flags(FlagsArithmeticMixin, metaclass=FlagsMeta):
    @property
    def is_member(self):
        """ `flags.is_member` is a shorthand for `flags.properties is not None`.
        If this property is False then this Flags instance has either zero bits or holds a combination
        of flag member bits.
        If this property is True then the bits of this Flags instance match exactly the bits associated
        with one of the members. This however doesn't necessarily mean that this flag instance isn't a
        combination of several flags because the bits of a member can be the subset of another member.
        For example if member0_bits=0x1 and member1_bits=0x3 then the bits of member0 are a subset of
        the bits of member1. If a flag instance holds the bits of member1 then Flags.is_member returns
        True and Flags.properties returns the properties of member1 but __len__() returns 2 and
        __iter__() yields both member0 and member1.
        """
        return type(self).__bits_to_properties__.get(int(self)) is not None

    @property
    def properties(self):
        """
        :return: Returns None if this flag isn't an exact member of a flags class but a combination of flags,
        returns an object holding the properties (e.g.: name, data, index, ...) of the flag otherwise.
        We don't store flag properties directly in Flags instances because this way Flags instances that are
        the (temporary) result of flags arithmetic don't have to maintain these fields and it also has some
        benefits regarding memory usage. """
        return type(self).__bits_to_properties__.get(int(self))

    @property
    def name(self):
        properties = self.properties
        return self.properties.name if properties else None

    @property
    def data(self):
        properties = self.properties
        return self.properties.data if properties else UNDEFINED

    def __getattr__(self, name):
        try:
            member = type(self).__members__[name]
        except KeyError:
            raise AttributeError(name)
        return member in self

    def __iter__(self):
        members = type(self).__members_without_aliases__.values()
        return (member for member in members if member in self)

    def __reversed__(self):
        members = reversed(list(type(self).__members_without_aliases__.values()))
        return (member for member in members if member in self)

    def __len__(self):
        return sum(1 for _ in self)

    def __hash__(self):
        return int(self) ^ hash(type(self))

    def __reduce_ex__(self, proto):
        value = int(self) if type(self).__pickle_int_flags__ else self.to_simple_str()
        return type(self), (value,)

    def __str__(self):
        # Warning: The output of this method has to be a string that can be processed by bits_from_str()
        return self.__internal_str()

    def __internal_str(self):
        if not type(self).__dotted_single_flag_str__:
            return '%s(%s)' % (type(self).__name__, self.to_simple_str())
        contained_flags = list(self)
        if len(contained_flags) != 1:
            # This is the zero flag or a set of flags (as a result of arithmetic)
            # or a flags class member that is a superset of another flags member.
            return '%s(%s)' % (type(self).__name__, '|'.join(member.name for member in contained_flags))
        return '%s.%s' % (type(self).__name__, contained_flags[0].properties.name)

    def __repr__(self):
        contained_flags = list(self)
        if len(contained_flags) != 1:
            # This is the zero flag or a set of flags (as a result of arithmetic)
            # or a flags class member that is a superset of another flags member.
            return '<%s bits=0x%04X>' % (self.__internal_str(), int(self))
        return '<%s bits=0x%04X data=%r>' % (self.__internal_str(), contained_flags[0].properties.bits,
                                             contained_flags[0].properties.data)

    def to_simple_str(self):
        return '|'.join(member.name for member in self)

    @classmethod
    def from_simple_str(cls, s):
        """ Accepts only the output of to_simple_str(). The output of __str__() is invalid as input. """
        if not isinstance(s, str):
            raise TypeError("Expected an str instance, received %r" % (s,))
        return cls(cls.bits_from_simple_str(s))

    @classmethod
    def from_str(cls, s):
        """ Accepts both the output of to_simple_str() and __str__(). """
        if not isinstance(s, str):
            raise TypeError("Expected an str instance, received %r" % (s,))
        return cls(cls.bits_from_str(s))

    @classmethod
    def bits_from_simple_str(cls, s):
        member_names = (name.strip() for name in s.split('|'))
        member_names = filter(None, member_names)
        bits = 0
        for member_name in filter(None, member_names):
            member = cls.__all_members__.get(member_name)
            if member is None:
                raise ValueError("Invalid flag '%s.%s' in string %r" % (cls.__name__, member_name, s))
            bits |= int(member)
        return bits

    @classmethod
    def bits_from_str(cls, s):
        """ Converts the output of __str__ into an integer. """
        try:
            if len(s) <= len(cls.__name__) or not s.startswith(cls.__name__):
                return cls.bits_from_simple_str(s)
            c = s[len(cls.__name__)]
            if c == '(':
                if not s.endswith(')'):
                    raise ValueError
                return cls.bits_from_simple_str(s[len(cls.__name__)+1:-1])
            elif c == '.':
                member_name = s[len(cls.__name__)+1:]
                return int(cls.__all_members__[member_name])
            else:
                raise ValueError
        except ValueError as ex:
            if ex.args:
                raise
            raise ValueError("%s.%s: invalid input: %r" % (cls.__name__, cls.bits_from_str.__name__, s))
        except KeyError as ex:
            raise ValueError("%s.%s: Invalid flag name '%s' in input: %r" % (cls.__name__, cls.bits_from_str.__name__,
                                                                             ex.args[0], s))
