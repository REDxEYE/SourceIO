# -*- coding: utf-8 -*-
import collections
import functools
import operator

__all__ = ['Dict', 'OrderedDict', 'FrozenDict', 'FrozenOrderedDict', 'ReadonlyDictProxy']


# version_info[0]: Increase in case of large milestones/releases.
# version_info[1]: Increase this and zero out version_info[2] if you have explicitly modified
#                  a previously existing behavior/interface.
#                  If the behavior of an existing feature changes as a result of a bugfix
#                  and the new (bugfixed) behavior is that meets the expectations of the
#                  previous interface documentation then you shouldn't increase this, in that
#                  case increase only version_info[2].
# version_info[2]: Increase in case of bugfixes. Also use this if you added new features
#                  without modifying the behavior of the previously existing ones.
version_info = (0, 0, 1)
__version__ = '.'.join(str(n) for n in version_info)
__author__ = 'István Pásztor'
__license__ = 'MIT'


class Items(object):
    """ An object that we use in place of the items method of our dictionaries.
    By replacing the items method of dictionary instances with an instance of this
    we can still call dict_instance.items() to get the items and we can also perform
    attribute-style readonly access on dict_instance.items to have access to dict items.
    For example: dict_instance.items.my_item """
    __slots__ = ('_dict', '_orig_items')

    @staticmethod
    def _getattr(self_, key):
        return super(Items, self_).__getattribute__(key)

    @staticmethod
    def _setattr(self_, key, value):
        super(Items, self_).__setattr__(key, value)

    def __init__(self, _dict, _orig_items):
        Items._setattr(self, '_dict', _dict)
        Items._setattr(self, '_orig_items', _orig_items)

    def __call__(self):
        return Items._getattr(self, '_orig_items')()

    def __contains__(self, item):
        return Items._getattr(self, '_dict').__contains__(item)

    def __iter__(self):
        return Items._getattr(self, '_dict').__iter__()

    def __len__(self):
        return Items._getattr(self, '_dict').__len__()

    def __getattribute__(self, item):
        try:
            return Items._getattr(self, '_dict')[item]
        except KeyError:
            raise AttributeError("Couldn't retrieve dictionary key '%s' with attribute access" % item)

    def __getitem__(self, item):
        return Items._getattr(self, '_dict')[item]

    def __setattr__(self, key, value):
        raise AttributeError("Item assignment through attribute access isn't supported")

    def __delattr__(self, item):
        raise AttributeError("Item deletion through attribute access isn't supported")

    def __setitem__(self, key, value):
        raise TypeError("Item assignment isn't supported")

    def __delitem__(self, key):
        raise TypeError("Item deletion isn't supported")


class MutableItems(Items):
    """ Mutable version of Items. Allows modification of dictionary items. """
    __slots__ = ()

    def __setattr__(self, key, value):
        Items._getattr(self, '_dict')[key] = value

    def __delattr__(self, item):
        try:
            del Items._getattr(self, '_dict')[item]
        except KeyError:
            raise AttributeError("Key '%s' was not found for deletion through attribute access" % item)

    def __setitem__(self, key, value):
        Items._getattr(self, '_dict').__setitem__(key, value)

    def __delitem__(self, key):
        Items._getattr(self, '_dict').__delitem__(key)


class ReadonlyItemsMixin(object):
    """ Provides attribute access to dictionary items on dict_instance.items with high priority.
    This means that attribute access on dict_instance.items can be used only to retrieve dict items. """
    __slots__ = ('__items',)

    def __init__(self, _dict):
        super(ReadonlyItemsMixin, self).__init__()
        self.__items = Items(_dict, _dict.items)

    @property
    def items(self):
        return self.__items


class ReadonlyDictProxy(ReadonlyItemsMixin, collections.Mapping):
    __slots__ = ('__dict',)

    def __init__(self, wrapped_dict):
        super(ReadonlyDictProxy, self).__init__(wrapped_dict)
        self.__dict = wrapped_dict

    def __getitem__(self, key):
        return self.__dict[key]

    def __getattr__(self, item):
        try:
            return self.__dict[item]
        except KeyError:
            raise AttributeError("Couldn't retrieve %s key '%s' with attribute access" % (type(self).__name__, item))

    def copy(self):
        """ Returns another proxy instance wrapping the same dictionary as this proxy. """
        return type(self)(self.__dict)

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        return '<%s %r>' % (type(self).__name__, self.__dict)


class ExtendedCopyMixin(object):
    """ This mixin adds a copy method that has an update parameter. It also solves the
    problem when the inherited dict base class copy (like the builtin dict.copy) returns
    an instance of the inherited base class instead of an instance of our subclass. """
    __slots__ = ()

    def copy(self, **update_items):
        """
        The standard dict.copy() method receives no parameters. This extended copy()
        can be used to create a copy that has extra added keys and/or some modified keys
        compared to the is instance.
        :param update_items: items to add or modify in the copied instance.
        """
        return type(self)(self, **update_items)


class FromKeysMixin(object):
    __slots__ = ()

    @classmethod
    def fromkeys(cls, keys, value=None):
        items = [(key, value) for key in keys]
        return cls(items)


class FrozenDict(FromKeysMixin, ReadonlyDictProxy):
    __slots__ = ('__hash',)
    __inner_dict__ = dict

    def __init__(self, *args, **kwargs):
        _dict = self.__inner_dict__(*args, **kwargs)
        super(FrozenDict, self).__init__(_dict)
        self.__hash = None

    def copy(self, **update_items):
        if not update_items:
            # Taking advantage of being immutable.
            return self
        return type(self)(self, **update_items)

    def __hash__(self):
        if self.__hash is None:
            hashes = map(hash, self.items())
            self.__hash = functools.reduce(operator.xor, hashes, 0)
        return self.__hash


class FrozenOrderedDict(FrozenDict):
    __slots__ = ()
    __inner_dict__ = collections.OrderedDict


class ReadAttributeAccessMixin(object):
    """ Provides read attribute access to dictionary items directly on the dict.
    Items have lower priority than the existing attributes of the dict instance. """
    __slots__ = ()

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError("Couldn't retrieve %s key '%s' with attribute access" % (type(self).__name__, item))


class WriteAttributeAccessMixin(object):
    """ Provides read attribute access to dictionary items directly on the dict.
    Items have lower priority than the existing attributes of the dict instance. """
    __slots__ = ()

    def __setattr__(self, key, value):
        if not self._has_init_finished():
            return super(WriteAttributeAccessMixin, self).__setattr__(key, value)
        # After finishing __init__ every __setattr__ call will be treated as item assignment.
        # This is a very dangerous and fragile method but at this point it works. If a base
        # class (like collections.OrderedDict) changes its implementation and starts assigning
        # instance variables after its __init__ then this solution will fail miserably.
        self[key] = value

    def __delattr__(self, item):
        if not self._has_init_finished():
            return super(WriteAttributeAccessMixin, self).__delattr__(item)
        try:
            del self[item]
        except KeyError:
            raise AttributeError("%s has no '%s' key to delete through attribute access" % (type(self).__name__, item))

    def _has_init_finished(self):
        raise NotImplementedError


# We cannot use here an ItemsMixin like ReadonlyItemsMixin because
# multiple inheritance with non-empty __slots__ doesn't work.
# For this reason we "inline" the MutableItems code into the class
# body instead of mixing it in. The same goes for OrderedDict.
class Dict(WriteAttributeAccessMixin, ReadAttributeAccessMixin, ExtendedCopyMixin, dict):
    __slots__ = ('__items', '_init_finished')

    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)
        self.__items = MutableItems(self, super(Dict, self).items)
        self._init_finished = True

    # Making items a property instead of an attribute in order to make it readonly.
    @property
    def items(self):
        return self.__items

    def _has_init_finished(self):
        return getattr(self, '_init_finished', False)


# Note: seemingly the collections.OrderedDict implementation doesn't have __slots__
# so an unnecessary __dict__ is created for each OrderedDict instance. Later I may
# drop collections.OrderedDict and come up with my own implementation to aid this.
class OrderedDict(WriteAttributeAccessMixin, ReadAttributeAccessMixin, ExtendedCopyMixin, collections.OrderedDict):
    __slots__ = ('__items', '_init_finished')

    def __init__(self, *args, **kwargs):
        super(OrderedDict, self).__init__(*args, **kwargs)
        self.__items = MutableItems(self, super(OrderedDict, self).items)
        self._init_finished = True

    @property
    def items(self):
        return self.__items

    def _has_init_finished(self):
        return getattr(self, '_init_finished', False)
