import time
import warnings


class timer:
    """
    The timer that provides timecost function
    To create a timer :
    tm = timer()
        tm.start()
        .. code-block:: python
        time_cost = tm.end()
    """

    def __init__(self):
        self._startt = time.time()

    def start(self):
        self._startt = time.time()

    def stop(self):
        timens = time.time() - self._startt
        return timens


def tuple2str(t: tuple, splitch=","):
    s = ""
    for i, v in enumerate(t):
        s += str(v)
        if i != len(t) - 1:
            s += splitch
    return s


class Registry:
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def __setitem__(self, name, obj):
        # no action other than issuing a warning for
        # an already registered object
        if name in self._obj_map:
            warnings.warn(
                f"'{name}' object was already registered in '{self._name}' registry"
            )
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            # decorator acts as an object to be registered
            def deco(func_or_class):
                name = func_or_class.__name__
                self.__setitem__(name, func_or_class)
                return func_or_class

            return deco

        name = obj.__name__
        self.__setitem__(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(f"'{name}' object can't found in '{self._name}' registry")
        return ret

    def __getitem__(self, item):
        if self._obj_map.__contains__(item) is False:
            raise KeyError(f"'{item}' object can't found in '{self._name}' registry")
        return self._obj_map[item]

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


NODEPROFILER_REGISTRY = Registry("nodeprofiler")
NODE_REGISTRY = Registry("NODE")


class Global:
    """
    The Global class that provides global container;
    To use Global container:
        .. code-block:: python
        GLOBAL['tensor_map'] = A
    """

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def __getitem__(self, item):
        if not self.__contains__(item):
            return None
        return self._obj_map[item]

    def __setitem__(self, key, value):
        self._obj_map[key] = value

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


GLOBAL = Global("Shared")


def volume(shape: []):
    # if not isinstance(shape,list):
    #     return 1 #scalar
    val = 1 if len(shape) > 0 else 0
    for v in shape:
        val *= v
    return val
