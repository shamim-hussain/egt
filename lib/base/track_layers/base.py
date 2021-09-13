
from types import SimpleNamespace
from contextlib import contextmanager
from collections import defaultdict

class CustomLayers(SimpleNamespace):
    def __init__(self, *layers, **renamed_layers):
        layers_dict = dict(
            (lr.__name__, lr) for lr in layers
        )
        layers_dict.update(**renamed_layers)
        super().__init__(**layers_dict)


class TrackedLayers:
    def __init__(self, *layers_modules):
        self._layers_dict = {}
        self._layers_modules = list(layers_modules)
        self._namespaces = defaultdict(list)
        self._current_namespace = '/'
    
    def track_module(self, layers_module):
        self._layers_modules.append(layers_module)
    
    def get_layers_dict(self):
        return self._layers_dict
    
    def get_layer(self, name):
        return self._layers_dict[name]
    
    @contextmanager
    def namespace(self, name):
        old_namespace = self._current_namespace
        try:
            self._current_namespace = self._current_namespace + name + '/'
            yield
        finally:
            self._current_namespace = old_namespace
    
    def get_namespaces(self):
        return self._namespaces

    def __getattr__(self,attr):
        layer_class = None
        for module in self._layers_modules:
            if hasattr(module, attr):
                layer_class = getattr(module, attr)
                break
        
        if layer_class is None:
            raise KeyError(attr)
            
        def get_layer(*args, **kwargs):
            layer_name = kwargs['name'] 
            if not layer_name in self._layers_dict:
                new_layer = layer_class(*args,**kwargs)
                self._layers_dict[layer_name] = new_layer
                self._namespaces[self._current_namespace].append(new_layer)
            return self._layers_dict[layer_name]
        
        return get_layer