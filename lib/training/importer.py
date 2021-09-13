
import importlib

def import_object(object_name):
    module_name, object_name = object_name.rsplit('.', 1)
    imported_module = importlib.import_module(module_name)
    return getattr(imported_module, object_name)



def import_scheme(scheme_name):
    return import_object('lib.training.schemes.'+scheme_name+'.SCHEME')
