import numpy as np
import h5py
import tensorflow as tf

from .pipeline import *

def get_meta(db_file, db_name):
    return dict(db_file[db_name].attrs.items())


def get_tokens(db_file, db_name, partition, chunked=False):
    sup_grp = db_file[db_name][partition]
    
    if not chunked:
        grp_name = f'/{db_name}/{partition}/'
        tokens = list(grp_name+t for t in sup_grp)
    else:
        tokens = []
        for gname, grp in sup_grp.items():
            grp_name = f'/{db_name}/{partition}/{gname}/'
            print(f'Reading tokens: {grp_name}*',flush=True)
            tokens.extend(grp_name+t for t in grp)

    return tokens


def read_data(group, key):
    if isinstance(key, tuple):
        return group[key[0]].attrs[key[1]]
    else:
        return group[key][()]


def read_record(db_file, token, keys):
    return tuple(read_data(db_file[token], k) for k in keys)


class RecordReader:
    def __init__(self, db_file, record_names, 
                 record_keys, record_types, record_shapes,
                 token_key='record_name'):
        self.db_file = db_file
        self.record_names = record_names
        self.record_keys = record_keys
        self.record_types = record_types
        self.record_shapes = record_shapes
        self.token_key = token_key
            
    def read_record_np(self, token_tf):
        token = token_tf.numpy().decode()
        return read_record(self.db_file, token, self.record_keys)

    def __call__(self, token_tf):
        outputs = tf.py_function(self.read_record_np, [token_tf], 
                                 self.record_types)
        for out, shp in zip(outputs, self.record_shapes):
            out.set_shape(shp)
        
        record = {}
        if self.token_key is not None:
            record[self.token_key]=token_tf

        record.update(zip(self.record_names, outputs))
        
        return record

