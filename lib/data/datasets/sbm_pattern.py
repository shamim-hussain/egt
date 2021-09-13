
import tensorflow as tf

from ..dataset_base import DatasetBase
from ..graph_dataset_base import (GraphDatasetBase, MatrixDatasetBase,
                                  SVDDatasetBase, EigenDatasetBase)

dataset_name = 'SBM_PATTERN'
partition_names = ['training','validation','test']

record_names=['num_nodes','edges','node_features','target']

record_proto={
    'num_nodes'     :{
        'key'       : ('data','num_nodes'),
        'type'      : tf.int32,
        'shape'     : [],
    },
    'edges'         :{
        'key'       : 'data/edges',
        'type'      : tf.int32,
        'shape'     : [None,2],
    },
    'node_features' :{
        'key'       : 'data/features/nodes/feat',
        'type'      : tf.int32,
        'shape'     : [None]
    },
    'target'        :{
        'key'       : 'targets/node_labels',
        'type'      : tf.int32,
        'shape'     : [None],
    },
}

record_keys   = list(record_proto[r]['key']   for r in record_names)
record_types  = list(record_proto[r]['type']  for r in record_names)
record_shapes = list(record_proto[r]['shape'] for r in record_names)



class Dataset(GraphDatasetBase, DatasetBase):
    def __init__(self,
                 max_length = None,
                 mask_value = -1,
                 **kwargs
                 ):
        super().__init__(max_length = max_length,
                         **kwargs)
        self.mask_value = mask_value
    
    def get_dataset_name(self):
        return dataset_name
    def get_record_names(self):
        return record_names
    def get_record_keys(self):
        return record_keys
    def get_record_types(self):
        return record_types
    def get_record_shapes(self):
        return record_shapes
    
    def get_paddings(self):
        return dict(
            record_name     = b''             ,
            num_nodes       = 0               ,
            edges           = -1              ,
            node_features   = self.mask_value ,
            target          = 0               ,
        )
    
    def get_padded_shapes(self):
        return dict(
            record_name     = [] ,
            num_nodes       = [] ,
            edges           = [None,2] ,
            node_features   = [self.max_length] ,
            target          = [self.max_length] ,
        )


class MatrixDataset(MatrixDatasetBase, Dataset):
    pass

class SVDDataset(SVDDatasetBase, MatrixDataset):
    pass

class EigenDataset(EigenDatasetBase, MatrixDataset):
    def __init__(self,
                 num_features = 2,
                 sparse       = True,
                 **kwargs
                 ):
        super().__init__(
            num_features = num_features ,
            sparse       = sparse       , 
            **kwargs
        )

