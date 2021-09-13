
import tensorflow as tf

from ..graph import FeatureMatrix
from ..dataset_base import DatasetBase
from ..graph_dataset_base import (GraphDatasetBase, MatrixDatasetBase,
                                  SVDDatasetBase)

dataset_name = 'TSP'
partition_names = ['training','validation','test']

record_names=['num_nodes','edges','node_features','edge_features','edge_labels']

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
        'type'      : tf.float32,
        'shape'     : [None,2]
    },
    'edge_features' :{
        'key'       : 'data/features/edges/feat',
        'type'      : tf.float32,
        'shape'     : [None,1]
    },
    'edge_labels'   :{
        'key'       : 'targets/edge_labels',
        'type'      : tf.int32,
        'shape'     : [None],
    },
}

record_keys   = list(record_proto[r]['key']   for r in record_names)
record_types  = list(record_proto[r]['type']  for r in record_names)
record_shapes = list(record_proto[r]['shape'] for r in record_names)



class Dataset(GraphDatasetBase, DatasetBase):
    def __init__(self,
                 max_length  = None,
                 mask_value  = -1.,
                 target_mask = 0,
                 **kwargs
                 ):
        super().__init__(max_length = max_length,
                         **kwargs)
        self.mask_value = mask_value
        self.target_mask   = target_mask
    
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
            record_name     = b'',
            num_nodes       = 0  ,
            edges           = -1 ,
            node_features   = self.mask_value ,
            edge_features   = self.mask_value ,
            edge_labels     = self.target_mask ,
        )
    
    def get_padded_shapes(self):
        return dict(
            record_name     = [] ,
            num_nodes       = [] ,
            edges           = [None,2] ,
            node_features   = [self.max_length,2],
            edge_features   = [None,1],
            edge_labels     = [None] ,
        )


class MatrixDataset(MatrixDatasetBase, Dataset):
    def __init__(self,
                 mark_invalid_features = True ,
                 return_edge_features  = False,
                 return_edge_labels    = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.return_edge_features  = return_edge_features
        self.mark_invalid_features = mark_invalid_features
        self.return_edge_labels    = return_edge_labels
        
        self.include_if('edge_features', lambda: self.return_edge_features)
        self.include_if('edge_labels', lambda: self.return_edge_labels)

    def load_split(self, split):        
        db_record = super().load_split(split)

        # AT = tf.data.experimental.AUTOTUNE
        
        featm_kwargs = dict(increment_by_1=True, decrement_by_1=True) \
                               if self.mark_invalid_features \
                                  else {}
        featm = FeatureMatrix(**featm_kwargs)
        
        targm = FeatureMatrix(feature_key='edge_labels',
                              output_key='target')

        mapped_record = db_record.map(featm).map(targm)
        return mapped_record

    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            feature_matrix = self.mask_value,
            target         = self.target_mask,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            feature_matrix = [self.max_length, self.max_length, 1],
            target = [self.max_length, self.max_length],
        )



class SVDDataset(SVDDatasetBase, MatrixDataset):
    pass
