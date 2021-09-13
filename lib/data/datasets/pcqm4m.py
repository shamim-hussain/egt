
import tensorflow as tf
import numpy as np

from ..graph import FeatureMatrix
from ..dataset_base import DatasetBase
from pathlib import Path
from ..graph_dataset_base import (GraphDatasetBase, MatrixDatasetBase,
                                  SVDDatasetBase, EigenDatasetBase)

dataset_name = 'PCQM4M'
partition_names = ['training','validation','test']

record_names=['num_nodes','edges','node_features','edge_features','target']

record_proto={
    'num_nodes'     :{
        'key'       : ('data','num_nodes'),
        'type'      : tf.int32,
        'shape'     : [],
    },
    'edges'         :{
        'key'       : 'data/edges',
        'type'      : tf.int8,
        'shape'     : [None,2],
    },
    'node_features' :{
        'key'       : 'data/features/nodes/feat',
        'type'      : tf.int8,
        'shape'     : [None,9],
    },
    'edge_features' :{
        'key'       : 'data/features/edges/feat',
        'type'      : tf.int8,
        'shape'     : [None,3],
    },
    'target'        :{
        'key'       : 'targets/value',
        'type'      : tf.float32,
        'shape'     : [],
    },
}

record_keys   = list(record_proto[r]['key']   for r in record_names)
record_types  = list(record_proto[r]['type']  for r in record_names)
record_shapes = list(record_proto[r]['shape'] for r in record_names)

node_feat_dims = [53,  3,  7, 10, 5,  5,  6,  2,  2]
edge_feat_dims = [4, 3, 2]

max_node_feat_dims = [92,  3, 11, 12,  7,  5,  6,  2,  2]
max_edge_feat_dims = [4, 3, 2]

from pathlib import Path
class Dataset(GraphDatasetBase, DatasetBase):
    def __init__(self,
                 max_length     = 51,
                 mask_value     = -1,
                 cache_data     = True,
                 cache_path     = None,
                 **kwargs
                 ):
        super().__init__(max_length = max_length,
                         **kwargs)
        self.mask_value     = mask_value
        self.cache_data     = cache_data 
        self.cache_path     = cache_path
        
        self._shuffle_tokens = False
        self._shuffle_db = False
    
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
    
    def is_chunked(self):
        return True
    
    def cast_db(self, db_record):
        def feat_cast(inputs):
            inputs['edges'        ] = tf.cast(inputs['edges'        ], tf.int32) 
            inputs['node_features'] = tf.cast(inputs['node_features'], tf.int32) 
            inputs['edge_features'] = tf.cast(inputs['edge_features'], tf.int32)
            return inputs
        
        db_record = db_record.map(feat_cast)
        return db_record
    
    def load_split(self, split, shuffle_flg=True, cast_flg=True):        
        db_record = super().load_split(split)
        if self.cache_data:
            cpath = self.cache_path
            if cpath is None:
                db_record = db_record.cache()
            else:
                cpath = Path(cpath)/split
                cpath.mkdir(parents=True, exist_ok=True)
                cpath = str(cpath/split)
                db_record = db_record.cache(cpath)
        
        if (split in self.shuffle_splits) and shuffle_flg:
            db_record = db_record.shuffle(len(self.record_tokens[split]))
        
        if cast_flg:
            db_record = self.cast_db(db_record)
        
        return db_record
    
    def cache(self, paths=None, clear=False):
        return self.load_data()
    
    def get_paddings(self):
        return dict(
            record_name     = b'',
            num_nodes       = 0  ,
            edges           = -1,
            node_features   = self.mask_value ,
            edge_features   = self.mask_value ,
            target          = 0. ,
        )
    
    def get_padded_shapes(self):
        return dict(
            record_name     = []                 ,
            num_nodes       = []                 ,
            edges           = [None,2]           ,
            node_features   = [self.max_length,9],
            edge_features   = [None,3]           ,
            target          = []                 ,
        )



class MatrixDataset(MatrixDatasetBase, Dataset):
    def __init__(self, 
                 mark_invalid_features = True ,
                 return_edge_features  = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_edge_features  = return_edge_features
        self.mark_invalid_features = mark_invalid_features
        
        self.include_if('edge_features', lambda: self.return_edge_features)

    def load_split(self, split):        
        db_record = super().load_split(split)

        # AT = tf.data.experimental.AUTOTUNE
        
        featm_kwargs = dict(increment_by_1=True, decrement_by_1=True) \
                               if self.mark_invalid_features \
                                  else {}
        featm = FeatureMatrix(**featm_kwargs)

        mapped_record = db_record.map(featm)
        return mapped_record
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            feature_matrix = self.mask_value,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            feature_matrix = [self.max_length, self.max_length, 3],
        )


from ..svd import SVDFeatures
from ..pipeline import SelectFeatures, MergeFeatures
class SVDDataset(SVDDatasetBase, MatrixDataset):
    def __init__(self,
                 svd_cache_data     = True,
                 svd_cache_path     = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.svd_cache_data     = svd_cache_data 
        self.svd_cache_path     = svd_cache_path
        
        self._shuffle_db = True
    
    def load_split(self, split):        
        db_wo_svd = super().load_split(split, shuffle=False)
        
        AT = tf.data.experimental.AUTOTUNE

        singular = SVDFeatures(num_features   = self.num_features,
                               mult_sing_vals = self.mult_sing_vals,
                               norm_first     = self.norm_for_svd,
                               norm_symmetric = self.norm_sym_for_svd)
        selfeat = SelectFeatures(['record_name','singular_vectors'])
        
        db_svd_t = db_wo_svd.map(singular,AT).map(selfeat)
        
        def db_svd_gen():
            for elem in db_svd_t:
                yield elem
                
        db_svd = tf.data.Dataset.from_generator(db_svd_gen,
                                                output_types={
                                                    'record_name':tf.string,
                                                    'singular_vectors': tf.float32,
                                                },
                                                output_shapes= {
                                                    'record_name':[],
                                                    'singular_vectors': [None, self.num_features,2]
                                                    }
                                                )
        
        if self.svd_cache_data:
            cpath = self.svd_cache_path
            if cpath is None:
                db_svd = db_svd.cache()
            else:
                cpath = Path(cpath)/split
                cpath.mkdir(parents=True, exist_ok=True)
                cpath = str(cpath/split)
                db_svd = db_svd.cache(cpath)
        
        mfeat = MergeFeatures(verify_key='record_name')
        db_out = tf.data.Dataset.zip((db_wo_svd, db_svd)).map(mfeat)
        return db_out


from ..graph import GraphMatrix
class CachedSVDDataset(Dataset):
    def __init__(self,
                 num_features         = 64,
                 t_num_features       = 8,
                 
                 norm_for_svd         = False,
                 norm_sym_for_svd     = False,
                 mult_sing_vals       = True,

                 svd_cache_path       = None,
                 combined_cache_path  = None,
                 
                 
                 normalize        = False,
                 symmetric        = False,
                 laplacian        = False,
                 return_mat       = False,
                 return_sing_vals = False,
                 
                 **kwargs
                 ):
        super().__init__(**kwargs)                                                            
        self.num_features         = num_features
        self.t_num_features       = t_num_features         
        self.norm_for_svd         = norm_for_svd        
        self.norm_sym_for_svd     = norm_sym_for_svd    
        self.mult_sing_vals       = mult_sing_vals      
        self.svd_cache_path       = svd_cache_path      
        self.combined_cache_path  = combined_cache_path 
        
    def load_split(self, split): 
        db_wo_cast = super().load_split(split, shuffle_flg=False, cast_flg=False)
        AT = tf.data.experimental.AUTOTUNE
        
        matrix = GraphMatrix(normalize=False, 
                             symmetric=False,
                             laplacian=False)
        singular = SVDFeatures(num_features   = self.num_features,
                               mult_sing_vals = self.mult_sing_vals,
                               norm_first     = self.norm_for_svd,
                               norm_symmetric = self.norm_sym_for_svd)
        
        selfeat = SelectFeatures(['record_name','singular_vectors'])
        db_svd_t = self.cast_db(db_wo_cast).map(matrix,AT).map(singular,AT).map(selfeat)
        
        def db_svd_gen():
            for elem in db_svd_t:
                yield elem
                
        db_svd = tf.data.Dataset.from_generator(db_svd_gen,
                                                output_types={
                                                    'record_name':tf.string,
                                                    'singular_vectors': tf.float32,
                                                },
                                                output_shapes= {
                                                    'record_name':[],
                                                    'singular_vectors': [None, self.num_features,2]
                                                    }
                                                )
        
        cpath = Path(self.svd_cache_path)/split
        cpath.mkdir(parents=True, exist_ok=True)
        db_svd = db_svd.cache(str(cpath/split))
        
        mfeat = MergeFeatures(verify_key='record_name')
        db_combined = tf.data.Dataset.zip((db_wo_cast, db_svd)).map(mfeat)
        
        def svd_trim_fn(inputs):
            inputs['singular_vectors'] = inputs['singular_vectors'][:,:self.t_num_features,:]
            return inputs
        
        db_combined_t = db_combined.map(svd_trim_fn)
        
        def db_combined_gen():
            for elem in db_combined_t:
                yield elem
                
        db_combined_g = tf.data.Dataset.from_generator(db_combined_gen,
                                                output_types=dict(
                                                    **dict((r,record_proto[r]['type']) 
                                                           for r in record_names),
                                                    record_name=tf.string,
                                                    singular_vectors= tf.float32,
                                                ),
                                                output_shapes= dict(
                                                    **dict((r,record_proto[r]['shape'])
                                                           for r in record_names),
                                                    record_name=[],
                                                    singular_vectors= [None, self.t_num_features,2]
                                                    )
                                                )
        
        cpath = Path(self.combined_cache_path)/split
        cpath.mkdir(parents=True, exist_ok=True)
        db_out = db_combined_g.cache(str(cpath/split))
        
        db_out = self.cast_db(db_out)
        
        if (split in self.shuffle_splits):
            db_out = db_out.shuffle(len(self.record_tokens[split]))
        
        return db_out
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            singular_vectors   = 0.,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            singular_vectors = [self.max_length, self.t_num_features, 2],
        )


class CachedSVDMatrixDataset(MatrixDatasetBase, CachedSVDDataset):
    def __init__(self, 
                 mark_invalid_features = True ,
                 return_edge_features  = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_edge_features  = return_edge_features
        self.mark_invalid_features = mark_invalid_features
        
        self.include_if('edge_features', lambda: self.return_edge_features)

    def load_split(self, split):        
        db_record = super().load_split(split)

        # AT = tf.data.experimental.AUTOTUNE
        
        featm_kwargs = dict(increment_by_1=True, decrement_by_1=True) \
                               if self.mark_invalid_features \
                                  else {}
        featm = FeatureMatrix(**featm_kwargs)

        mapped_record = db_record.map(featm)
        return mapped_record
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            feature_matrix = self.mask_value,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            feature_matrix = [self.max_length, self.max_length, 3],
        )