
from threading import Condition
import tensorflow as tf

from .svd import SVDFeatures
from .eigen_gt import EigenFeatures
from .graph import GraphMatrix
from .pipeline import ExcludeFeatures



class GraphDatasetBase:
    def __init__(self, 
                 max_length = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self._inclusion_conditions = {}
    
    def include_if(self, feature_name, condition):
        self._inclusion_conditions[feature_name]  = condition
    
    def map_data_split(self, split, data):
        excluded_feats = []
        for feature_name, condition in self._inclusion_conditions.items():
            if not condition():
                excluded_feats.append(feature_name)
        
        if len(excluded_feats)>0:
            return data.map(ExcludeFeatures(excluded_feats))
        else:
            return data


class MatrixDatasetBase:
    def __init__(self                                             ,
                 normalize            = False                     ,
                 symmetric            = False                     ,
                 laplacian            = False                     ,
                 return_edges         = False                     ,
                 matrix_pad_value     = 0.                        ,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.normalize            = normalize   
        self.symmetric            = symmetric
        self.laplacian            = laplacian
        self.return_edges         = return_edges
        self.matrix_pad_value     = matrix_pad_value
        
        self.include_if('edges', lambda: self.return_edges)

    def load_split(self, split):
        db_record = super().load_split(split)

        # AT = tf.data.experimental.AUTOTUNE

        matrix = GraphMatrix(normalize=self.normalize, 
                             symmetric=self.symmetric,
                             laplacian=self.laplacian)

        mapped_record = db_record.map(matrix)

        return mapped_record
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            graph_matrix   = self.matrix_pad_value,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            graph_matrix = [self.max_length, self.max_length],
        )


class SVDDatasetBase:
    def __init__(self,
                 normalize        = False,
                 symmetric        = False,
                 laplacian        = False,
                 num_features     = 16,
                 norm_for_svd     = False,
                 norm_sym_for_svd = False,
                 mult_sing_vals   = True,
                 return_mat       = False,
                 return_sing_vals = False,
                 **kwargs
                 ):
        super().__init__(
            normalize      = normalize      ,
            symmetric      = symmetric      ,
            laplacian      = laplacian      ,
            **kwargs
        )
        self.num_features        = num_features
        self.norm_for_svd        = norm_for_svd
        self.norm_sym_for_svd    = norm_sym_for_svd
        self.mult_sing_vals      = mult_sing_vals  
        self.return_mat          = return_mat
        self.return_sing_vals    = return_sing_vals     
        
        self.include_if('graph_matrix', lambda: return_mat)
        self.include_if('singular_values', lambda: return_sing_vals) 

    def load_split(self, split):        
        db_matrix = super().load_split(split)

        AT = tf.data.experimental.AUTOTUNE

        singular = SVDFeatures(num_features   = self.num_features,
                               mult_sing_vals = self.mult_sing_vals,
                               norm_first     = self.norm_for_svd,
                               norm_symmetric = self.norm_sym_for_svd)
        
        return db_matrix.map(singular,AT)
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            singular_values    = 0.,
            singular_vectors   = 0.,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            singular_values =  [self.num_features],
            singular_vectors = [self.max_length, self.num_features, 2],
        )


class EigenDatasetBase:
    def __init__(self,
                 num_features = 8,
                 sparse       = True,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.sparse       = sparse

    def load_split(self, split):        
        db_matrix = super().load_split(split)

        AT = tf.data.experimental.AUTOTUNE

        eigen = EigenFeatures(num_features=self.num_features, 
                              sparse=self.sparse)
        return db_matrix.map(eigen,AT)
    
    def get_paddings(self):
        return dict(
            **super().get_paddings(),
            eigen_vectors = 0.,
        )
    
    def get_padded_shapes(self):
        return dict(
            **super().get_padded_shapes(),
            eigen_vectors = [self.max_length, self.num_features],
        )


