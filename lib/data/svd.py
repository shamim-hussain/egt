from operator import le
import tensorflow as tf
import numpy as np

from .graph import normalize_adjacency

def eager_svd_h(A):
    S, U, V = tf.linalg.svd(A)
    return U, S, V

def get_svd_features(A, num_features=None):
    length = tf.shape(A)[0]
    length_s = A.shape[0]

    U, S, V = tf.py_function(eager_svd_h, [A], [tf.float32]*3)

    UV = tf.stack([U,V], axis=0)
    
    S.set_shape([length_s])
    UV.set_shape([2, length_s, length_s])

    if num_features is not None:
        pad_len = tf.maximum(0, num_features - length)
            
        S = tf.pad(S, [(0, pad_len)])
        UV = tf.pad(UV, [(0,0), (0,0), (0, pad_len)])
        
        S = S[:num_features]
        UV = UV[:, :, :num_features]
        
        S.set_shape([num_features])
        UV.set_shape([2, length_s, num_features])

    return UV, S


class SVDFeatures:
    def __init__(self                                    ,
                 num_features       = None               ,
                 norm_first         = False              ,
                 norm_symmetric     = False              ,
                 mult_sing_vals     = True               ,
                 input_key          = 'graph_matrix'     ,
                 vector_key         = 'singular_vectors' ,
                 value_key          = 'singular_values'  ):
        
        self.input_key      = input_key
        self.num_features   = num_features
        self.norm_first     = norm_first
        self.norm_symmetric = norm_symmetric
        self.mult_sing_vals = mult_sing_vals
        self.vector_key     = vector_key
        self.value_key      = value_key
    
    def __call__(self, inputs):
        A = inputs[self.input_key]
        if self.norm_first:
            A = normalize_adjacency(A, symmetric=self.norm_symmetric)
        
        UV, S = get_svd_features(A, self.num_features)
        if self.mult_sing_vals:
            UV = UV * tf.sqrt(S)
        
        UV = tf.transpose(UV, [1, 2, 0])

        return_dict = {
            **inputs           ,
            self.vector_key : UV,
            self.value_key  : S,
        }
        
        return return_dict



