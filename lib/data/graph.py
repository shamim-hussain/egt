import tensorflow as tf


def add_self_loop_edges(edges, shape):
    if isinstance(shape, list):
        shape = shape[0]
    
    rng = tf.cast(tf.range(shape), edges.dtype)
    rng = tf.tile(rng[:,None], [1,2])

    edges = tf.concat([edges, rng],axis=0)
    return edges

def get_graph_matrix(edges, shape, 
                     features=None, self_loop=False,
                     increment_by_1 = False,
                     decrement_by_1 = False,
                     dtype = tf.float32):
    if not isinstance(shape, list):
        shape = [shape, shape]
    
    if edges.dtype not in [tf.int32, tf.int64]:
        edges = tf.cast(edges, tf.int32)    
    
    num_edges = tf.shape(edges)[0]
    if features is None:
        features = tf.ones([num_edges], dtype=dtype)
    elif increment_by_1:
        features = features + 1
    
    feature_dims = features.shape[1:].as_list()
    mat_dim = shape + feature_dims

    mat = tf.scatter_nd(edges, features, mat_dim)
    if self_loop:
        mat += tf.eye(*shape, dtype=mat.dtype)

    if decrement_by_1:
        mat = mat - 1
    return mat


def normalize_adjacency(A, symmetric=False):
    d = tf.reduce_sum(A, axis=1, keepdims=True)
    if not symmetric:
        return tf.math.divide_no_nan(A,d)
    else:
        d_minus_half = tf.math.divide_no_nan(1.,tf.sqrt(d))
        perm = list(range(A.shape.rank))
        perm[0], perm[1] = 1 , 0
        d_minus_half_t = tf.transpose(d_minus_half, perm)
        return d_minus_half*A*d_minus_half_t




def get_adjacency(edges, shape, normalize=True, 
                  symmetric=False,
                  add_self_loops=True):
    if add_self_loops:
        edges = add_self_loop_edges(edges,shape)
    A = get_graph_matrix(edges, shape)
    if normalize:
        A = normalize_adjacency(A, symmetric=symmetric)
    return A


def get_laplacian(edges, shape, add_self_loops=True):
    A = get_adjacency(edges, shape, 
                      normalize      = True,
                      symmetric      = True, 
                      add_self_loops = add_self_loops)

    if not isinstance(shape, list):
        shape = [shape, shape]
    D = tf.eye(*shape, dtype=tf.float32) - A
    return D


class FeatureMatrix:
    def __init__(self                                 ,
                 increment_by_1 = False               ,
                 decrement_by_1 = False               ,
                 feature_key    = 'edge_features'     ,
                 edge_key       = 'edges'             , 
                 shape_key      = 'num_nodes'         ,
                 output_key     = 'feature_matrix'  ):    
        self.feature_key    = feature_key
        self.increment_by_1 = increment_by_1
        self.decrement_by_1 = decrement_by_1
        self.edge_key       = edge_key      
        self.shape_key      = shape_key     
        self.output_key     = output_key 
        
    def __call__(self, inputs):
        edges, shape = inputs[self.edge_key], inputs[self.shape_key]
        features = inputs[self.feature_key]
        output = get_graph_matrix(edges, shape, features,
                                    increment_by_1 = self.increment_by_1,
                                    decrement_by_1 = self.decrement_by_1,)
        return_matrix = {
            **inputs,
            self.output_key : output,
        }

        return return_matrix

class GraphMatrix:
    def __init__(self                                 , 
                 normalize      = True                ,
                 symmetric      = False               ,
                 laplacian      = False               ,
                 add_self_loops = True                ,
                 edge_key       = 'edges'             , 
                 shape_key      = 'num_nodes'         ,
                 output_key     = 'graph_matrix'      ):
        self.normalize      = normalize
        self.symmetric      = symmetric
        self.laplacian      = laplacian     
        self.add_self_loops = add_self_loops
        self.edge_key       = edge_key      
        self.shape_key      = shape_key     
        self.output_key     = output_key 
        
    def __call__(self, inputs):
        edges, shape = inputs[self.edge_key], inputs[self.shape_key]
        if not self.laplacian:
            output = get_adjacency(edges, shape, 
                                self.normalize, 
                                self.symmetric, 
                                self.add_self_loops)
        else:
            output = get_laplacian(edges, shape, self.add_self_loops)

        return_matrix = {
            **inputs,
            self.output_key : output,
        }

        return return_matrix
    
