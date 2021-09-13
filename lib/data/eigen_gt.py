import scipy.sparse as sp
import numpy as np
import tensorflow as tf


def eigen_pe_sp(edges, num_nodes, pos_enc_dim):
    if isinstance(edges, tf.Tensor):
        edges = edges.numpy()
    if isinstance(num_nodes, tf.Tensor):
        num_nodes = num_nodes.numpy()
    if isinstance(pos_enc_dim, tf.Tensor):
        pos_enc_dim = pos_enc_dim.numpy()
    
    
    rows_mat = edges[:,0]
    cols_mat = edges[:,1]
    
    data_mat = np.ones(rows_mat.shape, dtype='float32')

    A = sp.csr_matrix((data_mat, (rows_mat,cols_mat)),
                      shape=(num_nodes,num_nodes),
                      dtype='float32')
    N = sp.diags(A.sum(axis=1).A.squeeze().clip(1) ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2)
    EigVec = EigVec[:, EigVal.argsort()]
    
    pos_enc = np.real(EigVec[:,1:pos_enc_dim+1]).astype('float32')

    return pos_enc


def eigen_pe_np(edges, num_nodes, pos_enc_dim):
    if isinstance(edges, tf.Tensor):
        edges = edges.numpy()
    if isinstance(num_nodes, tf.Tensor):
        num_nodes = num_nodes.numpy()
    if isinstance(pos_enc_dim, tf.Tensor):
        pos_enc_dim = pos_enc_dim.numpy()
    
    
    rows_mat = edges[:,0]
    cols_mat = edges[:,1]
    
    data_mat = np.ones(rows_mat.shape, dtype='float32')

    A = sp.csr_matrix((data_mat, (rows_mat,cols_mat)),
                      shape=(num_nodes,num_nodes),
                      dtype='float32')
    N = sp.diags(A.sum(axis=1).A.squeeze().clip(1) ** -0.5, dtype=float)
    L = sp.eye(num_nodes) - N * A * N

    EigVal, EigVec = np.linalg.eig(L.toarray())
    EigVec = np.real(EigVec[:, EigVal.argsort()])
    pos_enc = EigVec[:,1:pos_enc_dim+1].astype('float32')

    return pos_enc


def eigen_pe(edges, num_nodes, pos_enc_dim, sparse=True):
    if sparse:
        out = tf.py_function(eigen_pe_sp, [edges, num_nodes, pos_enc_dim], tf.float32)
    else:
        out = tf.py_function(eigen_pe_np, [edges, num_nodes, pos_enc_dim], tf.float32)
        
    if not isinstance(num_nodes, tf.Tensor):
        out.set_shape([num_nodes, None])
    if not isinstance(pos_enc_dim, tf.Tensor):
        out.set_shape([None, pos_enc_dim])
    return out


class EigenFeatures:
    def __init__(self                                     ,
                 num_features        = 8                  ,
                 sparse              = True               ,
                 edges_input_key     = 'edges'            ,
                 num_nodes_input_key = 'num_nodes'        ,
                 vector_key          = 'eigen_vectors'    ):
        
        self.num_features        = num_features            
        self.sparse              = sparse             
        self.edges_input_key     = edges_input_key    
        self.num_nodes_input_key = num_nodes_input_key
        self.vector_key          = vector_key         
    
    def __call__(self, inputs):
        edges = inputs[self.edges_input_key]
        num_nodes = inputs[self.num_nodes_input_key]
        
        U = eigen_pe(edges, num_nodes, self.num_features, self.sparse)
        
        return_dict = {
            **inputs           ,
            self.vector_key : U,
        }
        
        return return_dict
