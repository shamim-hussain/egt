import tensorflow as tf
from tensorflow import keras

class EGT(keras.layers.Layer):
    def __init__(self,
                 num_heads           = 8,
                 clip_logits_value   = [-5., 5.],
                 scale_degree        = False,
                 scaler_type         = 'log',
                 edge_input          = True,
                 gate_input          = True,
                 attn_mask           = False,
                 num_virtual_nodes   = 0,
                 random_mask_prob    = 0.0,
                 attn_dropout        = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking=True
        
        if scale_degree and not gate_input:
            raise ValueError('scale_degree requires gate_input')
        
        if not scaler_type in ('log', 'linear'):
            raise ValueError('scaler_type must be log or linear')
        
        self.num_heads           = num_heads                 
        self.clip_logits_value   = clip_logits_value   
        self.scale_degree        = scale_degree 
        self.edge_input          = edge_input
        self.gate_input          = gate_input
        self.attn_mask           = attn_mask
        self.num_virtual_nodes   = num_virtual_nodes
        self.random_mask_prob    = random_mask_prob
        self.scaler_type         = scaler_type
        self.attn_dropout        = attn_dropout
        
        if self.gate_input:
            self.call = self.call_gated
        else:
            self.call = self.call_ungated
    
    def get_config(self):
        config = super().get_config()
        config.update(
            num_heads           = self.num_heads,
            clip_logits_value   = self.clip_logits_value,
            scale_degree        = self.scale_degree,
            edge_input          = self.edge_input,
            gate_input          = self.gate_input,
            attn_mask           = self.attn_mask,
            num_virtual_nodes   = self.num_virtual_nodes,
            random_mask_prob    = self.random_mask_prob,
            scaler_type         = self.scaler_type,
        )
        return config
    
    def call_gated(self, inputs, mask=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # get inputs
        QKV, *inputs = inputs                     # b,l,3dh
        if self.edge_input: E, *inputs = inputs   # b,l,l,h
        G, *inputs = inputs   # b,l,l,h
        if self.attn_mask: M, *inputs = inputs    # b,l,l,h
        if hasattr(mask, '__getitem__'): mask = mask[0]
        
        # split query, key, value
        QKV_shape = QKV.shape
        assert QKV_shape[2] % (self.num_heads*3) == 0
        dot_dim = QKV_shape[2] // (self.num_heads*3)
        QKV_shape_d = tf.shape(QKV)
        QKV = tf.reshape(QKV, (QKV_shape_d[0], QKV_shape_d[1], 3,
                               dot_dim, self.num_heads))         # b,l,3dh -> b,l,3,d,h
        QKV.set_shape([QKV_shape[0], QKV_shape[1], 3, dot_dim, self.num_heads]) 
        Q,K,V = tf.unstack(QKV, num=3, axis=2)  # b,l,d,h
        
        # form attention logits from nodes
        A_hat = tf.einsum('bldh,bmdh->blmh', Q, K) # b,l,l,h
        A_hat.set_shape([QKV_shape[0], QKV_shape[1], QKV_shape[1], self.num_heads])
        if self.clip_logits_value is not None:
            A_hat = tf.clip_by_value(A_hat, self.clip_logits_value[0], self.clip_logits_value[1])
        
        # update attention logits with edges
        H_hat = A_hat                         # b,l,l,h
        if self.edge_input: H_hat = H_hat + E # b,l,l,h
        
        # update attention logits with masks
        H_hat_ = H_hat
        G_ = G
        if mask is not None:
            mask_ = (tf.cast(mask[:,None,:,None], H_hat.dtype) - 1) * 1e9 # b,l -> b,1,l,1
            H_hat_ = H_hat_ + mask_
            G_ = G_ + mask_
            
        if self.attn_mask:
            if not M.dtype is H_hat.dtype:
                M = tf.cast(M, H_hat.dtype)
            M_ = (M - 1) * 1e9
            H_hat_ = H_hat_ + M_
            G_ = G_ + M_
        
        if self.random_mask_prob > 0.0 and training:
            uniform_noise = tf.random.uniform(tf.shape(H_hat_), 
                                              minval=0., maxval=1., dtype=H_hat_.dtype)
            random_mask_ = tf.where(uniform_noise < self.random_mask_prob, -1e9, 0.)
            H_hat_ = H_hat_ + random_mask_
            G_ = G_ + random_mask_
        
        # form attention weights
        A_tild = tf.nn.softmax(H_hat_, axis=2) # b,l,l,h
        gates = tf.sigmoid(G_)
        A_tild = A_tild * gates
        
        # attention output
        if self.attn_dropout > 0.0 and training:
            A_tild = tf.nn.dropout(A_tild, self.attn_dropout)
        
        # form output
        V_att = tf.einsum('blmh,bmdh->bldh', A_tild, V) # b,l,d,h
        
        # scale degree
        if self.scale_degree:
            degrees = tf.reduce_sum(gates, axis=2, keepdims=True) # b,l,l,h -> b,l,1,h
            if self.scaler_type == 'log':
                degree_scalers = tf.math.log(1 + degrees) # b,l,1,h
            elif self.scaler_type == 'linear':
                degree_scalers = degrees
            else:
                raise ValueError(f'Unknown scaler type {self.scaler_type}')
            if self.num_virtual_nodes > 0:
                non_vn_scalers = degree_scalers[:,self.num_virtual_nodes:]
                degree_scalers = tf.pad(non_vn_scalers,
                                        [(0,0),(self.num_virtual_nodes,0),(0,0),(0,0)],
                                        mode='CONSTANT', constant_values=1)
            V_att = V_att * degree_scalers
        
        # reshape output
        V_att = tf.reshape(V_att, (QKV_shape_d[0], QKV_shape_d[1],
                                   dot_dim*self.num_heads)) # b,l,dh
        V_att.set_shape([QKV_shape[0], QKV_shape[1], dot_dim*self.num_heads])
        
        return V_att, H_hat, A_tild
    
    def call_ungated(self, inputs, mask=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        
        # get inputs
        QKV, *inputs = inputs                     # b,l,3dh
        if self.edge_input: E, *inputs = inputs   # b,l,l,h
        
        if self.attn_mask: M, *inputs = inputs    # b,l,l,h
        if hasattr(mask, '__getitem__'): mask = mask[0]
        
        # split query, key, value
        QKV_shape = QKV.shape
        assert QKV_shape[2] % (self.num_heads*3) == 0
        dot_dim = QKV_shape[2] // (self.num_heads*3)
        QKV_shape_d = tf.shape(QKV)
        QKV = tf.reshape(QKV, (QKV_shape_d[0], QKV_shape_d[1], 3,
                               dot_dim, self.num_heads))         # b,l,3dh -> b,l,3,d,h
        QKV.set_shape([QKV_shape[0], QKV_shape[1], 3, dot_dim, self.num_heads]) 
        Q,K,V = tf.unstack(QKV, num=3, axis=2)  # b,l,d,h
        
        # form attention logits from nodes
        A_hat = tf.einsum('bldh,bmdh->blmh', Q, K) # b,l,l,h
        A_hat.set_shape([QKV_shape[0], QKV_shape[1], QKV_shape[1], self.num_heads])
        if self.clip_logits_value is not None:
            A_hat = tf.clip_by_value(A_hat, self.clip_logits_value[0], self.clip_logits_value[1])
        
        # update attention logits with edges
        H_hat = A_hat                         # b,l,l,h
        if self.edge_input: H_hat = H_hat + E # b,l,l,h
        
        # update attention logits with masks
        H_hat_ = H_hat
        
        if mask is not None:
            mask_ = (tf.cast(mask[:,None,:,None], H_hat.dtype) - 1) * 1e9 # b,l -> b,1,l,1
            H_hat_ = H_hat_ + mask_
            
            
        if self.attn_mask:
            if not M.dtype is H_hat.dtype:
                M = tf.cast(M, H_hat.dtype)
            M_ = (M - 1) * 1e9
            H_hat_ = H_hat_ + M_
            
        
        if self.random_mask_prob > 0.0 and training:
            uniform_noise = tf.random.uniform(tf.shape(H_hat_), 
                                              minval=0., maxval=1., dtype=H_hat_.dtype)
            random_mask_ = tf.where(uniform_noise < self.random_mask_prob, -1e9, 0.)
            H_hat_ = H_hat_ + random_mask_
            
        
        # form attention weights
        A_tild = tf.nn.softmax(H_hat_, axis=2) # b,l,l,h
        
        # attention output
        if self.attn_dropout > 0.0 and training:
            A_tild = tf.nn.dropout(A_tild, self.attn_dropout)
        
        # form output
        V_att = tf.einsum('blmh,bmdh->bldh', A_tild, V) # b,l,d,h
        
        # reshape output
        V_att = tf.reshape(V_att, (QKV_shape_d[0], QKV_shape_d[1],
                                   dot_dim*self.num_heads)) # b,l,dh
        V_att.set_shape([QKV_shape[0], QKV_shape[1], dot_dim*self.num_heads])
        
        return V_att, H_hat, A_tild
            
    def compute_mask(self, inputs, mask=None):
        if hasattr(mask, '__getitem__'): mask = mask[0]
        return [mask, None, None]
            
        