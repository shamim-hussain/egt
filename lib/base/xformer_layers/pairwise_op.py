import tensorflow as tf
tfk=tf.keras

from .shaping import split_dim, create_heads

op_names = {}
def add_op(name):
    def add_named_op(op):
        op_names[name] = op
        return op
    return add_named_op

@add_op('dot')
def pairwise_dot(row, col, scale=True):
    mat = tf.matmul(row, col, transpose_b=True)
    if scale:
        mat = mat * (row.shape[-1]**-.5)
    return mat

@add_op('l2d')
def pairwise_l2d(row, col, scale=False):
    dotp = tf.matmul(row, col, transpose_b=True)

    row2 = tf.reduce_sum(tf.square(row), axis=-1)
    col2 = tf.reduce_sum(tf.square(col), axis=-1)

    mat = 2*dotp - row2[...,None] - col2[...,None,:]
    if scale:
        mat = mat * (row.shape[-1]**-.5)
    return mat

@add_op('addsub')
def pairwise_addsub(row, col, add=True, sub=True,
                    symmetric=True):
    if not (add or sub):
        raise ValueError('Both add and sub are false')

    rowb = row[...,None,:]
    colb = col[...,None,:,:]

    mats = []
    if add:
        mats.append(rowb+colb)
    if sub:
        if symmetric:
            mats.append(tf.abs(rowb-colb))
        else:
            mats.append(rowb-colb)
    
    if len(mats) == 1:
        out = mats[0]
    else:
        out = tf.concat(mats, axis=-1)
    return out
    
@add_op('cat')
def pairwise_cat(row, col):
    rowb = row[...,None,:]
    colb = col[...,None,:,:]
    
    rm = [1] * rowb.shape.rank
    cm = [1] * colb.shape.rank
    rm[-2] = tf.shape(colb)[-2]
    cm[-3] = tf.shape(rowb)[-3]
    
    rowb = tf.tile(rowb, rm)
    colb = tf.tile(colb, cm)
    
    mat = tf.concat([rowb, colb],axis=-1)
    return mat

class PairwiseOp(tfk.layers.Layer):
    def __init__(self, op='dot', op_kwargs={},
                 num_heads=None, split_axis=-1,
                 **kwargs):
        super().__init__(**kwargs)

        self.op = op
        self.op_kwargs = op_kwargs
        self.num_heads=num_heads
        self.split_axis = split_axis
    
    def get_config(self):
        config=super().get_config().copy()
        config.update(
            op = self.op,
            op_kwargs = self.op_kwargs,
            num_heads = self.num_heads,
        )
        return config

    def call(self,inputs):
        if not isinstance(inputs, list):
            if not self.num_heads is None:
                inputs = create_heads(inputs, self.num_heads)
            row, col = split_dim(inputs, splits=2, axis=self.split_axis)
        else:
            row, col = inputs
            if not self.num_heads is None:
                row = create_heads(row, self.num_heads)
                col = create_heads(col, self.num_heads)

        op_fn = op_names[self.op]
        out = op_fn(row, col, **self.op_kwargs)
        return out
    
