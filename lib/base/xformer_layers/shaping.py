
import tensorflow as tf
tfk = tf.keras


def split_dim(features, splits, axis=-1):
    if isinstance(splits, int):
        splits = [1]*splits
    
    if len(splits) == 1:
        return [features]
    
    splits = list((features.shape[axis]*s)//sum(splits) for s in splits)
    split_feats = tf.split(features, splits, axis=axis)

    static_shape = [None]*features.shape.rank
    for feat, shape in zip(split_feats, splits):
        static_shape[axis] = shape
        feat.set_shape(static_shape)
    
    return split_feats


def swap_dims(features, d1, d2):
    perm = list(range(features.shape.rank))
    perm[d1], perm[d2] = perm[d2], perm[d1]
    output_features = tf.transpose(features, perm=perm)
    return output_features


def move_dim(features, from_dim, to_dim):
    perm = list(range(features.shape.rank))
    dim_to_move = perm[from_dim]
    perm[from_dim] = None
    perm.insert(to_dim, dim_to_move)
    perm.remove(None)
    output_features = tf.transpose(features, perm=perm)
    return output_features



def create_heads(features, num_heads, heads_first=True):
    if num_heads == 1:
        if heads_first:
            output_features = features[:,None,...]
        else:
            output_features = features[...,None,:]
        return output_features


    static_shape = features.shape
    dynamic_shape = tf.shape(features)

    num_features = static_shape[-1]
    assert num_features % num_heads == 0
    num_output_features = num_features // num_heads
    
    new_shape_dynamic = tf.concat([dynamic_shape[:-1],
                                   tf.constant([num_heads, num_output_features])],
                                   axis=0)
    new_shape_static = static_shape[:-1]+[num_heads, num_output_features]

    output_features = tf.reshape(features, new_shape_dynamic)
    output_features.set_shape(new_shape_static)

    if heads_first:
        output_features = move_dim(output_features, from_dim=-2, to_dim=1)
    
    return output_features


def flatten_heads(features, heads_first=True):
    if heads_first and features.shape[1] == 1:
        return tf.squeeze(features, axis=[1])
    elif (not heads_first) and features.shape[-2] == 1:
        return tf.squeeze(features, axis=[-2])

    if heads_first:
        features = move_dim(features, from_dim=1, to_dim=-1)
    
    static_shape = features.shape
    dynamic_shape = tf.shape(features)
    
    num_heads = static_shape[-2]
    num_feats = static_shape[-1]
    num_output_feats = num_heads*num_feats

    new_shape_dynamic = tf.concat([dynamic_shape[:-2],
                                   tf.constant([num_output_feats])],
                                   axis=0)
    new_shape_static = static_shape[:-2]+[num_output_feats]

    output_features = tf.reshape(features, new_shape_dynamic)
    output_features.set_shape(new_shape_static)

    return output_features          
