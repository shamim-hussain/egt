
import tensorflow as tf
tfk = tf.keras
from .shaping import move_dim



def move_ch2h(maybe_headed_tensor,
              channels_dim=-1, head_dim=1):
    if maybe_headed_tensor.shape.rank == 4:
        return move_dim(maybe_headed_tensor,
                        from_dim=channels_dim,
                        to_dim=head_dim)
    else:
        return maybe_headed_tensor


def merge_attention_heads(merge_type, headed_tensor):
    if merge_type == 'mean':
        return tf.reduce_mean(headed_tensor, axis=1)
    elif merge_type == 'max':
        return tf.reduce_max(headed_tensor, axis=1)
    elif merge_type == 'sum':
        return tf.reduce_sum(headed_tensor, axis=1)
    elif merge_type == 'prod':
        return tf.reduce_prod(headed_tensor, axis=1)
    else:
        raise ValueError(f'Unknown merge type "{merge_type}"')


def dot_product_attention(query, key, value, 
                          mask                = None,
                          attn_mask           = None,
                          scale_factor        = None,
                          bias                = None,
                          scale_logits        = True,
                          clip_logits_value   = None,
                          causal              = False,
                          pad                 = False,
                          merge_heads         = None,
                          attn_scale_factor   = None,
                          return_logits       = False,
                          return_matrix       = False,
                          big_number          = 1e9
                          ):

    query_shape = query.shape
    key_shape = key.shape
    value_shape = value.shape
    input_rank = query_shape.rank

    attention_dim = query_shape[-1]
    
    if pad:
        paddings = [(0,0)]*(input_rank-2) + [(1,0),(0,0)]
        key = tf.pad(key, paddings)
        value = tf.pad(value, paddings)

    # Create Priliminary Logits
    attention_logits = tf.matmul(query, key, transpose_b=True)


    # Scaling for dot product
    if scale_logits:
        attention_logits = attention_logits*(attention_dim**-.5)
    
    
    # Clipping for numerical stability
    if clip_logits_value is not None:
        if not isinstance(clip_logits_value, list):
            if isinstance(clip_logits_value, tuple):
                clip_logits_value = list(clip_logits_value)
            else:
                clip_logits_value = [-clip_logits_value, clip_logits_value, 0]
        if len(clip_logits_value) == 2:
            clip_logits_value.append(0)
        if len(clip_logits_value) < 3:
            raise ValueError
    
    # Clip before
    if clip_logits_value is not None and (not clip_logits_value[2]):
        attention_logits = tf.clip_by_value(attention_logits, *clip_logits_value[:2])

    # Scale factor and bias
    if scale_factor is not None:
        scale_factor = move_ch2h(scale_factor)
        attention_logits = attention_logits * scale_factor

    if bias is not None:
        bias = move_ch2h(bias)
        attention_logits = attention_logits + bias
    
    # Save for returning the logits
    logits_matrix = attention_logits

    # Clip after
    if clip_logits_value is not None and clip_logits_value[2]:
        attention_logits = tf.clip_by_value(attention_logits, *clip_logits_value[:2])

    # Masking
    if not mask is None:
        mask_rank = mask.shape.rank

        mask_slice = [Ellipsis]+[None]*(input_rank-mask_rank)+[slice(None)]
        mask = mask[mask_slice]

        if not mask.dtype is attention_logits.dtype:
            mask = tf.cast(mask, attention_logits.dtype)
        attention_logits = attention_logits + (mask-1)*big_number
    
    if not attn_mask is None:
        attn_mask = move_ch2h(attn_mask)
        if not attn_mask.dtype is attention_logits.dtype:
            attn_mask = tf.cast(attn_mask, attention_logits.dtype)
        attention_logits = attention_logits + (attn_mask-1)*big_number
        
    if causal:
        causal_mask_shape = [query.shape[-2], key.shape[-2]]
        if None in causal_mask_shape:
            causal_mask_shape = tf.shape(attention_logits)[-2:]

        causal_mask = tf.ones(causal_mask_shape,
                               dtype=attention_logits.dtype)
        causal_mask = tf.linalg.band_part(causal_mask,-1,0)
        attention_logits = attention_logits + (causal_mask-1)*big_number
    
    
    # Softmax Attention
    attention_matrix = tf.nn.softmax(attention_logits, axis=-1)
    
    # Merge Heads
    if merge_heads is not None:
        attention_matrix = merge_attention_heads(merge_type=merge_heads,
                                                 headed_tensor=attention_matrix)
    
    # Scale Attention Matrix
    if attn_scale_factor is not None:
        attn_scale_factor = move_ch2h(attn_scale_factor)
        attention_matrix = attention_matrix * attn_scale_factor
    
    output = tf.matmul(attention_matrix, value)
    
    if merge_heads is None:
        output.set_shape(query_shape[:-1]+value_shape[-1:])
    else:
        output.set_shape(query_shape[0:1]+query_shape[2:-1]+value_shape[-1:])


    # Format Outputs
    outputs = output

    if return_logits or return_matrix:
        outputs = (outputs,)
    
    if return_logits:
        logits = move_dim(logits_matrix, from_dim=1, to_dim=4)
        outputs = outputs + (logits,)
    
    if return_matrix:
        outputs = outputs + (attention_matrix,)

    return outputs
