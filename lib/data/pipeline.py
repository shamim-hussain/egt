
import tensorflow as tf

class AddLabel:
    def __init__(self                ,
                 label               ,
                 input_name = None   ,
                 label_name = 'target'):
        self.label        = label
        self.input_name   = input_name
        self.label_name   = label_name
    
    def __call__(self, inputs):
        if self.input_name is None:
            return {
                **inputs,
                self.label_name : self.label
            }
        else:
            return {
                self.input_name : inputs,
                self.label_name : self.label,
            }

class SelectFeatures:
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def __call__(self, inputs):
        outputs = dict((k,inputs[k]) for k in self.feature_names)
        return outputs

class ExcludeFeatures:
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def __call__(self, inputs):
        outputs = dict((k, inputs[k]) for k in inputs 
                                 if k not in self.feature_names)
        return outputs

class MergeFeatures:
    def __init__(self, verify_key=None):
        self.verify_key = verify_key
    def __call__(self, *inputs):
        outputs = {}

        if self.verify_key is not None:
            ref = inputs[0][self.verify_key]
            for k in range(1,len(inputs)):
                tf.assert_equal(ref, inputs[k][self.verify_key])
            outputs[self.verify_key] = ref
            outputs.update((k,v) for d in inputs for k,v in d.items()
                                                if k != self.verify_key)
        else:
            outputs.update((k,v) for d in inputs for k,v in d.items())
        
        return outputs

class CreateTargets:
    def __init__(self, target_names):
        if not (isinstance(target_names, list) 
                      or isinstance(target_names, tuple)):
            target_names = [target_names]

        self.target_names = target_names

    def __call__(self, inputs):
        X = dict((k, inputs[k]) for k in inputs 
                                 if k not in self.target_names)
        Y = dict((k, inputs[k]) for k in inputs 
                                 if k in self.target_names)
        return (X, Y)


