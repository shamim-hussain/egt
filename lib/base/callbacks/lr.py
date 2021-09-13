
from tensorflow.keras import callbacks


class WarmUp(callbacks.Callback):
    def __init__(self, num_batches=10000, final_lr=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.num_batches=num_batches
        self.final_lr=final_lr
        self.batch_count=0
    
    def on_batch_begin(self, batch, logs=None):
        if self.batch_count > self.num_batches:
            return
        self.batch_count += 1
        self.model.optimizer.lr.assign(self.final_lr*self.batch_count/
                                       self.num_batches)
    
    def on_epoch_end(self, epoch, logs=None):
        print(f'Current lr = {self.model.optimizer.lr.numpy()}',flush=True)