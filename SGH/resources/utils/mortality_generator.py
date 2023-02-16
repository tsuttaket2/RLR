from resources.utils import common_utils
import threading
import os
import numpy as np
import random


class BatchGen(object):

    def __init__(self, reader, batch_size, steps, 
                 shuffle, return_names=False):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        if steps is None:
            self.n_examples = reader.get_number_of_examples()
            self.steps = (self.n_examples + batch_size - 1) // batch_size
        else:
            self.n_examples = steps * batch_size
            self.steps = steps

        self.chunk_size = min(1024, self.steps) * batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()
        self.timestep = reader.timestep
    def _generator(self):
        B = self.batch_size
        
        def get_bin(t):
            eps = 1e-6
            return int(t / self.timestep - eps)
        
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size
                
                ret = common_utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]
                
                masks=[]
                masks_T=[]
                for i,T in enumerate(ts):
                    mask_temp=np.ones((get_bin(T) + 1),dtype=('float32'))
                        
                    masks_T_temp=np.zeros((get_bin(T) + 1),dtype=('float32'))
                    masks_T_temp[-1]=1
                    assert masks_T_temp.shape[0]==Xs[i].shape[0]
                    masks_T.append(masks_T_temp)
                    masks.append(mask_temp)
                (Xs, ys, ts, masks, masks_T, names) = common_utils.sort_and_shuffle([Xs, ys, ts, masks, masks_T, names], B)
                
                for i in range(0, current_size, B):
                    X = common_utils.pad_zeros(Xs[i:i + B])
                    batch_mask = common_utils.pad_zeros(masks[i:i + B])
                    batch_mask_T = common_utils.pad_zeros(masks_T[i:i + B])
                    y = np.array(ys[i:i + B])
                    batch_names = names[i:i+B]
                    batch_ts = ts[i:i+B]
                    batch_data = (X, y)
                    if not self.return_names:
                        yield batch_data
                    else:
                        yield {"data": batch_data, "names": batch_names, "ts": batch_ts, "masks": batch_mask, "masks_T": batch_mask_T}
                
    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()
    
class BatchGen_Fulldata(object):

    def __init__(self, reader, batch_size, steps, 
                 shuffle, return_names=False):
        self.reader = reader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_names = return_names

        self.n_examples = reader.get_number_of_examples()
        self.steps = (self.n_examples + batch_size - 1) // batch_size
        
        self.chunk_size = batch_size
        self.lock = threading.Lock()
        self.generator = self._generator()
        self.timestep = reader.timestep
    def _generator(self):
        B = self.batch_size
        
        def get_bin(t):
            eps = 1e-6
            return int(t / self.timestep - eps)
        
        while True:
            if self.shuffle:
                self.reader.random_shuffle()
            remaining = self.n_examples
            while remaining > 0:
                current_size = min(self.chunk_size, remaining)
                remaining -= current_size
                
                ret = common_utils.read_chunk(self.reader, current_size)
                Xs = ret["X"]
                ts = ret["t"]
                ys = ret["y"]
                names = ret["name"]
                
                masks=[]
                masks_T=[]
                for i,T in enumerate(ts):
                    mask_temp=np.ones((get_bin(T) + 1),dtype=('float32'))
                        
                    masks_T_temp=np.zeros((get_bin(T) + 1),dtype=('float32'))
                    masks_T_temp[-1]=1
                    assert masks_T_temp.shape[0]==Xs[i].shape[0]
                    masks_T.append(masks_T_temp)
                    masks.append(mask_temp)
                (Xs, ys, ts, masks, masks_T, names) = common_utils.sort_and_shuffle([Xs, ys, ts, masks, masks_T, names], B)
                
                for i in range(0, current_size, B):
                    X = common_utils.pad_zeros(Xs[i:i + B])
                    batch_mask = common_utils.pad_zeros(masks[i:i + B])
                    batch_mask_T = common_utils.pad_zeros(masks_T[i:i + B])
                    y = np.array(ys[i:i + B])
                    batch_names = names[i:i+B]
                    batch_ts = ts[i:i+B]
                    batch_data = (X, y)
                    if not self.return_names:
                        yield batch_data
                    else:
                        yield {"data": batch_data, "names": batch_names, "ts": batch_ts, "masks": batch_mask, "masks_T": batch_mask_T}
                
    def __iter__(self):
        return self.generator

    def next(self):
        with self.lock:
            return next(self.generator)

    def __next__(self):
        return self.next()