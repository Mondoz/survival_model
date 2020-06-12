import sys
sys.path.append("/ps2/cv4/qytian/program/mxnet-medical-20180302-dev-v6/python")
import mxnet as mx
import numpy as np
import pandas as pd
import os
'''
test_csv = os.path.join(os.getcwd(), 'data', 'test_csv.csv')
train_csv = os.path.join(os.getcwd(), 'data', 'train_csv.csv')
train = [x for x in open(train_csv).readlines()]
'''
class TrainLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=32, shuffle=True):
        super(TrainLoader, self).__init__()

        # save parameters as properties
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.roidb = roidb
        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)


        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None

        # get first batch to fill in provide_data and provide_label
        self.reset()
        # self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            return self.get_batch()

        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        batch = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        self.cur += self.batch_size
        return batch

class Train_random_Loader(mx.io.DataIter):
    def __init__(self, roidb,size, batch_size=32, shuffle=True):
        super(Train_random_Loader, self).__init__()

        # save parameters as properties
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.roidb = roidb
        # infer properties from roidb
        self.size = size
        self.index = np.arange(self.size)


        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None

        # get first batch to fill in provide_data and provide_label
        #self.reset()
        # self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            return self.get_batch()

        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        o_batch = np.random.choice(self.roidb, self.batch_size, replace= False)
        batch = [x for x in o_batch]
        self.cur += self.batch_size
        return batch
        
class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, batch_size=32, shuffle=True):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.roidb = roidb
        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)


        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None

        # get first batch to fill in provide_data and provide_label
        #self.reset()
        # self.get_batch()

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        if self.iter_next():
            return self.get_batch()

        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        batch = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        self.cur += self.batch_size
        return batch
if __name__ == '__main__':

    loader = TestLoader(train)