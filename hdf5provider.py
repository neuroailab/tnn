"""
data provider from hdf5 file suitable for use with tensorflow models
"""
import math
import numpy as np
import h5py
from yamutils import fast

class HDF5DataProvider(object):
    def __init__(self,
                 hdf5source,
                 sourcelist,
                 batch_size,
                 subslice=None,
                 mini_batch_size=None,
                 preprocess=None,
                 postprocess=None,
                 pad=False):
        self.hdf5source = hdf5source
        self.sourcelist = sourcelist
        self.file = h5py.File(self.hdf5source, 'r')
        self.subslice = subslice
        self.subsliceinds = None
        if preprocess is None:
            preprocess = {}
        self.preprocess = preprocess
        if postprocess is None:
            postprocess = {}
        self.postprocess = postprocess
        self.data = {}
        self.sizes = {}
        for source in self.sourcelist:
            self.data[source] = self.file[source]
            if source in self.preprocess:
                print('Preprocessing %s ...' % source)
                self.data[source] = self.preprocess[source](self.data[source])
                print('... done')
        for source in sourcelist:
            if self.subslice is None:
                self.sizes[source] = self.data[source].shape
            else:
                if self.subsliceinds is None:
                    if isinstance(self.subslice, str):
                        self.subsliceinds = self.file[self.subslice][:]
                    elif hasattr(self.subslice, '__call__'):
                        self.subsliceinds = self.subslice(self.file, self.sourcelist)
                    else:
                        self.subsliceinds = self.subslice[:]
                    self.subsliceinds = self.subsliceinds.nonzero()[0]
                sz = self.data[source].shape
                self.sizes[source] = (self.subsliceinds.shape[0],) + sz[1:]
            if not hasattr(self, 'data_length'):
                self.data_length = self.sizes[source][0]
            assert self.sizes[source][0] == self.data_length, (self.sizes[source], self.data_length)
        self.batch_size = batch_size
        if mini_batch_size is None:
            mini_batch_size = self.batch_size
        self.mini_batch_size = mini_batch_size
        self.total_batches = int(math.ceil(self.data_length / float(self.batch_size)))
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad

    def setEpochBatch(self, epoch, batch_num):
        self.curr_epoch = epoch
        self.curr_batch_num = batch_num

    def getNextBatch(self):
        data = self.getBatch(self.curr_batch_num)
        self.incrementBatchNum()
        return data
    
    def incrementBatchNum(self):
        m = self.total_batches
        if (self.curr_batch_num >= m-1):
            self.curr_epoch += 1
        self.curr_batch_num = (self.curr_batch_num + 1) % m
            
    def getBatch(self, cbn):
        data = {}
        startv = cbn * self.batch_size 
        endv = (cbn + 1) * self.batch_size
        if self.pad and endv > self.data_length:
            startv = self.data_length - self.batch_size
            endv = startv + self.batch_size
        sourcelist = self.sourcelist
        for source in sourcelist:
            data[source] = self.getData(self.data[source], slice(startv, endv))
            if self.postprocess.has_key(source):
                data[source] = self.postprocess[source](data[source], self.file)
        return data
       
    def getData(self, dsource, sliceval):
        if self.subslice is None:
            return dsource[sliceval]
        else:
            subslice_inds = self.subsliceinds[sliceval]
            mbs = self.mini_batch_size
            bn0 = subslice_inds.min() / mbs
            bn1 = subslice_inds.max() / mbs
            stims = []
            for _bn in range(bn0, bn1 + 1):
                _s = np.asarray(dsource[_bn * mbs: (_bn + 1) * mbs])
                new_inds = fast.isin(np.arange(_bn * mbs, (_bn + 1) * mbs), subslice_inds)
                new_array = _s[new_inds]
                stims.append(new_array)
            stims = np.concatenate(stims)
            return stims


def get_unique_labels(larray):
    larray = larray[:]
    labels_unique = np.unique(larray)
    s = larray.argsort()
    cat_s = larray[s]
    ss = np.array([0] + ((cat_s[1:] != cat_s[:-1]).nonzero()[0] + 1).tolist() + [len(cat_s)])
    ssd = ss[1:] - ss[:-1]
    labels = np.repeat(np.arange(len(labels_unique)), ssd)
    larray = labels[fast.perminverse(s)]
    return larray.astype(np.int64)
