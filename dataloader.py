from torch.utils.data.sampler import Sampler
import numpy as np


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, chunk_size, train=True):
        self.dataset_length = len(dataset)
        self.labels = np.zeros(len(dataset), dtype=np.int8)
        self.chunk_size = chunk_size
        self.num_instances = 500 if train else 100
        for i in range(len(dataset)):
            self.labels[i] = dataset[i][1]
        
        indices = np.reshape(np.argsort(self.labels), (100, self.num_instances)).transpose()
        np.random.shuffle(indices)
        self.indices = np.transpose(indices)
        np.random.shuffle(self.indices)
        
       

        assert batch_size % chunk_size == 0
        assert self.num_instances % chunk_size == 0

    def __iter__(self):
        iter_list = None

        for i in range(0,self.num_instances,self.chunk_size):
            if iter_list is None:
                iter_list = self.indices[:, i:i+4]
            else:
                iter_list = np.vstack((iter_list, self.indices[:, i:i+4]))
        return iter(iter_list.flatten().tolist())

    def __len__(self):
        return self.dataset_length