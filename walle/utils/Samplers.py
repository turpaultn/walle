import random
from copy import deepcopy

import numpy as np
from torch.utils.data import Sampler

from utils.Logger import LOG


class ClusterRandomSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        super(ClusterRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_sizes : list, list of batch sizes that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : Dataset, a Dataset to sample from. Should have a cluster_indices property
        batch_sizes : list, list of batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        super(MultiStreamBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        indices = self.data_source.cluster_indices
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                # tolist, otherwise get an error np.int64 is not JSON serializable
                indices[i] = np.random.permutation(indices[i]).tolist()
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


class CategoriesSampler(Sampler):
    """ From a list of labels, create the batches with the number of labels per class needed, and the number of classes
        Drops the extra items, not fitting into exact batches
        Args:
            serie_labels : pd.Series, a serie containing the labels available
            classes : list, list of classes in the serie labels
            n_per_class : int, the number of data per class wanted
            n_classes: int, number of classes to be seen in each batch
    """

    # In the future try to weight the number of labels taken by the number in the dataset
    def __init__(self, serie_labels, classes, n_per_class, n_classes=None):
        super(CategoriesSampler, self).__init__(serie_labels)
        self.n_per_class = n_per_class
        self.classes = classes
        # self.n_batch = int(len(serie_labels) // (n_per_class * len(classes)))
        self.n_batch = int(serie_labels.value_counts().min() / n_per_class)
        self.serie_labels = serie_labels
        self.n_classes = n_classes
        LOG.debug(f"sampler has: {self.n_batch} batches of {n_per_class} samples per classes, "
                  f"len serie: {len(serie_labels)}")

        self.ind_per_class = []
        for label in classes:
            ind = np.argwhere(serie_labels.str.contains(label)).reshape(-1).tolist()
            if len(ind) > 0:
                self.ind_per_class.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        indexes_per_class = deepcopy(self.ind_per_class)
        for i_batch in range(self.n_batch):
            batch = []
            if self.n_classes is not None:
                kept_ind_classes = random.sample(indexes_per_class, self.n_classes)
            else:
                kept_ind_classes = indexes_per_class
            for classes_ind in kept_ind_classes:
                if len(classes_ind) >= self.n_per_class:
                    batch_ind = random.sample(classes_ind, self.n_per_class)
                    for ii in batch_ind:
                        classes_ind.remove(ii)
                    batch.extend(batch_ind)
            yield batch


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n

    return zip(*args)
