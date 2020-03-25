import time

import numpy as np
import torch
import json
from utils.Logger import LOG


class ScalerSum(object):
    """
    operates on one or multiple existing datasets and applies operations
    """

    def __init__(self, mean=None, mean_of_square=None):
        self.mean_ = mean
        self.mean_of_square_ = mean_of_square
        if self.mean_ is not None and self.mean_of_square_ is not None:
            variance = self.variance(self.mean_, self.mean_of_square_)
            self.std_ = self.std(variance)
        else:
            self.std_ = None

    @staticmethod
    def sum(data, axis=-1):
        """compute the mean incrementaly"""
        # -1 means have at the end a mean vector of the last dimension
        n = 0
        if axis == -1:
            sum_ = data
            while len(sum_.shape) != 1:
                n += sum_.shape[0]
                sum_ = np.sum(sum_, axis=0, dtype=np.float64)
        else:
            n = data.shape[axis]
            sum_ = np.sum(data, axis=axis, dtype=np.float64)
        return sum_, n

    @staticmethod
    def variance(mean, mean_of_square):
        """compute variance thanks to mean and mean of square"""
        return mean_of_square - mean**2

    def means(self, dataset):
        """
       Splits a dataset in to train test validation.
       :param dataset: dataset, from DataLoad class, each sample is an (X, y) tuple.
       """
        LOG.info('computing mean')
        start = time.time()
        sum_ = 0
        sum_square = 0
        n = 0
        n_sq = 0

        for sample in dataset:
            if type(sample) in [tuple, list] and len(sample) == 2:
                batch_x, _ = sample
            else:
                batch_x = sample
            if type(batch_x) is torch.Tensor:
                batch_x_arr = batch_x.numpy()
            else:
                batch_x_arr = batch_x

            su, nn = self.sum(batch_x_arr, axis=-1)
            sum_ += su
            n += nn

            su_sq, nn_sq = self.sum(batch_x_arr ** 2, axis=-1)
            sum_square += su_sq
            n_sq += nn_sq

        self.mean_ = sum_ / n
        self.mean_of_square_ = sum_square / n_sq

        LOG.debug('time to compute means: ' + str(time.time() - start))
        return self

    @staticmethod
    def std(variance):
        if (variance < 0).any():
            print(variance)
        return np.sqrt(variance + np.finfo("float").eps)

    def calculate_scaler(self, dataset):
        self.means(dataset)
        variance = self.variance(self.mean_, self.mean_of_square_)
        self.std_ = self.std(variance)

        return self.mean_, self.std_

    def normalize(self, batch):
        if type(batch) is torch.Tensor:
            batch_ = batch.numpy()
            batch_ = (batch_ - self.mean_) / (self.std_ + np.finfo("float").eps)
            return torch.Tensor(batch_)
        else:
            return (batch - self.mean_) / (self.std_ + np.finfo("float").eps)

    def state_dict(self):
        if type(self.mean_) is not np.ndarray:
            raise NotImplementedError("Save scaler only implemented for numpy array means_")

        dict_save = {"mean_": self.mean_.tolist(),
                     "mean_of_square_": self.mean_of_square_.tolist()}
        return dict_save

    def save(self, path):
        dict_save = self.state_dict()
        with open(path, "w") as f:
            json.dump(dict_save, f)

    @classmethod
    def load_state_dict(cls, state_dict):
        mean_ = np.array(state_dict["mean_"])
        mean_of_square_ = np.array(state_dict["mean_of_square_"])

        return cls(mean=mean_, mean_of_square=mean_of_square_)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            dict_save = json.load(f)

        return cls.load_state_dict(dict_save)
