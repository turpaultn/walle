import os
import warnings

import librosa
import numpy as np
import pandas as pd
import soundfile
import math
import torch
from dcase_util.data import DecisionEncoder
from torch import nn
from torch.optim.optimizer import Optimizer

from models.CombineModel import CombineModel


class ManyHotEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """
    def __init__(self, labels, n_frames=None):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.n_frames = n_frames

    def encode_weak(self, labels):
        """ Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        if labels is None:
            y = np.zeros(len(self.labels), dtype=np.float32) - 1
            return y

        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        y = np.zeros(len(self.labels), dtype=np.float32)
        for label in labels:
            if not pd.isna(label):
                i = self.labels.index(label)
                y[i] = 1
        return y

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """

        assert self.n_frames is not None, "n_frames need to be specified when using strong encoder"
        if label_df is None:
            y = np.zeros((self.n_frames, len(self.labels))) - 1
            return y

        y = np.zeros((self.n_frames, len(self.labels)))
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        i = self.labels.index(row["event_label"])
                        onset = int(row["onset"])
                        offset = int(row["offset"])
                        y[onset:offset, i] = 1  # means offset not included (hypothesis of overlapping frames, so ok)

        elif type(label_df) in [pd.Series, list, np.ndarray]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(label_df.index):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        i = self.labels.index(label_df["event_label"])
                        onset = int(label_df["onset"])
                        offset = int(label_df["offset"])
                        y[onset:offset, i] = 1
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label is not "":
                        i = self.labels.index(event_label)
                        y[:, i] = 1

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] is not "":
                        i = self.labels.index(event_label[0])
                        onset = int(event_label[1])
                        offset = int(event_label[2])
                        y[onset:offset, i] = 1

                else:
                    raise NotImplementedError("cannot encode strong, type mismatch: {}".format(type(event_label)))

        else:
            raise NotImplementedError("To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                                      "columns, or it is a list or pandas Series of event labels, "
                                      "type given: {}".format(type(label_df)))
        return y

    def decode_weak(self, labels):
        """ Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels):
        """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        result_labels = []
        for i, label_column in enumerate(labels.T):
            change_indices = DecisionEncoder().find_contiguous_regions(label_column)

            # append [label, onset, offset] in the result list
            for row in change_indices:
                result_labels.append([self.labels[i], row[0], row[1]])
        return result_labels

    def state_dict(self):
        return {"labels": self.labels,
                "n_frames": self.n_frames}

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        n_frames = state_dict["n_frames"]
        return cls(labels, n_frames)


def unique_classes(df):
    if type(df) == pd.Series:
        df = pd.DataFrame(df)
    if "event_label" in df.columns:
        unique_cls = df["event_label"].dropna().unique()  # dropna avoid the issue between string and float
    elif "event_labels" in df.columns:
        unique_cls = df.event_labels.str.split(',', expand=True).unstack().dropna().unique()
    else:
        unique_cls = []
        warnings.warn("No class in this dataframe, assuming it is unlabel: \n {}".format(df.head()))

    return unique_cls


def read_audio(path, target_fs=None, **kwargs):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path, **kwargs)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def write_audio(path, audio, sample_rate):
    """ Save an audio file
    Args:
        path: str, path used to store the audio
        audio: numpy.array, audio data to store
        sample_rate: int, the sampling rate
    """
    soundfile.write(file=path, data=audio, samplerate=sample_rate)


def create_folder(fd):
    """ Create folders of a path if not exists
    Args:
        fd: str, path to the folder to create
    """
    if not os.path.exists(fd):
        os.makedirs(fd)


def name_only(path):
    return os.path.splitext(os.path.basename(path))[0]


def pad_trunc_seq(x, max_len, pad_mode="zeros"):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.
      pad_mode: str (in ["zeros", "repeat"]), the way to pad the sequences

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    length = len(x)
    shape = x.shape
    if length < max_len:
        pad_shape = (max_len - length,) + shape[1:]
        if pad_mode == "zeros":
            pad = np.zeros(pad_shape)
        elif pad_mode == "repeat":
            pad = x[:pad_shape]
        else:
            raise NotImplementedError("only 'zeros' and 'repeat' pad_mode are implemented")
        x_new = np.concatenate((x, pad), axis=0)
    elif length > max_len:
        x_new = x[0:max_len]
    else:
        x_new = x
    return x_new


def weights_init(m):
    """ Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    # classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
        # m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.GRU):
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight.data)
        nn.init.constant(m.bias, 0)


def to_cuda_if_available(*args):
    """ Transfer object (Module, Tensor) to GPU if GPU available
    Args:
        args: objects to put on cuda if available

    Returns:
        Objects on GPU if GPUs available
    """
    res = list(args)
    if torch.cuda.is_available():
        for i, torch_obj in enumerate(args):
            res[i] = torch_obj.cuda()
    if len(res) == 1:
        return res[0]
    return res


def to_cpu(*args):
    """ Transfer object (Module, Tensor) to CPU if GPU available
        Args:
            args: objects to put on CPU (if not already)

        Returns:
            Objects on CPU
        """
    res = list(args)
    if torch.cuda.is_available():
        for i, torch_obj in enumerate(args):
            res[i] = torch_obj.cpu()

    if len(res) == 1:
        return res[0]
    return res


class SaveBest:
    """ Callback to get the best model
    Args:
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, val_comp="inf"):
        self.comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.best_epoch = 0
        self.current_epoch = 0

    def apply(self, value):
        """ Apply the callback
        Args:
            value: float, the value of the metric followed
        """
        decision = False
        if self.current_epoch == 0:
            decision = True
        if (self.comp == "inf" and value < self.best_val) or (self.comp == "sup" and value > self.best_val):
            self.best_epoch = self.current_epoch
            self.best_val = value
            decision = True
        self.current_epoch += 1
        return decision


class EarlyStopping:
    """ Callback of a model to store the best model based on a criterion
    Args:
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """
    def __init__(self, patience, val_comp="inf", init_patience=0):
        self.patience = patience
        self.first_early_wait = init_patience
        self.val_comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.current_epoch = 0
        self.best_epoch = 0

    def apply(self, value):
        """ Apply the callback

        Args:
            value: the value of the metric followed
        """
        current = False
        if self.val_comp == "inf":
            if value < self.best_val:
                current = True
        if self.val_comp == "sup":
            if value > self.best_val:
                current = True
        if current:
            self.best_val = value
            self.best_epoch = self.current_epoch
        elif self.current_epoch - self.best_epoch > self.patience and self.current_epoch > self.first_early_wait:
            self.current_epoch = 0
            return True
        self.current_epoch += 1
        return False


def change_view_frames(array, nb_frames):
    array = array.view(-1, array.shape[1], nb_frames, array.shape[-1])
    return array


def save_model(state, filename=None, overwrite=True):
    """ Function to save Pytorch models.
    # Argument
        dic_params: Dict. Must includes "model" (and possibly "optimizer")which is a dict with "name","args","kwargs",
        example:
        state = {
                 'epoch': epoch_ + 1,
                 'model': {"name": classif_model.get_name(),
                           'args': rnn_args,
                           "kwargs": rnn_kwargs,
                           'state_dict': classif_model.state_dict()},
                 'optimizer': {"name": classif_model.get_name(),
                               'args': '',
                               "kwargs": optim_kwargs,
                               'state_dict': optimizer_classif.state_dict()},
                 'loss': loss_mean_bce
                 }
        filename: String. Where to save the model.
        overwrite: Bool. Whether to overwrite existing file or not.
    # Raises
        Warning if filename exists and overwrite isn't set to True.
    """
    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
            torch.save(state, filename)
        else:
            warnings.warn('Found existing file at {}'.format(filename) +
                          'specify `overwrite=True` if you want to overwrite it')
    else:
        torch.save(state, filename)


# Not in the pytorch version I'm using and stuck at for my GPU, so copy pasting it here
class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
# ##################
# MANDATORY FOR get_class to work !!!!!!
# It puts the name of the class in globals()
# ###################
from models.CNN import CNN
from models.FullyConnected import FullyConnected
from torch.optim import *


def get_class(name):
    try:
        cls = globals()[name]
    except KeyError as ke:
        raise KeyError("Impossible to load the object from a string, check the class name and "
                       "if the situation is covered" + str(ke))
    return cls


def load_optimizer(state, model):
    assert "optimizer" in state, "in load_model, to return optimizer, it should be in the saved file"
    optimizer = get_class(state["optimizer"]["name"])(filter(lambda p: p.requires_grad, model.parameters()),
                                                      *state['optimizer']['args'],
                                                      **state['optimizer']['kwargs'])
    optimizer.load_state_dict(state['optimizer']["state_dict"])
    return optimizer


def load_model_from_state(state, return_optimizer=False):
    model = get_class(state["model"]["name"])(*state['model']['args'],
                                              **state['model']['kwargs'])
    # Assuming we load model
    model.load(parameters=state['model']["state_dict"])

    if return_optimizer:
        optimizer = load_optimizer(state, model)
        return model, optimizer
    return model


def load_model(filename, return_optimizer=False, return_state=False, **kwargs):
    """ Function to load Pytorch models.
    # Argument
        filename: String. Where to load the model from.
        example:
        state = {
                 'epoch': epoch_ + 1,
                 'model': {"name": classif_model.get_name(),
                           'args': rnn_args,
                           "kwargs": rnn_kwargs,
                           'state_dict': classif_model.state_dict()},
                 'optimizer': {"name": classif_model.get_name(),
                               'args': '',
                               "kwargs": optim_kwargs,
                               'state_dict': optimizer_classif.state_dict()},
                 'loss': loss_mean_bce
                 }
        return_optimizer: Bool. Whether to return optimizer or not.
        kwargs: arguments taken in torch.load
    # Returns
        A Pytorch model instance.
        An Pytorch optimizer instance.
    """
    state = torch.load(filename, **kwargs)
    if type(state["model"]["name"]) == list:
        submodels = []
        for name in state["model"]["name"]:
            model = get_class(name)(*state['model']['args'][name],
                                    **state['model']['kwargs'][name])
            try:
                # Assuming we load model
                model.load(parameters=state['model']["state_dict"][name])
            except AttributeError:
                warnings.warn("A module doesn't have load implemented, so not loaded")
            submodels.append(model)
        model = CombineModel(*submodels)
    else:
        model = load_model_from_state(state, return_optimizer=False)

    model = to_cuda_if_available(model)
    res = model
    if return_optimizer:
        optimizer = load_optimizer(state, model)
        res = model, optimizer
        if return_state:
            res = model, optimizer, state
    else:
        if return_state:
            res = model, state
    return res


def number_of_parameters(model):
    total = 0
    tensor_list = list(model.state_dict().items())
    for layer_tensor_name, tensor in tensor_list:
        if tensor.requires_grad:
            print('Layer {:20}: \t nb_elements: {:5}'.format(layer_tensor_name, torch.numel(tensor)))
            total += torch.numel(tensor)
    # return (sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total


class ViewModule(nn.Module):
    def __init__(self, shape):
        super(ViewModule, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(self.shape)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        tt = torch.cuda.FloatTensor
    else:
        tt = torch.FloatTensor
    return (pred == label).type(tt).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
