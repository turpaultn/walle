import numpy as np
import torch

from utils.utils import pad_trunc_seq, change_view_frames


class ApplyLog(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, added_value=0.01):
        self.added_value = added_value

    def __call__(self, sample):
        """ Apply the transformation
        Args:

        sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample) in [tuple, list]:
            # sample must be a tuple or a list, first parts are input, then last element is label
            if type(sample) is tuple:
                sample = list(sample)
            for i in range(len(sample) - 1):
                if not np.isnan(sample[i]).all():
                    # sample[i] = librosa.amplitude_to_db(sample[i].T).T
                    sample[i] = np.log(sample[i] + self.added_value)
        else:
            if not np.isnan(sample).all():
                # sample = librosa.amplitude_to_db(sample.T).T
                sample = np.log(sample.T + self.added_value).T

        return sample


class Unsqueeze:
    """Unsqueeze axis torch
    Args:
        axis: int, the axis to unsqueeze
    Attributes:
        axis: int, the axis to unsqueeze
    """

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, sample):
        """ Apply the transformation
        Args:

        sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample) in [tuple, list]:
            # sample must be a tuple or a list, first parts are input, then last element is label
            if type(sample) is tuple:
                sample = list(sample)
            for i in range(len(sample) - 1):
                sample[i] = sample[i].unsqueeze(self.axis)
        else:
            sample = sample.unsqueeze(self.axis)

        return sample


class PadOrTrunc:
    """ Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames, pad_mode="zeros"):
        self.nb_frames = nb_frames
        self.pad_mode = pad_mode

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) in [tuple, list]:
            if type(sample) is tuple:
                sample = list(sample)
            # sample must be a tuple or a list
            for k in range(len(sample) - 1):
                sample[k] = pad_trunc_seq(sample[k], self.nb_frames)

            if len(sample[-1].shape) == 2:
                sample[-1] = pad_trunc_seq(sample[-1], self.nb_frames)

        else:
            sample = pad_trunc_seq(sample, self.nb_frames)

        return sample


class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample : tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) in [tuple, list]:
            if type(sample) is tuple:
                sample = list(sample)
            # sample must be a tuple or a list, first parts are input, then last element is label
            for i in range(len(sample)):
                sample[i] = torch.from_numpy(sample[i]).float()  # even labels (we don't loop until -1)
        else:
            sample = torch.from_numpy(sample).float()

        return sample


class NormFrames:
    def __init__(self, dim=-2):
        self.dim = dim

    def __call__(self, sample):
        if type(sample) in [tuple, list]:
            if type(sample) is tuple:
                sample = list(sample)
            # sample must be a tuple or a list, first parts are input, then last element is label
            for i in range(len(sample)):
                sample[i] = sample[i] - sample[i].mean(dim=self.dim, keepdim=True)

        return sample


class Flatten:
    def __init__(self, axis=0, axis_output=None):
        self.axis = axis
        self.axis_output = axis_output

    def __call__(self, sample):
        """ Apply the transformation
                Args:
                    sample: tuple or list, a sample defined by a DataLoad class

                Returns:
                    list
                    The transformed tuple
                """
        if type(sample) in [tuple, list]:
            if type(sample) is tuple:
                sample = list(sample)
            # sample must be a tuple or a list
            for k in range(len(sample) - 1):
                sample[k] = sample[k].flatten(self.axis)
        else:
            sample = sample.flatten(self.axis)

        if self.axis_output is not None:
            sample[-1] = sample[-1].flatten(self.axis_output)

        return sample


class View:
    def __init__(self, view_shape, output_view_shape=None):
        self.view_shape = view_shape
        self.output_view_shape = output_view_shape

    def __call__(self, sample):
        """ Apply the transformation
                Args:
                    sample: tuple or list, a sample defined by a DataLoad class

                Returns:
                    list
                    The transformed tuple
                """
        if type(sample) in [tuple, list]:
            if type(sample) is tuple:
                sample = list(sample)
            # sample must be a tuple or a list
            for k in range(len(sample) - 1):
                sample[k] = sample[k].contiguous().view(self.view_shape)
        else:
            sample = sample.view(self.view_shape)

        if self.output_view_shape is not None:
            sample[-1] = sample[-1].contiguous().view(self.output_view_shape)

        return sample


class Normalize(object):
    """Normalize inputs
    Args:
        scaler: Scaler object, the scaler to be used to normalize the data
    Attributes:
        scaler : Scaler object, the scaler to be used to normalize the data
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple or list, a sample defined by a DataLoad class

        Returns:
            list
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)
        # sample must be a tuple or a list
        for k in range(len(sample) - 1):
            sample[k] = self.scaler.normalize(sample[k])

        return sample


class GetEmbedding:
    """Get an embedding from a CNN trained
        Args:
            cnn_model: derived nn.Module object, the model to get the embedding from.
            nb_frames_to_reshape: int, reshape the input dividing the actual nb of frames by this number of frames and
            apply the model on the created inputs and get back to the original size.
        Attributes:
            cnn_model: derived nn.Module object, the model to get the embedding from.
            nb_frames_to_reshape: int, reshape the input dividing the actual nb of frames by this number of frames and
            apply the model on the created inputs and get back to the original size.
        """
    def __init__(self, cnn_model, nb_frames_to_reshape=None, flatten=False):
        self.triplet_model = cnn_model
        self.nb_frames = nb_frames_to_reshape
        self.flatten = flatten

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample) is tuple:
            sample = list(sample)

        for k in range(len(sample) - 1):
            samp = sample[k]
            if self.nb_frames:
                original_nb_frames = samp.shape[-2]
                samp = samp.unsqueeze(0)
                samp = change_view_frames(samp, self.nb_frames)

            embed = self.triplet_model(samp)

            if self.nb_frames is not None:
                embed = change_view_frames(embed, original_nb_frames)
            embed = embed.squeeze(-1)
            embed = embed.squeeze(0)
            embed = embed.permute(1, 0)  # inverse frames and channel (frames, channel)

            if self.flatten:
                embed = embed.flatten()
            sample[k] = embed.detach()

        return sample


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>> transforms.Scale(),
        >>> transforms.PadTrim(max_len=16000),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


def get_transforms(frames, scaler=None, tensor=True, unsqueeze_axis=0, getembedding=None):
    transforms = [ApplyLog(), PadOrTrunc(nb_frames=frames)]

    if tensor:
        transforms.append(ToTensor())
    if unsqueeze_axis:
        transforms.append(Unsqueeze(unsqueeze_axis))
    if scaler is not None:
        transforms.append(Normalize(scaler=scaler))

    if getembedding is not None:
        transforms.append(getembedding)
    # if flatten:
    #     transforms.append(Flatten())
    return Compose(transforms)
