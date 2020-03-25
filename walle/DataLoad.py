import bisect
import glob
import json

import numpy as np
import pandas as pd
import torch
import random
import os
import warnings

from filelock import FileLock
from torch.utils.data import Dataset

from utils.utils import unique_classes

torch.manual_seed(0)
random.seed(0)


class DataLoadDf(Dataset):
    """ Class derived from pytorch Dataset
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels),
            it should contain these columns :
            "filename"
            "filename", "event_labels"
            "filename", "onset", "offset", "event_label"
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set infromation (filenames, labels, ...)
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """
    def __init__(self, df, encode_function, transform=None,
                 return_indexes=False, return_audioset=False, in_memory=True):
        self.df = df
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.filenames = df.filename.drop_duplicates()
        if return_audioset:
            indexes = self.filenames.index
            self.filenames_audioset = df.audioset.loc[indexes]
        self.return_audioset = return_audioset
        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
        self.unique_labels = unique_classes(df)

    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes

        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.filenames)
        return length

    def get_label(self, index):
        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns:
            label = self.df.iloc[index]["event_labels"]
            if type(label) is str:
                if label == "":
                    label = []
                else:
                    label = label.split(",")
            elif pd.isna(label):
                label = []
        elif {"onset", "offset", "event_label"}.issubset(self.df.columns):
            cols = ["onset", "offset", "event_label"]
            label = self.df[self.df.filename == self.filenames.iloc[index]][cols]
            if label.empty:
                label = []
        else:
            label = None
            if "filename" not in self.df.columns:
                raise NotImplementedError(
                    "Dataframe to be encoded doesn't have specified columns: columns allowed: 'filename' for unlabeled;"
                    "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                    "for strong labels, yours: {}".format(self.df.columns))

        return label

    def get_feature_file_func(self, filename):
        """Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        if not self.in_memory:
            data = np.load(filename).astype(np.float32)
        else:
            if self.features.get(filename) is None:
                data = np.load(filename).astype(np.float32)
                self.features[filename] = data
            else:
                data = self.features[filename]
        return data

    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)

        """
        features = self.get_feature_file_func(self.filenames.iloc[index])

        label = self.get_label(index)

        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
        else:
            y = label

        sample = features, y
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)

        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)

        return sample

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def modify_object(self, df=None, encode_function=None, transform=None, return_indexes=None):
        if df is None:
            df = self.df
        if encode_function is None:
            encode_function = self.encode_function
        if transform is None:
            transform = self.transform
        if return_indexes is None:
            return_indexes = self.return_indexes

        return self.__class__(df=df, encode_function=encode_function, transform=transform,
                              return_indexes=return_indexes)


class DataLoadDfTripletLabeled(DataLoadDf):
    """ Class derived from DataLoadDfTripletAbstract
        Prepare the data to be use in a batch mode with triplets.
        We create triplets using the labels

    Args:
        df: pandas.DataFrame, the dataframe containing the set information (filenames, labels, ...)
        encode_function : function(), function which encode labels
        transform : function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes : bool, (Default value = False), whether or not to return indexes when use __getitem__
        number: int, (Default value = None) number of triplets to be computed

    Attributes:
        dir_correspondance: dict, store already computed pair of anchor-positive samples
        number: int, number of triplets to be computed
        counter: int, count the number of triplets


    """

    def __init__(self, df, encode_function=None, transform=None, return_indexes=False,
                 number=None, use_neg_label=True, ind_name_ext=""):
        super(DataLoadDfTripletLabeled, self).__init__(df, encode_function, transform,
                                                       return_indexes)
        self.dir_correspondance = os.path.join("stored_data", "correspondance_label", ind_name_ext)
        if not os.path.exists(self.dir_correspondance):
            os.makedirs(self.dir_correspondance)
        else:
            self.clear_corr()
        self.use_neg_label = use_neg_label
        if number is None:
            self.number = len(self.df)
        else:
            self.number = number

        self.counter = 0

    def __len__(self):
        return self.number

    def clear_corr(self):
        for fname in glob.glob(os.path.join(self.dir_correspondance, "*")):
            os.remove(fname)
        self.counter = 0

    def reset_correspondance_index(self, index):
        """ Reset the stored pairs between anchors and positive/negative"""
        path_index = os.path.join(self.dir_correspondance, str(index) + ".json")
        if os.path.exists(path_index):
            os.remove(path_index)

    def get_pos_neg(self, label, index):
        """ Get the positive and negative of a segment using labels.
        Do not associate two times the same files as much as possible.
        Args:
            label: str, the label of the anchor
            index: int, the index of the anchor
        Returns
            tuple
            (positive features, negative features)
        """
        label_ = list(label)
        name_json = os.path.join(self.dir_correspondance, str(index) + ".json")
        if os.path.exists(name_json):
            try:
                with open(name_json, "r") as f_json:
                    dic_correspondance = json.loads(f_json.read())
            except json.decoder.JSONDecodeError:
                dic_correspondance = {"pos": [index],
                                      "neg": [index]}
        else:
            dic_correspondance = {"pos": [index],
                                  "neg": [index]}

        drop_pos_index = dic_correspondance["pos"]
        drop_neg_index = dic_correspondance["neg"]

        if label_ is []:
            if "event_labels" in self.df.columns:
                column = "event_labels"
            elif "event_label" in self.df.columns:
                column = "event_label"
            else:
                raise NotImplementedError("Columns available are only 'event_label' or 'event_labels' in df, "
                                          "yours are {}".format(self.df.columns))
            pos_ind_dataset = self.df[pd.isna(self.df[column])].index
            neg_ind_dataset = self.df[~pd.isna(self.df[column])].index

        # means weak, the other test could be to check if "event_labels" is in self.df
        elif type(label_) is list:
            if "event_labels" in self.df.columns:
                pos_ind_dataset = self.df[self.df["event_labels"].str.contains('|'.join(label_)).fillna(False)].index
                neg_ind_dataset = self.df[~self.df["event_labels"].str.contains('|'.join(label_)).fillna(False)].index
            elif "event_label" in self.df.columns:
                pos_ind_dataset = self.df[self.df["event_label"].isin(label_["event_label"]).fillna(False)].index
                neg_ind_dataset = self.df[~self.df["event_label"].isin(label_["event_label"]).fillna(False)].index
            else:
                raise NotImplementedError("Columns available are only 'event_label' or 'event_labels' in df, "
                                          "yours are {}".format(self.df.columns))
        else:
            raise NotImplementedError("Reference label for triplet should be a list")

        pos_ind_stay = list(set(pos_ind_dataset) - set(drop_pos_index))
        neg_ind_stay = list(set(neg_ind_dataset) - set(drop_neg_index))

        # If no label available
        if len(pos_ind_stay) == 0 or len(neg_ind_stay) == 0:
            self.reset_correspondance_index(index)
            print("iterations stop after: {}".format(self.counter))
            dic_correspondance = {"pos": [index],
                                  "neg": [index]}
            drop_pos_index = dic_correspondance["pos"]
            drop_neg_index = dic_correspondance["neg"]
            pos_ind_stay = list(set(pos_ind_dataset) - set(drop_pos_index))
            neg_ind_stay = list(set(neg_ind_dataset) - set(drop_neg_index))

            if len(pos_ind_stay) == 0:
                if len(pos_ind_dataset) == 1:
                    # When subpart_data can happen
                    pos_ind_stay = [index]
                else:
                    pos_ind_stay = list(set(pos_ind_dataset) - set(drop_pos_index))
                warnings.warn("The positive label is not a proper one, an already chosen one has been taken")
            if len(neg_ind_stay) == 0:
                if len(neg_ind_dataset) == 1:
                    # When subpart_data can happen
                    neg_ind_stay = [index]
                else:
                    neg_ind_stay = list(set(neg_ind_dataset) - set(drop_neg_index))

        pos_ind = random.choice(pos_ind_stay)

        dic_correspondance["pos"].append(pos_ind)
        x_pos = self.get_feature_file_func(self.filenames.iloc[pos_ind])
        if self.use_neg_label:
            neg_ind = random.choice(neg_ind_stay)
            x_neg = self.get_feature_file_func(self.filenames.iloc[neg_ind])
            dic_correspondance["neg"].append(neg_ind)
        else:
            x_neg = np.zeros(x_pos.shape) + np.nan
        with FileLock(name_json):
            with open(name_json, "w") as f:
                f.write(json.dumps(dic_correspondance))
                # json.dump(dic_correspondance, f)
        return x_pos, x_neg

    def get_triplet(self, label, index):
        """ Get triplets associated with an anchor file
        Args:
            label: str, the label to use
            index: int, the index of the file

        Returns
            tuple
            triplet (anchor, positive, negative) features.
        """
        x = self.get_feature_file_func(self.filenames.iloc[index])
        x_pos, x_neg = self.get_pos_neg(label, index=index)

        return x, x_pos, x_neg

    def get_sample(self, index):
        """ Get samples assiociated to a file

        Args:
            index: int, the index of the anchor file

        Returns:
            tuple
            Anchor features, positive features, negative features, encoded label
        """
        self.counter += 1
        # Useful when number > len(self.df)
        if index > self.number:
            print("Stopping the generator")
            print(index)
            print(self.counter)
            raise IndexError()

        index_ = index % len(self.df)

        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            label = self.get_label(index_)

            x, x_pos, x_neg = self.get_triplet(label, index_)

        else:
            raise NotImplementedError(
                "Dataframe to be encoded doesn't have specified columns: columns allowed:"
                "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                "for strong labels, yours: {}".format(self.df.columns))

        if self.encode_function is not None:
            y = self.encode_function(label)
        else:
            raise NotImplementedError("We did not implement without an encode function yet")
        sample = x, x_pos, x_neg, y
        return sample


class DataLoadDfTripletLabeledExhaustif(DataLoadDf):
    """ Class derived from DataLoadDfTripletAbstract
        Prepare the data to be use in a batch mode with triplets.
        We create triplets using the labels

    Args:
        df: pandas.DataFrame, the dataframe containing the set information (filenames, labels, ...)
        encode_function : function(), function which encode labels
        transform : function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes : bool, (Default value = False), whether or not to return indexes when use __getitem__
        number: int, (Default value = None) number of triplets to be computed

    Attributes:
        dir_correspondance: dict, store already computed pair of anchor-positive samples
        number: int, number of triplets to be computed
        counter: int, count the number of triplets


    """

    def __init__(self, df, encode_function=None, transform=None, return_indexes=False,
                 number=None):
        super(DataLoadDfTripletLabeledExhaustif, self).__init__(df, encode_function, transform,
                                                                return_indexes)
        self.dir_correspondance = "stored_data/correspondance_label"
        if not os.path.exists(self.dir_correspondance):
            os.makedirs(self.dir_correspondance)
        classes = unique_classes(df)
        print("all classes in df exhaustif: {}".format(classes))
        if number is None:
            self.number = 0
            for label in classes:
                indexes = df[df.event_labels == label].index
                self.number += int(len(indexes) * (len(indexes) - 1) / 2 * (len(df) - len(indexes)))
            print("number of triplets exhausted: {}".format(self.number))
            triplets = []
            for label in classes:
                indexes = df[df.event_labels == label].index
                for a in range(len(indexes) - 1):
                    for p in range(a+1, len(indexes)):
                        for n in df.drop(indexes).index:
                            triplets.append([indexes[a], indexes[p], n])
            self.triplets = triplets

        else:
            self.number = number
            triplets = []
            for label in classes:
                ii = 0
                triplets_classes = False
                indexes = df[df.event_labels == label].index
                negative_indexes = df.drop(indexes).index
                while triplets_classes is False:
                    a = random.choice(indexes)
                    p = random.choice(indexes.drop(a))
                    n = random.choice(negative_indexes)
                    triplets.append([a, p, n])
                    ii += 1
                    # Making sure we have enough triplet if unbalanced classes and wanting a lot of triplets
                    if ii > self.number // (len(classes) // 2):
                        triplets_classes = True

            self.triplets = random.sample(triplets, self.number)

        self.counter = 0

    def __len__(self):
        return self.number

    def reset_correspondance_index(self, index):
        """ Reset the stored pairs between anchors and positive/negative"""
        os.remove(os.path.join(self.dir_correspondance, str(index) + ".json"))

    def get_triplet(self, index):
        """ Get triplets associated with an anchor file
        Args:
            index: int, the index of the file

        Returns
            tuple
            triplet (anchor, positive, negative) features.
        """
        x = self.get_feature_file_func(self.filenames.iloc[self.triplets[index][0]])
        x_pos = self.get_feature_file_func(self.filenames.iloc[self.triplets[index][1]])
        x_neg = self.get_feature_file_func(self.filenames.iloc[self.triplets[index][2]])

        return x, x_pos, x_neg

    def get_sample(self, index):
        """ Get samples assiociated to a file

        Args:
            index: int, the index of the anchor file

        Returns:
            tuple
            Anchor features, positive features, negative features, encoded label
        """
        self.counter += 1
        # Useful when number > len(self.df)
        if index > self.number:
            print("Stopping the generator")
            print(index)
            print(self.counter)
            raise IndexError()

        # Anchor label
        index_ = self.triplets[index][0]

        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            label = self.get_label(index_)

            x, x_pos, x_neg = self.get_triplet(index)

        else:
            raise NotImplementedError(
                "Dataframe to be encoded doesn't have specified columns: columns allowed:"
                "'filename', 'event_labels' for weak labels; 'filename' 'onset' 'offset' 'event_label' "
                "for strong labels, yours: {}".format(self.df.columns))

        if self.encode_function is not None:
            y = self.encode_function(label)
        else:
            raise NotImplementedError("We did not implement without an encode function yet")
        sample = x, x_pos, x_neg, y
        return sample


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            length = len(e)
            r.append(length + s)
            s += length
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        count = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(count, size))
            count += size
        return cluster_ind

    def __init__(self, datasets, batch_sizes=None):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        if batch_sizes is not None:
            assert len(batch_sizes) == len(datasets), "If batch_sizes given, should be equal to the number " \
                                                      "of datasets "
        self.batch_sizes = batch_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class Subset:
    """
    Subset of a dataset to be used when separating in multiple subsets

    Args:
        dataload_df: DataLoadDf or similar, dataset to be split
    indices: sequence, list of indices to keep in this subset
    """
    def __init__(self, dataload_df, indices):
        self.indices = indices
        self.df = dataload_df.df.loc[indices].reset_index(inplace=False, drop=True)
        self.dataload_df = dataload_df.modify_object(df=self.df)

    def __getitem__(self, idx):
        return self.dataload_df[idx]
