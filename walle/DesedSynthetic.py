# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, Justin Salamon, Ankit Parag Shah, 2019, v1.0
# This software is distributed under the terms of the License MIT
#########################################################################

from __future__ import print_function

import numpy as np
import os
import librosa
import time
import pandas as pd

import soundfile

import config as cfg
from utils.Logger import LOG
from utils.utils import create_folder, read_audio, ManyHotEncoder, pad_trunc_seq, unique_classes, name_only


class DesedSynthetic:
    """ DESED_synthetic, takes the audio files from a folder defined, compute features and give the annotations in the
    right format. It expects strong labels (since it is synthetic data).

    Args:
        local_path: str, (Default value = "") base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, (Default value = "features) base directory to store the features
        recompute_features: bool, (Default value = False) wether or not to recompute features
        save_log_feature: bool, (Default value = True) whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)

    Attributes:
        local_path: str, base directory where the dataset is, to be changed if
            dataset moved
        base_feature_dir: str, base directory to store the features
        recompute_features: bool, wether or not to recompute features
        save_log_feature: bool, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)
        feature_dir : str, directory to store the features

    """
    def __init__(self, local_path="", base_feature_dir="features", recompute_features=False,
                 save_log_feature=True):

        self.local_path = local_path
        self.recompute_features = recompute_features
        self.save_log_feature = save_log_feature

        self.base_feature_dir = base_feature_dir

        feature_dir = os.path.join(self.base_feature_dir, "sr" + str(cfg.sample_rate) + "_win" + str(cfg.n_window)
                                   + "_hop" + str(cfg.hop_length) + "_mels" + str(cfg.n_mels))
        if not self.save_log_feature:
            feature_dir += "_nolog"

        self.feature_dir = os.path.join(feature_dir, "features")
        self.metadata_dir = os.path.join(feature_dir, "metadata")
        # create folder if not exist
        create_folder(self.metadata_dir)
        create_folder(self.feature_dir)
        self.classes = []

    def get_df_feat_dir(self, csv_path, subpart_data=None, frames_in_sec=None, segment=False, fixed_segment=None):
        """ Initialize the dataset, extract the features dataframes
        Args:
            csv_path: str, csv path in the initial dataset
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.
            frames_in_sec: int, allow to divide full segments into smaller segments of this number of frames.
            segment: bool, whether or not to segment event when having strong labels.
            fixed_segment: float, in seconds, the size of the kept segment. If >audio length, the audio length is kept.
                If segment is True, and >label, it takes the surrounding (allow creating weak labels).
        Returns:
            pd.DataFrame
            The dataframe containing the right features and labels
        """
        feature_dir = os.path.join(self.feature_dir, name_only(csv_path))
        create_folder(feature_dir)
        meta_name = os.path.join(self.local_path, csv_path)

        assert (not segment or frames_in_sec is None), "if you want to segment, you can't give frames"
        if segment:
            df = self.extract_features_from_meta_segment(meta_name, feature_dir, subpart_data=subpart_data,
                                                         fixed_segment=fixed_segment)
        elif frames_in_sec is not None:
            df = self.extract_features_from_meta_frames(meta_name, feature_dir, frames_in_sec,
                                                        subpart_data=subpart_data)
            # get_classes is done inside the method because of get_labels
        else:
            df = self.extract_features_from_meta(meta_name, feature_dir, subpart_data=subpart_data)
            self.get_classes(df)

        return df

    def get_classes(self, df):
        """ Get the different classes of the dataset
        Returns:
            A list containing the classes
        """
        new_classes = self.classes.copy()
        unique_labels = unique_classes(df)
        new_classes.extend(unique_labels)
        new_classes = list(set(new_classes))
        new_classes.sort()
        self.classes = new_classes

    @staticmethod
    def get_subpart_data(df, subpart_data):
        column = "filename"
        if not subpart_data > len(df[column].unique()):
            filenames = df[column].drop_duplicates().sample(subpart_data, random_state=10)
            df = df[df[column].isin(filenames)].reset_index(drop=True)
            LOG.debug("Taking subpart of the data, len : {}, df_len: {}".format(subpart_data, len(df)))
        return df

    @staticmethod
    def get_df_from_meta(meta_name, subpart_data=None):
        """
        Extract a pandas dataframe from a csv file

        Args:
            meta_name : str, path of the csv file to extract the df
            subpart_data: int, the number of file to take in the dataframe if taking a small part of the dataset.

        Returns:
            dataframe
        """
        df = pd.read_csv(meta_name, header=0, sep="\t")
        if subpart_data is not None:
            df = DesedSynthetic.get_subpart_data(df, subpart_data)
        return df

    @staticmethod
    def get_audio_dir_path_from_meta(filepath):
        """ Get the corresponding audio dir from a meta filepath

        Args:
            filepath : str, path of the meta filename (csv)

        Returns:
            str
            path of the audio directory.
        """
        base_filepath = os.path.splitext(filepath)[0]
        audio_dir = base_filepath.replace("metadata", "audio")
        if audio_dir.split('/')[-2] in ['validation']:
            if not os.path.exists(audio_dir):
                audio_dir = '/'.join(audio_dir.split('/')[:-1])
        audio_dir = os.path.abspath(audio_dir)
        return audio_dir

    @staticmethod
    def periodic_hann(window_length):
        """From Google code (VGG Audioset), Calculate a "periodic" Hann window.

        The classic Hann window is defined as a raised cosine that starts and
        ends on zero, and where every value appears twice, except the middle
        point for an odd-length window.  Matlab calls this a "symmetric" window
        and np.hanning() returns it.  However, for Fourier analysis, this
        actually represents just over one cycle of a period N-1 cosine, and
        thus is not compactly expressed on a length-N Fourier basis.  Instead,
        it's better to use a raised cosine that ends just before the final
        zero value - i.e. a complete cycle of a period-N cosine.  Matlab
        calls this a "periodic" window. This routine calculates it.

        Args:
          window_length: The number of points in the returned window.

        Returns:
          A 1D np.array containing the periodic hann window.
        """
        return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                                   np.arange(window_length)))

    @staticmethod
    def calculate_mel_spec(audio, sample_rate=cfg.sample_rate, n_window=cfg.n_window, hop_length=cfg.hop_length,
                           n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, log_feature=True):
        """
        Calculate a mal spectrogram from raw audio waveform
        Note: The parameters of the spectrograms are in the config.py file.
        Args:
            audio : numpy.array, raw waveform to compute the spectrogram
            sample_rate: int, sample rate of teh audio file
            n_window: int, number of points by window in the stft computation
            hop_length: int, number of points between each beginning of stft window (hop_length<n_window means overlap)
            n_mels: int, number of frequency bands
            f_min: int, minimal frequency for the mel spectrogram
            f_max: int, maxmimal frequency for the mel spectrogram
            log_feature: bool, whether to apply log or not to the spectrogram
        Returns:
            numpy.array
            containing the mel spectrogram
        """
        # Compute spectrogram
        han_win = DesedSynthetic.periodic_hann(n_window)

        spec = librosa.stft(
            audio,
            n_fft=2 ** int(np.ceil(np.log(n_window) / np.log(2.0))),
            hop_length=hop_length,
            win_length=n_window,
            window=han_win,
            center=True,
            pad_mode="constant"
        )

        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=sample_rate,
            n_mels=n_mels,
            fmin=f_min, fmax=f_max,
            htk=True, norm=None)

        if log_feature:
            mel_spec = np.log(mel_spec + 0.01)
            # mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def extract_features_from_meta(self, csv_audio, feature_dir, subpart_data=None):
        """Extract log mel spectrogram features.

        Args:
            csv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
                the associated wav_filename is Yname_start_end.wav
            feature_dir: str, the path to the directory where the features are
            subpart_data: int, number of files to extract features from the csv.
        """
        t1 = time.time()
        df_meta = self.get_df_from_meta(csv_audio, subpart_data)
        LOG.info("{} Total file number: {}".format(csv_audio, len(df_meta.filename.unique())))

        for ind, wav_name in enumerate(df_meta.filename.unique()):
            if ind % 500 == 0:
                LOG.debug(ind)
            wav_dir = self.get_audio_dir_path_from_meta(csv_audio)
            wav_path = os.path.join(wav_dir, wav_name)

            out_filename = os.path.join(feature_dir, name_only(wav_name) + ".npy")

            if not os.path.exists(out_filename):
                if not os.path.isfile(wav_path):
                    LOG.error("File %s is in the csv file but the feature is not extracted!" % wav_path)
                    df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
                else:
                    (audio, _) = read_audio(wav_path, cfg.sample_rate)
                    if audio.shape[0] == 0:
                        print("File %s is corrupted!" % wav_path)
                    else:
                        mel_spec = self.calculate_mel_spec(audio, log_feature=self.save_log_feature)

                        np.save(out_filename, mel_spec)

                    LOG.debug("compute features time: %s" % (time.time() - t1))

        return df_meta.reset_index(drop=True)

    def get_features(self, wav_path, feature_dir, frames):
        (audio, _) = read_audio(wav_path, cfg.sample_rate)
        mel_spec = self.calculate_mel_spec(audio, log_feature=self.save_log_feature)

        # Trunc the data so it is a multiple of frames. Just change the nb of frames
        # if you want padding instead
        if frames > mel_spec.shape[0]:
            pad_trunc_length = frames
        else:
            pad_trunc_length = mel_spec.shape[0] - mel_spec.shape[0] % frames
        mel_spec = pad_trunc_seq(mel_spec, pad_trunc_length)

        # Reshape in multiple segments and save them
        mel_spec_frames = mel_spec.reshape(-1, frames, mel_spec.shape[-1])
        out_filenames = []
        wav_name = os.path.basename(wav_path)
        for cnt, sample in enumerate(mel_spec_frames):
            out_filename = os.path.join(feature_dir, name_only(wav_name)) + "fr" + str(frames) + "_" + \
                           str(cnt * frames) + "-" + str((cnt + 1) * frames) + ".npy"
            np.save(out_filename, sample)
            out_filenames.append(out_filename)
        cnt_max = len(mel_spec_frames)
        return audio, cnt_max

    def get_labels(self, ind, df_meta, wav_name, frames, out_filenames):
        cnt_max = len(out_filenames)
        if {"onset", "offset", "event_label"}.issubset(df_meta.columns):
            many_hot_encoder = ManyHotEncoder(self.classes, n_frames=cnt_max * frames)
            df_wav_name = df_meta[df_meta.filename == wav_name].copy()
            # Because values are in seconds in the file
            df_wav_name["onset"] = df_wav_name["onset"] * cfg.sample_rate // cfg.hop_length
            df_wav_name["offset"] = df_wav_name["offset"] * cfg.sample_rate // cfg.hop_length
            y = many_hot_encoder.encode_strong_df(df_wav_name)

            encoded_labels = y.reshape(-1, frames, y.shape[-1])
            weak_labels_frames = encoded_labels.max(axis=1)
            weak_labels_frames = [','.join(many_hot_encoder.decode_weak(weak_labels)) for
                                  weak_labels in
                                  weak_labels_frames]
            add_item = {
                "raw_filename": [wav_name for _ in range(len(out_filenames))],
                "filename": out_filenames,
                "event_labels": weak_labels_frames
            }

        elif "event_labels" in df_meta.columns:
            weak_labels_frames = [df_meta.iloc[ind]["event_labels"] for _ in
                                  range(len(out_filenames))]
            add_item = {
                "raw_filename": [wav_name for _ in range(len(out_filenames))],
                "filename": out_filenames,
                "event_labels": weak_labels_frames
            }
        else:
            add_item = {"raw_filename": [wav_name for _ in range(len(out_filenames))],
                        "filename": out_filenames}
        return add_item

    def extract_features_from_meta_frames(self, csv_audio, feature_dir, frames_in_sec, subpart_data=None):
        """Extract log mel spectrogram features.

        Args:
            csv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
                the associated wav_filename is Yname_start_end.wav
            feature_dir: str, the directory where the features are or will be created
            subpart_data: int, number of files to extract features from the csv.
            frames_in_sec: int, number of frames to take for a subsegment.
        """
        frames = int(frames_in_sec * cfg.sample_rate / cfg.hop_length)
        t1 = time.time()
        df_meta = pd.read_csv(csv_audio, header=0, sep="\t")
        LOG.info("{} Total file number: {}".format(csv_audio, len(df_meta.filename.unique())))

        # Csv to store the features
        ext_name = "_" + str(frames)
        if subpart_data is not None and subpart_data < len(df_meta.filename.unique()):
            ext_name += "_sub" + str(subpart_data)
            df_meta = self.get_subpart_data(df_meta, subpart_data)

        self.get_classes(df_meta)

        meta_base, meta_ext = os.path.splitext(csv_audio.split("/")[-1])
        csv_features = os.path.join(self.metadata_dir, meta_base + ext_name + meta_ext)

        wav_dir = self.get_audio_dir_path_from_meta(csv_audio)
        df_features = pd.DataFrame()

        path_exists = os.path.exists(csv_features)

        if not path_exists:
            LOG.debug("Creating new feature df")

            # Loop in all the filenames
            cnt_new_features = 0
            for ind, wav_name in enumerate(df_meta.filename.unique()):
                wav_path = os.path.join(wav_dir, wav_name)
                if not os.path.isfile(wav_path):
                    LOG.error("File %s is in the csv file but the feature is not extracted, deleting...!" % wav_path)
                    df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
                else:
                    try:
                        audio_len_sec = soundfile.info(wav_path).duration
                    except Exception as e:
                        print("File %s is corrupted, not added to df!" % wav_path)
                        print(e)
                        continue
                    if audio_len_sec == 0:
                        print("File %s is corrupted, not added to df!" % wav_path)
                    else:
                        files_exist = True
                        # How many features we can compute from this file ?
                        cnt_max = min(int(audio_len_sec // frames_in_sec), int(cfg.max_len_seconds // frames_in_sec))
                        if cnt_max == 0:
                            cnt_max = 1

                        base_wav_name = os.path.join(feature_dir, name_only(wav_name))
                        # Check if files already exist
                        out_filenames = [
                            base_wav_name + "fr" + str(frames) + "_" +
                            str(cnt * frames) + "-" + str((cnt + 1) * frames) + ".npy"
                            for cnt in range(cnt_max)
                        ]
                        for fname in out_filenames:
                            if not os.path.exists(fname):
                                files_exist = False
                                break

                        if not files_exist:
                            if cnt_new_features % 500 == 0:
                                LOG.debug(f"new features, {cnt_new_features}")
                            cnt_new_features += 1
                            audio, cnt_max = self.get_features(wav_path, feature_dir, frames)
                            out_filenames = [
                                base_wav_name + "fr" + str(frames) + "_" +
                                str(cnt * frames) + "-" + str((cnt + 1) * frames) + ".npy"
                                for cnt in range(cnt_max)
                            ]

                        # features label to add to the dataframe
                        add_item = self.get_labels(ind, df_meta, wav_name, frames, out_filenames)

                        df_features = df_features.append(pd.DataFrame(add_item), ignore_index=True)

            LOG.info(csv_features)
            df_features.to_csv(csv_features, sep="\t", header=True, index=False)
            df_features = pd.read_csv(csv_features, sep="\t")  # Otherwise event_labels is "" and not NaN
        else:
            df_features = self.get_df_from_meta(csv_features)  # No subpart data because should be in the name

        LOG.debug("compute features time: %s" % (time.time() - t1))
        return df_features

    @staticmethod
    def trunc_pad_segment(df, fixed_segment):
        def apply_ps_func(row, length):
            duration = (row["offset"] - row["onset"])
            # Choose fixed segment in the event
            if duration > length:
                ra = np.random.uniform(-1, 1)
                onset_bias = fixed_segment * ra
                row["onset"] = max(0, row["onset"] + onset_bias)
            # Bias the onset and the offset accordingly
            else:
                ra = np.random.rand()
                onset_bias = fixed_segment * ra
                row["onset"] = max(0, row["onset"] - onset_bias)

            row["offset"] = row["onset"] + fixed_segment
            if row["offset"] > cfg.max_len_seconds:
                row["offset"] = cfg.max_len_seconds
                row["onset"] = row["offset"] - fixed_segment
            return row
        assert "onset" in df.columns and "offset" in df.columns, "bias label only available with strong labels"
        LOG.info(f"Fix labels {fixed_segment} seconds")
        df = df.apply(apply_ps_func, axis=1, args=[fixed_segment])
        return df

    def extract_features_from_meta_segment(self, csv_audio, feature_dir, subpart_data=None, fixed_segment=None):
        """Extract log mel spectrogram features, but the csv needs to be strongly labeled.

        Args:
            csv_audio : str, file containing names, durations and labels : (name, start, end, label, label_index)
                the associated wav_filename is Yname_start_end.wav
            feature_dir: str, the path of the features directory.
            subpart_data: int, number of files to extract features from the csv.
            fixed_segment: float, in seconds, the size of the kept segment. If >audio length, the audio length is kept.
                If segment is True, and >label, it takes the surrounding (allow creating weak labels).
        """
        t1 = time.time()
        df_meta = self.get_df_from_meta(csv_audio, subpart_data)
        self.get_classes(df_meta)
        LOG.info("{} Total file number: {}".format(csv_audio, len(df_meta.filename.unique())))

        ext_name = "_segment_"
        if subpart_data:
            ext_name += str(subpart_data)

        if fixed_segment is not None:
            LOG.debug(f" durations before: "
                      f"{df_meta.groupby('event_label').apply(lambda x: (x.offset - x.onset).mean())}")
            ext_name += f"fix{fixed_segment}"
            df_meta = self.trunc_pad_segment(df_meta, fixed_segment)
            LOG.debug(f" durations after: "
                      f"{df_meta.groupby('event_label').apply(lambda x: (x.offset - x.onset).mean())}")

        meta_base, meta_ext = os.path.splitext(csv_audio.split("/")[-1])
        csv_features = os.path.join(self.metadata_dir, meta_base + ext_name + meta_ext)

        wav_dir = self.get_audio_dir_path_from_meta(csv_audio)
        df_features = pd.DataFrame()

        path_exists = os.path.exists(csv_features)

        if not path_exists:
            # Loop in all the filenames
            for ind, wav_name in enumerate(df_meta.filename.unique()):
                if ind % 500 == 0:
                    LOG.debug(ind)

                wav_path = os.path.join(wav_dir, wav_name)
                if not os.path.isfile(wav_path):
                    LOG.error("File %s is in the csv file but the feature is not extracted, deleting...!" % wav_path)
                    df_meta = df_meta.drop(df_meta[df_meta.filename == wav_name].index)
                else:
                    try:
                        audio_len_sec = soundfile.info(wav_path).duration
                    except Exception as e:
                        print("File %s is corrupted, not added to df!" % wav_path)
                        print(e)
                        continue
                    if audio_len_sec == 0:
                        print("File %s is corrupted, not added to df!" % wav_path)
                    else:
                        files_exist = True
                        # How many features we can compute from this file ?
                        sub_df = df_meta[df_meta.filename == wav_name]
                        cnt_max = len(sub_df)

                        if cnt_max == 0:
                            break

                        base_wav_name = name_only(wav_name)
                        ext_featname = "_seg"
                        if fixed_segment:
                            ext_featname += f"fix{fixed_segment}"
                            files_exist = False  # We should always recompute because of the randomness of onset offset
                        # Check if files already exist
                        out_filenames = [
                            base_wav_name + ext_featname + str(cnt) + ".npy"
                            for cnt in range(cnt_max)
                        ]
                        for fname in out_filenames:
                            fpath = os.path.join(feature_dir, fname)
                            if not os.path.exists(fpath):
                                files_exist = False
                                break

                        add_item = {
                            "raw_filename": [],
                            "filename": [],
                            "event_labels": []
                        }
                        for ii, (i, row) in enumerate(sub_df.iterrows()):
                            if not pd.isna(row.event_label):
                                if ii > 0:
                                    extnb = str(ii)
                                else:
                                    extnb = ""
                                out_filename = os.path.join(feature_dir, name_only(wav_name))
                                out_filename += ext_featname + extnb + ".npy"
                                if not files_exist:
                                    sr = soundfile.info(wav_path).samplerate
                                    (audio, _) = read_audio(wav_path, cfg.sample_rate,
                                                            start=int(row.onset * sr), stop=int(row.offset * sr))
                                    mel_spec = self.calculate_mel_spec(audio, log_feature=self.save_log_feature)
                                    if fixed_segment:
                                        pad_trunc_length = int(fixed_segment * cfg.sample_rate // cfg.hop_length)
                                        mel_spec = pad_trunc_seq(mel_spec, pad_trunc_length)
                                    np.save(out_filename, mel_spec)

                                add_item["raw_filename"].append(wav_name)
                                add_item["filename"].append(out_filename)
                                add_item["event_labels"].append(row["event_label"])

                        df_features = df_features.append(pd.DataFrame(add_item), ignore_index=True)

            df_features.to_csv(csv_features, sep="\t", header=True, index=False)
            df_features = pd.read_csv(csv_features, sep="\t")  # Otherwise event_labels is "" and not NaN
        else:
            df_features = self.get_df_from_meta(csv_features)  # No subpart data because should be in the name

        LOG.debug("compute features time: %s" % (time.time() - t1))
        return df_features


if __name__ == '__main__':
    import config as cfg
    dataset = DesedSynthetic("../dcase2019",
                             base_feature_dir="../dcase2019/features",
                             save_log_feature=False)

    weak_df1 = dataset.get_df_feat_dir(cfg.weak, subpart_data=25, frames_in_sec=9.60)

    weak_df = dataset.get_df_feat_dir(cfg.weak, subpart_data=25, frames_in_sec=9.60)

    weak_df2 = dataset.get_df_feat_dir(cfg.weak, subpart_data=25, frames_in_sec=9.6)

    weak_df3 = dataset.get_df_feat_dir(cfg.weak, subpart_data=25, frames_in_sec=.96)

    weak_df4 = dataset.get_df_feat_dir(cfg.test2018, subpart_data=25, frames_in_sec=.96)

    weak_df5 = dataset.get_df_feat_dir(cfg.test2018, subpart_data=25, frames_in_sec=1.)
