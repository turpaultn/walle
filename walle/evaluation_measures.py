# -*- coding: utf-8 -*-
from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy as np
import pandas as pd
import torch

import config as cfg
from utils.Logger import LOG
from utils.utils import ManyHotEncoder, to_cuda_if_available


def get_f_measure_by_class(torch_model, nb_tags, dataloader_, thresholds_=None, max=False):
    """ get f measure for each class given a model and a generator of data (batch_x, y)

    Args:
        torch_model : Model, model to get predictions, forward should return weak and strong predictions
        nb_tags : int, number of classes which are represented
        dataloader_ : generator, data generator used to get f_measure
        thresholds_ : int or list, thresholds to apply to each class to binarize probabilities
        max: bool, whether or not to take the max of the predictions

    Returns:
        macro_f_measure : list, f measure for each class

    """
    torch_model = to_cuda_if_available(torch_model)

    # Calculate external metrics
    tp = np.zeros(nb_tags)
    tn = np.zeros(nb_tags)
    fp = np.zeros(nb_tags)
    fn = np.zeros(nb_tags)
    for counter, (batch_x, y) in enumerate(dataloader_):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()

        pred_weak = torch_model(batch_x)
        pred_weak = pred_weak.cpu().data.numpy()
        labels = y.numpy()

        # Used only with a model predicting only strong outputs
        if len(pred_weak.shape) == 3:
            # Max because indicate the presence, give weak labels
            pred_weak = np.max(pred_weak, axis=1)

        if len(labels.shape) == 3:
            labels = np.max(labels, axis=1)
            labels = ProbabilityEncoder().binarization(labels,
                                                       binarization_type="global_threshold",
                                                       threshold=0.5)
        if counter == 0:
            LOG.info(f"shapes, input: {batch_x.shape}, output: {pred_weak.shape}, label: {labels.shape}")

        if not max:
            if thresholds_ is None:
                binarization_type = 'global_threshold'
                thresh = 0.5
            else:
                binarization_type = "class_threshold"
                assert type(thresholds_) is list
                thresh = thresholds_

            batch_predictions = ProbabilityEncoder().binarization(pred_weak,
                                                                  binarization_type=binarization_type,
                                                                  threshold=thresh,
                                                                  time_axis=0
                                                                  )
        else:
            batch_predictions = np.zeros(pred_weak.shape)
            batch_predictions[:, pred_weak.argmax(1)] = 1

        tp_, fp_, fn_, tn_ = intermediate_at_measures(labels, batch_predictions)
        tp += tp_
        fp += fp_
        fn += fn_
        tn += tn_

    print("Macro measures: TP: {}\tFP: {}\tFN: {}\tTN: {}".format(tp, fp, fn, tn))

    macro_f_score = np.zeros(nb_tags)
    mask_f_score = 2 * tp + fp + fn != 0
    macro_f_score[mask_f_score] = 2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]

    return macro_f_score


def intermediate_at_measures(encoded_ref, encoded_est):
    """ Calculate true/false - positives/negatives.

    Args:
        encoded_ref: np.array, the reference array where a 1 means the label is present, 0 otherwise
        encoded_est: np.array, the estimated array, where a 1 means the label is present, 0 otherwise

    Returns:
        tuple
        number of (true positives, false positives, false negatives, true negatives)

    """
    tp = (encoded_est + encoded_ref == 2).sum(axis=0)
    fp = (encoded_est - encoded_ref == 1).sum(axis=0)
    fn = (encoded_ref - encoded_est == 1).sum(axis=0)
    tn = (encoded_est + encoded_ref == 0).sum(axis=0)
    return tp, fp, fn, tn


def measure_classif(classif_model, test_loader, classes, suffix_print="Test", single_label=False):
    classif_model.eval()
    macro_f_score = get_f_measure_by_class(classif_model, len(classes), test_loader, max=single_label)
    # print("##macro_measure: " + str(macro_f_measure))
    # print(np.mean(macro_f_measure))
    results_serie_test = pd.DataFrame(macro_f_score, index=classes)[0]

    print("Audio Tagging resutlts on {} set:".format(suffix_print))
    print(results_serie_test * 100)
    print("AT mean macro f measure {}: {:.2f}".format(suffix_print, results_serie_test.mean() * 100))

    return results_serie_test.mean() * 100


def get_event_list_current_file(df, fname):
    """ Get list of events for a given filename
    Args:
        df: pd.DataFrame, the dataframe to search on
        fname: the filename to extract the value from the dataframe
    Returns:
         list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')

    return event_list_for_current_file


def event_based_evaluation_df(reference, estimated, t_collar=0.200, percentage_of_length=0.2):
    """ Calculate EventBasedMetric given a reference and estimated dataframe
    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score'
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.):
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return segment_based_metric


def macro_f_measure(tp, fp, fn):
    """ From intermediates measures, give the macro F-measure

    Args:
        tp: int, number of true positives
        fp: int, number of false positives
        fn: int, number of true negatives

    Returns:
        float
        The macro F-measure
    """
    macro_f_score = np.zeros(tp.shape[-1])
    mask_f_score = 2 * tp + fp + fn != 0
    macro_f_score[mask_f_score] = 2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]
    return macro_f_score


def get_weak_predictions(model, valid_dataset, weak_decoder, save_predictions=None):
    for i, (data, _) in enumerate(valid_dataset):
        data = to_cuda_if_available(data)

        pred_weak = model(data.unsqueeze(0))
        pred_weak = pred_weak.cpu()
        pred_weak = pred_weak.squeeze(0).detach().numpy()
        if i == 0:
            LOG.debug(pred_weak)
        pred_weak = ProbabilityEncoder().binarization(pred_weak, binarization_type="global_threshold",
                                                      threshold=0.5)
        pred = weak_decoder(pred_weak)
        pred = pd.DataFrame(pred, columns=["event_labels"])
        pred["filename"] = valid_dataset.filenames.iloc[i]
        if i == 0:
            LOG.debug("predictions: \n{}".format(pred))
            prediction_df = pred.copy()
        else:
            prediction_df = prediction_df.append(pred)

    if save_predictions is not None:
        LOG.info("Saving predictions at: {}".format(save_predictions))
        prediction_df.to_csv(save_predictions, index=False, sep="\t")
    return prediction_df


def compute_strong_metrics(predictions, valid_df, pooling_time_ratio):
    # In seconds
    predictions.onset = predictions.onset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)
    predictions.offset = predictions.offset * pooling_time_ratio / (cfg.sample_rate / cfg.hop_length)

    metric_event = event_based_evaluation_df(valid_df, predictions, t_collar=0.200,
                                             percentage_of_length=0.2)
    metric_segment = segment_based_evaluation_df(valid_df, predictions, time_resolution=1.)
    LOG.info(metric_event)
    LOG.info(metric_segment)
    return metric_event


def format_df(df, mhe):
    """ Make a weak labels dataframe from strongly labeled (join labels)
        Args:
            df: pd.DataFrame, the dataframe strongly labeled with onset and offset columns (+ event_label)
            mhe: ManyHotEncoder object, the many hot encoder object that can encode the weak labels

        Returns:
            weakly labeled dataframe
    """
    def join_labels(x):
        return pd.Series(dict(filename=x['filename'].iloc[0],
                              event_label=mhe.encode_weak(x["event_label"].drop_duplicates().dropna().tolist())))

    if "onset" in df.columns or "offset" in df.columns:
        df = df.groupby("filename", as_index=False).apply(join_labels)
    return df


def audio_tagging_results(reference, estimated):
    classes = []
    if "event_label" in reference.columns:
        classes.extend(reference.event_label.dropna().unique())
        classes.extend(estimated.event_label.dropna().unique())
        classes = list(set(classes))
        mhe = ManyHotEncoder(classes)
        reference = format_df(reference, mhe)
        estimated = format_df(estimated, mhe)
    else:
        classes.extend(reference.event_labels.str.split(',', expand=True).unstack().dropna().unique())
        classes.extend(estimated.event_labels.str.split(',', expand=True).unstack().dropna().unique())
        classes = list(set(classes))
        mhe = ManyHotEncoder(classes)

    matching = reference.merge(estimated, how='outer', on="filename", suffixes=["_ref", "_pred"])

    def na_values(val):
        if type(val) is np.ndarray:
            return val
        if pd.isna(val):
            return np.zeros(len(classes))
        return val

    if not estimated.empty:
        matching.event_label_pred = matching.event_label_pred.apply(na_values)
        matching.event_label_ref = matching.event_label_ref.apply(na_values)

        tp, fp, fn, tn = intermediate_at_measures(np.array(matching.event_label_ref.tolist()),
                                                  np.array(matching.event_label_pred.tolist()))
        macro_res = macro_f_measure(tp, fp, fn)
    else:
        macro_res = np.zeros(len(classes))

    results_serie = pd.DataFrame(macro_res, index=mhe.labels)
    return results_serie[0]
