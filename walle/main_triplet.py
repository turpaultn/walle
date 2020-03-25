#!/projects/pul51/shared/calcul/users/nturpault/anaconda3/envs/pytorch/bin/python
import argparse
import os
import os.path as osp
from copy import deepcopy
from pprint import pformat

import scipy
import warnings
import time

import torch

from common import get_model, get_optimizer, shared_args, datasets_classif, do_classif, get_dfs, measure_embeddings
from evaluation_measures import measure_classif

from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random

import config as cfg
from config import get_dirs
from DesedSynthetic import DesedSynthetic
from Embedding import calculate_embedding
from DataLoad import ConcatDataset, DataLoadDf, DataLoadDfTripletLabeled, \
    DataLoadDfTripletLabeledExhaustif
from utils.Samplers import MultiStreamBatchSampler, CategoriesSampler
from utils.Transforms import ApplyLog, Unsqueeze, ToTensor, View, Normalize, Compose
from utils.Logger import LOG
from utils.Scaler import ScalerSum
from utils.ramps import sigmoid_rampup
from utils.utils import ManyHotEncoder, create_folder, load_model, save_model, \
    to_cuda_if_available, to_cpu, EarlyStopping, SaveBest
from Embedding import get_embeddings_numpy
if torch.cuda.is_available():
    pass
else:
    pass

torch.manual_seed(0)
random.seed(0)


def get_model_name(parameters):
    model_name = ""
    if parameters.get("early_stopping") is not None:
        model_name += "es_"
    if parameters.get("conv_dropout") is not None:
        drop = parameters.get("conv_dropout")
        if drop > 0:
            model_name += "cdrop_" + str(drop) + "_"
    if parameters.get("triplet_margin") is not None:
        model_name += "margin_" + str(parameters.get("triplet_margin")) + "_"
    if parameters.get("subpart_data") is not None:
        model_name += "data_" + str(parameters.get("subpart_data"))
    if parameters.get("frames") is not None:
        model_name += "frames_" + str(parameters.get("frames"))
    if parameters.get("use_frames"):
        model_name += "nofile"
    model_name += parameters.get("weak_file")
    if parameters.get("pit"):
        model_name += "pit"
    if parameters.get("norm_embed"):
        model_name += "normemb"
    if parameters.get("agg_time"):
        model_name += parameters.get("agg_time")
    model_name += "pos_" + parameters.get("type_positive") + "_"
    model_name += "neg" + parameters.get("type_negative") + "_"

    return model_name


def compute_semi_hard_indexes(embedding, embedding_pos, embedding_neg):
    # assume it is numpy arrays in 2 dimensions
    pos_distance = np.sqrt(np.sum((embedding - embedding_pos) ** 2, axis=-1))
    neg_distances = scipy.spatial.distance.cdist(embedding_neg, embedding)
    diff = neg_distances - pos_distance
    diff[diff <= 0] = np.inf

    neg_indexes = np.argmin(diff, axis=0)
    return neg_indexes


def loop_batches_acc_grad(indexes, dataset, model_triplet, semi_hard_input=None, semi_hard_embed=None, i=0):
    out = []
    out_pos = []
    out_neg = []
    # zero the parameter gradients

    for j, ind in enumerate(indexes):
        samples = dataset[ind]

        inputs, inputs_pos, inputs_neg, pred_labels = samples
        inputs, inputs_pos = to_cuda_if_available(inputs, inputs_pos)
        if i < 2:
            LOG.debug("input shape: {}".format(inputs.shape))
        if semi_hard_input is not None or semi_hard_embed is not None:
            assert semi_hard_input is not None, "semi_hard_input and semi_hard_embed should be defined"
            assert semi_hard_embed is not None, "semi_hard_input and semi_hard_embed should be defined"
            model_triplet.eval()

            embed = get_embeddings_numpy(inputs, model_triplet)
            embed_pos = get_embeddings_numpy(inputs_pos, model_triplet)

            label_mask = (pred_labels.numpy() == -1).all(-1)
            semi_hard_mask = np.isnan(inputs_neg.detach().numpy()).reshape(inputs_neg.shape[0], -1).all(-1)
            mask = label_mask & semi_hard_mask

            if i < 2:
                LOG.debug("mask: {}".format(mask))
            negative_indexes = compute_semi_hard_indexes(embed[mask], embed_pos[mask], semi_hard_embed)
            inputs_neg[np.where(mask)] = semi_hard_input[negative_indexes]
        inputs_neg = to_cuda_if_available(inputs_neg)

        model_triplet.eval()
        with torch.no_grad():
            outputs_pos = model_triplet(inputs_pos)
            outputs_neg = model_triplet(inputs_neg)

        model_triplet.train()
        # forward + backward + optimize
        outputs = model_triplet(inputs)

        out.append(outputs)
        out_pos.append(outputs_pos)
        out_neg.append(outputs_neg)

    outputs = torch.stack(out, 0)
    outputs_pos = torch.stack(out_pos, 0)
    outputs_neg = torch.stack(out_neg, 0)
    return outputs, outputs_pos, outputs_neg


def loop_batches(samples, model_triplet, semi_hard_input=None, semi_hard_embed=None, i=0):
    inputs, inputs_pos, inputs_neg, pred_labels = samples
    inputs, inputs_pos = to_cuda_if_available(inputs, inputs_pos)
    if i < 2:
        LOG.debug("input shape: {}".format(inputs.shape))
    if semi_hard_input is not None or semi_hard_embed is not None:
        assert semi_hard_input is not None, "semi_hard_input and semi_hard_embed should be defined"
        assert semi_hard_embed is not None, "semi_hard_input and semi_hard_embed should be defined"
        model_triplet.eval()

        embed = get_embeddings_numpy(inputs, model_triplet)
        embed_pos = get_embeddings_numpy(inputs_pos, model_triplet)

        label_mask = (pred_labels.numpy() == -1).all(-1)
        semi_hard_mask = np.isnan(inputs_neg.detach().numpy()).reshape(inputs_neg.shape[0], -1).all(-1)
        mask = label_mask & semi_hard_mask

        if i < 2:
            LOG.debug("mask: {}".format(mask))
        negative_indexes = compute_semi_hard_indexes(embed[mask], embed_pos[mask], semi_hard_embed)
        inputs_neg[np.where(mask)] = semi_hard_input[negative_indexes]
    inputs_neg = to_cuda_if_available(inputs_neg)

    model_triplet.eval()
    with torch.no_grad():
        outputs_pos = model_triplet(inputs_pos)
        outputs_neg = model_triplet(inputs_neg)

    model_triplet.train()
    # forward + backward + optimize
    outputs = model_triplet(inputs)

    return outputs, outputs_pos, outputs_neg


def train_triplet_epoch(loader, model_triplet, optimizer, semi_hard_input=None,
                        semi_hard_embed=None, pit=False, margin=None, swap=False, acc_grad=False):

    start = time.time()
    loss_mean_triplet = []
    nb_triplets_used = 0
    nb_triplets = 0
    if acc_grad:
        lder = loader.batch_sampler
    else:
        lder = loader
    # for i, samples in enumerate(concat_loader_triplet):
    for i, samples in enumerate(lder):
        optimizer.zero_grad()
        if acc_grad:
            outs = loop_batches_acc_grad(samples, loader.dataset, model_triplet,
                                         semi_hard_input, semi_hard_embed, i=i)
        else:
            outs = loop_batches(samples, model_triplet, semi_hard_input, semi_hard_embed, i=i)

        outputs, outputs_pos, outputs_neg = outs
        if i == 0:
            LOG.debug("output CNN shape: {}".format(outputs.shape))
            LOG.debug(outputs.mean())
            LOG.debug(outputs_pos.mean())
            LOG.debug(outputs_neg.mean())
        dist_pos, dist_neg = get_distances(outputs, outputs_pos, outputs_neg,
                                           pit, swap,
                                           )

        if margin is not None:
            loss_triplet = torch.clamp(margin + dist_pos - dist_neg, min=0.0)
        else:
            loss_triplet = ratio_loss(dist_pos, dist_neg)

        pair_cnt = (loss_triplet.detach() > 0).sum().item()
        nb_triplets_used += pair_cnt
        nb_triplets += len(loss_triplet)

        # Normalize based on the number of pairs.
        if pair_cnt > 0:
            # loss_triplet = loss_triplet.sum() / pair_cnt
            loss_triplet = loss_triplet.mean()

            loss_triplet.backward()
            optimizer.step()
            loss_mean_triplet.append(loss_triplet.item())
        else:
            LOG.debug("batch doesn't have any loss > 0")

    epoch_time = time.time() - start
    LOG.info("Loss: {:.4f}\t"
             "Time: {}\t"
             "\tnb_triplets used: {} / {}\t"
             "".format(np.mean(loss_mean_triplet), epoch_time, nb_triplets_used, nb_triplets))
    ratio_triplet_used = nb_triplets_used / nb_triplets
    return model_triplet, loss_mean_triplet, ratio_triplet_used


def ratio_loss(d_pos, d_neg):
    sumexp = torch.exp(d_pos) + torch.exp(d_neg)
    loss_triplet = (torch.exp(d_pos) / sumexp) ** 2 + (1 - torch.exp(d_neg) / sumexp) ** 2
    return loss_triplet


def pairwise_distance(a, b, p=2, dim=-1):
    return (((a - b) ** p).sum(dim) + 1e-8).pow(1 / 2)


def get_distances(outputs, outputs_pos, outputs_neg, pit, swap=False):
    if not pit:
        outputs = outputs.view(outputs.shape[0], -1)
        outputs_pos = outputs_pos.view(outputs.shape[0], -1)
        outputs_neg = outputs_neg.view(outputs.shape[0], -1)

        dist_pos = pairwise_distance(outputs, outputs_pos)
        dist_neg = pairwise_distance(outputs, outputs_neg)
        if swap:
            dist_neg_s = pairwise_distance(outputs_pos, outputs_neg)
            dist_neg = torch.min(dist_neg, dist_neg_s)
    else:
        def dist(anch, other, p=2, **kwargs):
            # if not mean_frames: # Cannot do mean frames and pit ...
            if len(anch.shape) == 3:
                bs = anch.shape[0]
                anch = anch.view(bs, -1)
                other = other.view(bs, -1)
            else:
                anch = anch.view(-1)
                other = other.view(-1)

            dist = pairwise_distance(anch, other, p, **kwargs)
            return dist

        # Compute the permutation invariant loss
        with torch.no_grad():
            # Init, distance no roll
            dist_pos = dist(outputs, outputs_pos)
            dist_neg = dist(outputs, outputs_neg)
            final_embed_pos = outputs_pos.clone()
            final_embed_neg = outputs_neg.clone()
            for fr in range(1, outputs.shape[1]):
                # Rolled values
                rolled_pos = torch.roll(outputs_pos, fr, dims=1)
                rolled_neg = torch.roll(outputs_neg, fr, dims=1)
                cur_dpos = dist(outputs, rolled_pos)
                cur_dneg = dist(outputs, rolled_neg)
                # Compare distances, and update the values of the real positive (otherwise backward problem)
                mask_pos = cur_dpos < dist_pos
                final_embed_pos[mask_pos] = rolled_pos[mask_pos]
                mask_neg = cur_dneg < dist_neg
                final_embed_neg[mask_neg] = rolled_neg[mask_neg]
        dist_pos = dist(outputs, final_embed_pos)
        dist_neg = dist(outputs, final_embed_neg)

    return dist_pos, dist_neg


def validate_training_representation(triplet_model, validation_triplets_loader, margin=None, pit=False):
    triplet_model.eval()
    validation_loss = []
    # for counter, triplet_ in enumerate(validation_triplets_loader):
    for i, indexes in enumerate(validation_triplets_loader.batch_sampler):
        for j, ind in enumerate(indexes):
            triplet_ = validation_triplets_loader.dataset[ind]
            inputs_, inputs_pos_, inputs_neg_, pred_labels_ = triplet_

            inputs_, inputs_pos_, inputs_neg_ = to_cuda_if_available(inputs_, inputs_pos_, inputs_neg_)

            out = triplet_model(inputs_)
            pos = triplet_model(inputs_pos_)
            neg = triplet_model(inputs_neg_)

            with torch.no_grad():
                dist_pos, dist_neg = get_distances(out, pos, neg,
                                                   pit,
                                                   )

                if margin is not None:
                    triplet_loss = torch.clamp(margin + dist_pos - dist_neg, min=0.0).mean()
                else:
                    triplet_loss = ratio_loss(dist_pos, dist_neg).mean()

            triplet_loss = to_cpu(triplet_loss)
            validation_loss.append(triplet_loss.item())
    validation_loss = np.mean(validation_loss)
    triplet_model.train()
    return validation_loss


if __name__ == '__main__':
    LOG.info(__file__)
    # ###########
    # ## Argument
    # ###########
    t = time.time()
    print("Arguments have been set for a certain group of experiments, feel free to change it.")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--margin', type=float, default=None, dest="margin")
    parser.add_argument('--type_positive', type=str, default="nearest", dest="type_positive")
    parser.add_argument('--type_negative', type=str, default="semi_hard", dest="type_negative")
    # Experiences to compare the impact of number of labaled vs unlabeled triplets
    # Be careful if subpart data is not None!!!!!!
    parser.add_argument('--nb_labeled_triplets', type=int, default=None, dest="nb_labeled_triplets")
    parser.add_argument('--nb_unlabeled_triplets', type=int, default=None, dest="nb_unlabeled_triplets")
    parser.add_argument('--pit', action="store_true", default=False)
    parser.add_argument('--swap', action="store_true", default=False)
    parser.add_argument('--resume_training', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=cfg.n_epoch_embedding)

    parser = shared_args(parser)

    f_args = parser.parse_args()
    LOG.info(pformat(vars(f_args)))

    resume_training = f_args.resume_training
    if not resume_training:
        state = f_args.__dict__
    else:
        warnings.warn("When resume_training is indicated, no other parameter is taken into account")
        model_triplet, optimizer, state = load_model(resume_training, return_optimizer=True, return_state=True)
        f_args = argparse.Namespace(**state)
    ############
    #  Parameters experiences
    ###########
    type_positive = f_args.type_positive  # "label" # "all_augmented", "label", "jansen", "nearest"
    type_negative = f_args.type_negative  # "semi_hard" # "label"

    if type_positive == 'all_augmented':
        type_positive = "all"

    triplet_margin = f_args.margin
    subpart_data = f_args.subpart_data
    n_layers_RNN = cfg.n_layers_RNN
    nb_labeled_triplets = f_args.nb_labeled_triplets

    nb_unlabeled_triplets = f_args.nb_unlabeled_triplets

    test_path = cfg.test2018
    eval_path = cfg.eval2018
    val_list = None
    if f_args.weak_file == "1event":
        with open(os.path.join(cfg.relative_data_path, cfg.one_event_valid_list), "r") as f:
            val_list = f.read().split(",")
        weak_path = cfg.one_event_train
        test_path = cfg.one_event
    elif f_args.weak_file == "1event0.2":
        with open(os.path.join(cfg.relative_data_path, cfg.one_event_valid_list), "r") as f:
            val_list = f.read().split(",")
        weak_path = cfg.one_event_train200
        test_path = cfg.one_event200
    elif f_args.weak_file == "weak":
        weak_path = cfg.weak
    else:
        weak_path = cfg.weak
        warnings.warn("Wrong argument for weak_file, taking weak data")

    norm_embed = f_args.norm_embed
    pit = f_args.pit
    swap = f_args.swap
    agg_time = f_args.agg_time
    frames_in_sec = f_args.frames_in_sec
    segment = f_args.segment
    # #####################
    # End of arguments attribution
    # ####################
    collate_fn = torch.utils.data.dataloader.default_collate
    # Mainly for classif
    max_len_sec = cfg.max_len_seconds
    max_frames = cfg.max_frames
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    # #########
    # DATA
    # ########
    dataset = DesedSynthetic(cfg.relative_data_path,
                             base_feature_dir=cfg.base_feature_dir,
                             save_log_feature=False)

    dfs = get_dfs(dataset, weak_path, test_path, eval_path, subpart_data, valid_list=val_list,
                  frames_in_sec=frames_in_sec,
                  segment=segment,
                  dropna=f_args.dropna, unique_fr=f_args.unique_fr,
                  fixed_segment=f_args.fixed_segment
                  )

    if resume_training is None:
        classes = dataset.classes
        many_hot_encoder = ManyHotEncoder(classes)
    else:
        many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
        classes = many_hot_encoder.labels
    encode_function_label = many_hot_encoder.encode_weak

    # Datasets
    trans_fr = [ApplyLog(), ToTensor(), Unsqueeze(0)]

    train_weak_df_fr = dfs["train"]
    train_weak_dl_fr = DataLoadDf(train_weak_df_fr, encode_function_label, transform=Compose(trans_fr))

    if type_positive != "label" or type_negative != "label":
        unlabel_df_fr = dataset.get_df_feat_dir(cfg.unlabel, subpart_data=subpart_data, frames_in_sec=frames_in_sec)
        unlabel_dl_fr = DataLoadDf(unlabel_df_fr, encode_function_label, transform=Compose(trans_fr))
        datasets_mean = [train_weak_dl_fr, unlabel_dl_fr]
    else:
        datasets_mean = [train_weak_dl_fr]
    # Normalize
    if resume_training is None:
        scaler = ScalerSum()
        scaler.calculate_scaler(ConcatDataset(datasets_mean))
    else:
        scaler = ScalerSum.load_state_dict(state["scaler"])
    LOG.debug(scaler.mean_)

    trans_fr_scale = trans_fr + [Normalize(scaler)]
    if segment:
        trans_fr_scale.append(Unsqueeze(0))

    for dl in datasets_mean:
        dl.set_transform(Compose(trans_fr_scale))
    print(dl.transform)
    concat_frames = ConcatDataset(datasets_mean)

    trans_fr_sc_embed = deepcopy(trans_fr_scale)
    if not segment:
        trans_fr_sc_embed.append(Unsqueeze(0))

    train_weak_embed = DataLoadDf(train_weak_df_fr, encode_function_label,
                                  transform=Compose(trans_fr_sc_embed))
    valid_weak_df_fr = dfs["valid"]
    valid_weak_dl_fr = DataLoadDf(valid_weak_df_fr, encode_function_label,
                                  transform=Compose(trans_fr_sc_embed))

    # ##############
    # Triplet dataset
    # #############
    type_positive = "label"
    number = nb_labeled_triplets
    print("number label positive: " + str(number))
    # TODO, when use_negative_label is false, give another dataset to search the negative sample,
    # or implement only positive and do negative semi hard mining outside
    if type_negative == "label":
        use_negative_label = True
        semi_hard_dataset = False
        unlabel = False
    else:
        use_negative_label = False
        unlabel = True

    weak_train_dl_triplet = DataLoadDfTripletLabeled(train_weak_df_fr,
                                                     encode_function_label,
                                                     transform=Compose(trans_fr_scale),
                                                     number=number,
                                                     use_neg_label=use_negative_label,
                                                     ind_name_ext="weak_fr")

    concat_dataset_triplet = weak_train_dl_triplet

    print("len concat dataset: " + str(concat_dataset_triplet.__len__()))

    # Remove classe with less than 2 examples, useful just when subpart data is on
    for c in classes:
        cl_df = valid_weak_df_fr[valid_weak_df_fr.event_labels.fillna("").str.contains(c)]
        if len(cl_df) < 2:
            valid_weak_df_fr = valid_weak_df_fr.drop(cl_df.index)
    valid_weak_df_fr = valid_weak_df_fr.reset_index()
    weak_valid_dl_triplet = DataLoadDfTripletLabeledExhaustif(valid_weak_df_fr,
                                                              encode_function_label,
                                                              transform=Compose(trans_fr_sc_embed),
                                                              number=cfg.number_test)

    test_df_fr = dfs["test"]
    test_dl_fr = DataLoadDf(test_df_fr, encode_function_label, transform=Compose(trans_fr_sc_embed))
    min_length = np.inf
    for c in classes:
        len_pos_label = len(
            test_df_fr[test_df_fr["event_labels"].str.contains('|'.join(c)).fillna(False)].index)
        if len_pos_label < min_length:
            min_length = len_pos_label
    print("min classes examples: " + str(min_length))
    number = min_length * 10
    print('number valid: ' + str(number))

    test_triplets = DataLoadDfTripletLabeledExhaustif(test_df_fr,
                                                      encode_function_label,
                                                      transform=Compose(trans_fr_sc_embed),
                                                      number=min(cfg.number_test, len(test_df_fr.dropna())))
    # #########
    # End of DATA
    # ########

    if resume_training is None:
        state.update({
            'scaler': scaler.state_dict(),
            'many_hot_encoder': many_hot_encoder.state_dict()
        })

    model_directory, log_directory = get_dirs("pretrained" + "_bs_" + str(batch_size) + "adam")

    if frames_in_sec is not None:
        fr = frames_in_sec
    elif segment:
        fr = "seg"
    else:
        fr = "unknown"
    params_name = {
        "early_stopping": cfg.early_stopping,
        "conv_dropout": cfg.conv_dropout,
        "frames": fr,
    }
    params_name.update(f_args.__dict__)

    base_model_name = get_model_name(params_name)

    # ##############
    # Model
    # ############
    shuffle = True
    if unlabel:
        # Assume weak is first, and unlabel second
        # nb_labeled triplets already taken into account
        len_weak = len(concat_dataset_triplet.datasets[0])
        if nb_unlabeled_triplets is not None:
            len_unlab = nb_unlabeled_triplets
        else:
            len_unlab = len(concat_dataset_triplet.datasets[1])

        batch_sizes = [
            round(batch_size * len_weak / (len_weak + len_unlab)),
            round(batch_size * len_unlab / (len_weak + len_unlab))
        ]
        LOG.info("batch_sizes: {}".format(batch_sizes))
        sampler = MultiStreamBatchSampler(concat_dataset_triplet, batch_sizes)
        triplet_loader = DataLoader(concat_dataset_triplet, batch_sampler=sampler,
                                    num_workers=cfg.num_workers,
                                    collate_fn=collate_fn)
    else:
        min_class = np.inf
        for cl in classes:
            n_cl = train_weak_df_fr.event_labels.str.contains(cl).sum()
            if n_cl < min_class and n_cl != 0:
                min_class = n_cl
        min_class = max(min_class, round(batch_size/len(classes)))  # At least one batch to test when subpart_data
        n_per_class = max(round(batch_size / len(classes)), 1)
        sampler = CategoriesSampler(train_weak_df_fr.event_labels, classes,
                                    n_per_class=n_per_class)
        LOG.info("sampler same nb data per class, len sampler: {}".format(len(sampler)))
        triplet_loader = DataLoader(concat_dataset_triplet,
                                    batch_sampler=sampler,
                                    # shuffle=shuffle, batch_size=batch_size, drop_last=False,
                                    num_workers=cfg.num_workers,
                                    # collate_fn=collate_fn
                                    )

    valid_triplets_loader = DataLoader(weak_valid_dl_triplet, batch_size=batch_size, shuffle=False,
                                       num_workers=cfg.num_workers,
                                       drop_last=True, collate_fn=collate_fn)

    test_triplets_loader = DataLoader(test_triplets, batch_size=batch_size, shuffle=False,
                                      num_workers=cfg.num_workers,
                                      drop_last=True, collate_fn=collate_fn)

    # #########
    # # Model and optimizer
    # ########
    if resume_training is None:
        model_triplet, state = get_model(state, f_args)
        optimizer, state = get_optimizer(model_triplet, state)

    LOG.info(model_triplet)
    pytorch_total_params = sum(p.numel() for p in model_triplet.parameters() if p.requires_grad)
    LOG.info("number of parameters in the model: {}".format(pytorch_total_params))
    model_triplet.train()
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    LOG.info(optimizer)
    model_triplet = to_cuda_if_available(model_triplet)

    # ##########
    # # Callbacks
    # ##########
    if cfg.save_best:
        save_best_call = SaveBest(val_comp="sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup")

    # ##########
    # # Training
    # ##########
    save_results = pd.DataFrame()

    model_name_triplet = base_model_name + "triplet"

    if cfg.save_best:
        model_path_pretrain = os.path.join(model_directory, model_name_triplet, "best_model")
    else:
        model_path_pretrain = os.path.join(model_directory, model_name_triplet, "epoch_" + str(f_args.epochs))
    print("path of model : " + model_path_pretrain)
    create_folder(os.path.join(model_directory, model_name_triplet))

    batch_size_classif = cfg.batch_size_classif
    # Hard coded because no semi_hard in this version
    semi_hard_embed = None
    semi_hard_input = None
    if not os.path.exists(model_path_pretrain) or cfg.recompute_embedding:
        margin = triplet_margin
        for epoch in range(f_args.epochs):
            t_start_epoch = time.time()

            if cfg.rampup_margin_length is not None:
                margin = sigmoid_rampup(epoch, cfg.rampup_margin_length) * triplet_margin
            model_triplet.train()
            model_triplet, loss_mean_triplet, ratio_used = train_triplet_epoch(triplet_loader,
                                                                               # triplet_loader,
                                                                               model_triplet, optimizer,
                                                                               semi_hard_input, semi_hard_embed,
                                                                               pit=pit,
                                                                               margin=margin, swap=swap,
                                                                               acc_grad=segment
                                                                               )
            model_triplet.eval()
            loss_mean_triplet = np.mean(loss_mean_triplet)

            embed_dir = "stored_data/embeddings"
            embed_dir = os.path.join(embed_dir, model_name_triplet, "embeddings")
            create_folder(embed_dir)
            fig_dir = os.path.join(embed_dir, "figures")
            create_folder(fig_dir)

            # Validate
            val_loss = validate_training_representation(model_triplet, valid_triplets_loader, margin,
                                                        # norm_embed=norm_embed,
                                                        pit=pit,
                                                        )
            LOG.info(f"######### ---->  Validation triplet loss : {val_loss}")
            test_loss = validate_training_representation(model_triplet, test_triplets_loader, margin,
                                                         # norm_embed=norm_embed,
                                                         pit=pit,
                                                         )
            LOG.info(f"######### ---->  Test triplet loss : {test_loss}")

            t_embed = time.time()
            measures_emb_train = dict()
            measures_emb_valid = dict()
            measures_emb_test = dict()
            wait_embed = 5
            # epoch_ext = str(frames_in_sec) + str(epoch)
            epoch_ext = f"{fr}_{epoch}"
            if train_weak_embed is not None and epoch % wait_embed == 1:
                name = "train" + epoch_ext
                measures_emb_train = measure_embeddings(train_weak_embed, model_triplet,
                                                        os.path.join(embed_dir, name),
                                                        os.path.join(fig_dir, name),
                                                        "train",
                                                        )

            if valid_weak_dl_fr is not None and epoch % wait_embed == 1:
                name = "valid" + epoch_ext
                measures_emb_valid = measure_embeddings(valid_weak_dl_fr, model_triplet,
                                                        os.path.join(embed_dir, name),
                                                        os.path.join(fig_dir, name),
                                                        "valid",
                                                        )

            if test_dl_fr is not None and epoch % wait_embed == 1:
                name = "test" + epoch_ext
                measures_emb_test = measure_embeddings(test_dl_fr, model_triplet,
                                                       os.path.join(embed_dir, name),
                                                       os.path.join(fig_dir, name),
                                                       "test",
                                                       )
            LOG.info(f"time embed: {time.time() - t_embed}")
            # print statistics
            print('[%d / %d, %5d] Triplet loss: %.3f' %
                  (epoch + 1, f_args.epochs,
                   # len(triplet_loader) + 1,
                   len(sampler) + 1,
                   loss_mean_triplet))

            ###########
            # Callbacks
            ###########
            results = {"triplet_loss": loss_mean_triplet,
                       "val_loss": val_loss,
                       "test_loss": test_loss,
                       }
            results.update(measures_emb_train)
            results.update(measures_emb_valid)
            results.update(measures_emb_test)
            print_results = "\n"
            for k in results:
                print_results += f"\t {k}: {results[k]} \n"

            LOG.info(print_results)
            save_results = save_results.append(results, ignore_index=True)
            save_results.to_csv(os.path.join(log_directory, model_name_triplet + ".csv"),
                                sep="\t", header=True, index=False)

            state["epoch"] = epoch + 1
            state["model"]["state_dict"] = model_triplet.state_dict()
            state["optimizer"]["state_dict"] = optimizer.state_dict()
            state.update(results)

            # CALLBACKS
            mval = measures_emb_valid.get("protovalid")
            if mval is None:
                mval = 0
            if cfg.early_stopping is not None:
                if early_stopping_call.apply(mval):
                    print("EARLY STOPPING")
                    break

            if cfg.model_checkpoint is not None:
                if epoch % cfg.model_checkpoint == cfg.model_checkpoint - 1:
                    model_path_chkpt = os.path.join(model_directory, model_name_triplet, "epoch_" + str(epoch))
                    save_model(state, model_path_chkpt)

            if cfg.save_best:
                if save_best_call.apply(mval):
                    save_model(state, model_path_pretrain)

            print(f"Time epoch complete: {time.time() - t_start_epoch}")
        save_results.to_csv(os.path.join(log_directory, model_name_triplet + ".csv"), sep="\t", header=True,
                            index=False)

        if cfg.save_best:
            print(f"best model at epoch : {save_best_call.best_epoch} with validation loss {save_best_call.best_val}")
            print(model_path_pretrain)
            model_triplet = load_model(model_path_pretrain)
        else:
            model_path_pretrain = os.path.join(model_directory,
                                               model_name_triplet, "epoch_" + str(f_args.epochs))
            save_model(state, model_path_pretrain)
            print(model_path_pretrain)
        print('Finished Training')
    else:
        model_triplet = load_model(model_path_pretrain)

    if eval_path:
        eval_df_fr = dfs.get("eval")
        eval_dl = DataLoadDf(eval_df_fr, encode_function_label,
                             transform=Compose(trans_fr_sc_embed))
    else:
        eval_dl = None
    dataloaders = datasets_classif(model_triplet, train_weak_embed, valid_weak_dl_fr, test_dl_fr, f_args,
                                   many_hot_encoder, classes, save_name=base_model_name, eval_dl=eval_dl)
    classif_model, classif_state = do_classif(dataloaders, many_hot_encoder, classes,
                                              save_model_dir=osp.join(model_directory, base_model_name + "classif"),
                                              result_path=osp.join(log_directory, base_model_name + "classif" + ".csv"))

    model_triplet = to_cuda_if_available(model_triplet)
    model_triplet.eval()

    if agg_time is not None:
        trans_embedding = [ToTensor(), View(-1)]
    else:
        trans_embedding = [ToTensor()]
    test_df1 = dfs["test1"]
    test_dl1 = DataLoadDf(test_df1, encode_function_label, transform=Compose(trans_fr_sc_embed))
    embed_set1 = "final_test1"
    test_embed_dir1 = os.path.join(embed_dir, embed_set1)
    df_test_embed1, _ = calculate_embedding(test_dl1, model_triplet, savedir=test_embed_dir1, concatenate="append")
    test_embed1 = DataLoadDf(df_test_embed1, encode_function_label, transform=Compose(trans_embedding))
    test_embed_loader1 = DataLoader(test_embed1, batch_size=batch_size_classif, shuffle=False,
                                    num_workers=num_workers,
                                    drop_last=False)

    test_df10 = dfs["test10"]
    test_dl10 = DataLoadDf(test_df10, encode_function_label, transform=Compose(trans_fr_sc_embed))
    embed_set10 = "final_test10"
    test_embed_dir10 = os.path.join(embed_dir, embed_set10)
    df_test_embed10, _ = calculate_embedding(test_dl10, model_triplet, savedir=test_embed_dir10,
                                             concatenate="append")
    test_embed10 = DataLoadDf(df_test_embed10, encode_function_label, transform=Compose(trans_embedding))
    test_embed_loader10 = DataLoader(test_embed10, batch_size=batch_size_classif, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

    model_triplet = to_cpu(model_triplet)
    classif_model = to_cuda_if_available(classif_model)
    classif_model.eval()
    mean_test_results1 = measure_classif(classif_model, test_embed_loader1,
                                         classes=classes,
                                         suffix_print="test1")

    mean_test_results10 = measure_classif(classif_model, test_embed_loader10,
                                          classes=classes,
                                          suffix_print="test10")

    print(f"Time of the program: {time.time() - t}")
    from orion.client import report_results

    report_results(
        [dict(
            name="mean_test_results",
            type="objective",
            value=float(100 - classif_state["macro_measure_valid"] * 100)
        )
        ]
    )
