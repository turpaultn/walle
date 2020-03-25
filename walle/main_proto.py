import argparse
import os
import os.path as osp
from copy import deepcopy
from pprint import pformat

import time
import warnings

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import pandas as pd

from DataLoad import DataLoadDf
from utils.Transforms import ApplyLog, Unsqueeze, ToTensor, View, Normalize, Compose
from Embedding import calculate_embedding
from utils.Samplers import CategoriesSampler
import config as cfg
from DesedSynthetic import DesedSynthetic
from evaluation_measures import measure_classif
from common import get_model, get_optimizer, shared_args, datasets_classif, do_classif, get_dfs, measure_embeddings
from utils.Logger import LOG
from utils.Scaler import ScalerSum
from utils.utils import ManyHotEncoder, create_folder, to_cuda_if_available, EarlyStopping, SaveBest, to_cpu, \
    load_model, save_model, count_acc, euclidean_metric


def get_model_name(parameters):
    model_name = ""
    if parameters.get("shot") is not None:
        model_name += "s_" + str(parameters.get("shot"))
    if parameters.get("query") is not None:
        model_name += "q_" + str(parameters.get("query"))
    if parameters.get("conv_dropout") is not None:
        drop = parameters.get("conv_dropout")
        if drop > 0:
            model_name += "cdrop_" + str(drop) + "_"
    if parameters.get("train-way") is not None:
        model_name += "trw_" + str(parameters.get("train-way")) + "_"
    if parameters.get("test-way") is not None:
        model_name += "tw_" + str(parameters.get("test-way")) + "_"
    if parameters.get("subpart_data") is not None:
        model_name += "data_" + str(parameters.get("subpart_data"))
    if parameters.get("frames") is not None:
        model_name += "frames_" + str(parameters.get("frames"))
    model_name += parameters.get("weak_file")
    if parameters.get("pit"):
        model_name += "pit"
    if parameters.get("norm_embed"):
        model_name += "normemb"
    if parameters.get("agg_time"):
        model_name += parameters.get("agg_time")

    return model_name


def proto_batches_acc(indexes, dset, prot_model, way, shot):
    proto_l = []
    query_l = []
    for j, ind in enumerate(indexes):
        batch = dset[ind]
        # label = torch.arange(args.train_way).repeat(args.query)
        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in batch]
            # label = label.type(torch.cuda.LongTensor)
        else:
            data, _ = batch
            # label = label.type(torch.LongTensor)
        if j < shot * way:
            prot_model.eval()
            proto_l.append(prot_model(data))
        else:
            prot_model.train()
            query_l.append(prot_model(data))

    prot = torch.cat(proto_l, 0)
    quer = torch.cat(query_l, 0)
    return prot, quer


def proto_batches(batch, prot_model, way, shot):
    # label = torch.arange(args.train_way).repeat(args.query)
    if torch.cuda.is_available():
        data, _ = [_.cuda() for _ in batch]
    else:
        data, _ = batch

    p = shot * way
    data_shot, data_query = data[:p], data[p:]
    prot_model.eval()
    prot = prot_model(data_shot)
    prot_model.train()
    quer = prot_model(data_query)
    return prot, quer


def proto_epoch(sampler, dset, prot_model, f_args, train=False, segment=False):
    loss_mean = 0
    acc_mean = 0
    if train:
        way = f_args.train_way
    else:
        way = f_args.test_way

    if segment:
        lder = sampler
    else:
        lder = DataLoader(dset, batch_sampler=sampler, num_workers=cfg.num_workers)

    cnt = 0
    for samples in lder:
        if train:
            optimizer.zero_grad()
        if segment:
            prot, quer = proto_batches_acc(samples, dset, prot_model, way, f_args.shot)
        else:
            prot, quer = proto_batches(samples, prot_model, way, f_args.shot)

        label = torch.arange(way).repeat(f_args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        if cnt == 0 and train:
            print(prot.shape)
            print(quer.shape)
            print(label.shape)

        # p = f_args.shot * way
        # data_shot, data_query = data[:p], data[p:]

        # proto = model(data_shot)
        prot = prot.reshape(f_args.shot, way, -1).mean(dim=0)

        # query = model(data_query)
        if args.agg_time is not None:
            quer = quer.reshape(quer.shape[0], -1)
        logi = euclidean_metric(quer, prot)
        loss = F.cross_entropy(logi, label)

        acc = count_acc(logi, label)
        # f_meas = intermediate_at_measures(label, logits)

        loss_mean += loss.item()
        acc_mean += acc

        if train:
            loss.backward()
            optimizer.step()

        cnt += 1
    if cnt > 0:
        loss_mean = loss_mean / cnt
        acc_mean = acc_mean / cnt
    else:
        warnings.warn("No training has been performed")
    return loss_mean, acc_mean


if __name__ == '__main__':
    LOG.info(__file__)
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)  # How many to get for proto
    parser.add_argument('--query', type=int, default=1)  # How many to eval
    parser.add_argument('--train-way', type=int, default=10)
    parser.add_argument('--test-way', type=int, default=10)

    parser.add_argument('--n_layers_RNN', type=int, default=2)
    parser.add_argument('--dim_RNN', type=int, default=64)

    parser.add_argument('--test-only', action="store_true", default=False)
    parser.add_argument('--load', default='./stored_data/model/proto-1/max-acc.pth')
    parser = shared_args(parser)

    args = parser.parse_args()
    pformat(vars(args))

    test_path = cfg.test2018
    eval_path = cfg.eval2018
    val_list = None
    if args.weak_file == "weak":
        weak_path = cfg.weak
    elif args.weak_file == "1event":
        with open(os.path.join(cfg.relative_data_path, cfg.one_event_valid_list), "r") as f:
            val_list = f.read().split(",")
        weak_path = cfg.one_event_train
        test_path = cfg.one_event
        eval_path = cfg.eval2018
    elif args.weak_file == "1event0.2":
        with open(os.path.join(cfg.relative_data_path, cfg.one_event_valid_list), "r") as f:
            val_list = f.read().split(",")
        weak_path = cfg.one_event_train200
        test_path = cfg.one_event200
    else:
        weak_path = cfg.weak
        warnings.warn("Wrong argument for weak_file, taking weak data")

    model_directory, log_directory = cfg.get_dirs("proto")
    LOG.info("model_dir: {} \n log_dir: {}".format(model_directory, log_directory))
    max_len_sec = cfg.max_len_seconds
    subpart_data = args.subpart_data
    dataset = DesedSynthetic(cfg.relative_data_path,
                             base_feature_dir=cfg.base_feature_dir,
                             save_log_feature=False)
    dfs = get_dfs(dataset, weak_path, test_path, eval_path, subpart_data, valid_list=val_list,
                  frames_in_sec=args.frames_in_sec, segment=args.segment, dropna=args.dropna,
                  unique_fr=args.unique_fr, fixed_segment=args.fixed_segment)
    train_weak_df = dfs["train"]
    classes = dataset.classes
    many_hot_encoder = ManyHotEncoder(classes)

    # ##############
    # Triplet dataset
    # #############
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    list_trans_fr = [
        ApplyLog(),
        ToTensor(),
        Unsqueeze(0)
    ]
    if args.segment:
        list_trans_fr.append(Unsqueeze(0))

    train_set = DataLoadDf(train_weak_df,
                           many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)
    LOG.debug("len train : {}".format(len(train_set)))
    # train_load = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True,
    #                         drop_last=True, collate_fn=default_collate)

    # scaler = Scaler()
    scaler = ScalerSum()
    scaler.calculate_scaler(train_set)
    LOG.debug(scaler.mean_)

    list_trans_fr.append(Normalize(scaler))
    train_set.set_transform(Compose(list_trans_fr))
    # Validation data
    valid_weak_df = dfs["valid"]
    if valid_weak_df is not None:
        valid_set = DataLoadDf(valid_weak_df,
                               many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)

    list_trans_val = deepcopy(list_trans_fr)
    if not args.segment:
        list_trans_val.append(Unsqueeze(0))

    train_dl_emb = DataLoadDf(train_weak_df,
                              many_hot_encoder.encode_weak, Compose(list_trans_val), return_indexes=False)
    valid_dl_emb = DataLoadDf(valid_weak_df,
                              many_hot_encoder.encode_weak, Compose(list_trans_val), return_indexes=False)
    LOG.debug("len train: {}".format(len(train_set)))
    test_df = dfs["test"]
    test_dl_emb = DataLoadDf(test_df,
                             many_hot_encoder.encode_weak, Compose(list_trans_val), return_indexes=False)
    test_dl = DataLoadDf(test_df,
                         many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)

    train_sampler = CategoriesSampler(train_set.df.event_labels, classes,
                                      args.shot + args.query, n_classes=args.train_way)

    test_set_val = DataLoadDf(test_df,
                              many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)
    LOG.debug("len test: {}".format(len(test_set_val)))
    test_sampler = CategoriesSampler(test_set_val.df.event_labels,
                                     classes, args.shot + args.query, n_classes=args.test_way)

    valid_set_fr_val = DataLoadDf(valid_weak_df,
                                  many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)
    LOG.debug("len valid: {}".format(len(valid_set_fr_val)))
    valid_sampler = CategoriesSampler(valid_set_fr_val.df.event_labels,
                                      classes, args.shot + args.query, n_classes=args.test_way)

    params_name = {
        "early_stopping": cfg.early_stopping,
        "conv_dropout": cfg.conv_dropout,
        "frames": cfg.frames_in_sec,
    }
    params_name.update(args.__dict__)

    base_model_name = get_model_name(params_name)
    # Model
    state = {
        "scaler": scaler.state_dict(),
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "args": vars(args),
    }
    model, state = get_model(state, args)
    optimizer, state = get_optimizer(model, state)
    model = to_cuda_if_available(model)
    LOG.info(model)

    # ##########
    # # Callbacks
    # ##########
    if cfg.save_best:
        save_best_call = SaveBest(val_comp="sup")
    if cfg.early_stopping is not None:
        early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup")
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # x, y = next(iter(train_loader))
    x, y = train_set[0]
    print("x shape {}, y shape {}".format(x.shape, y.shape))
    if not args.test_only:
        trlog = {"max_acc": 0}
        save_results = pd.DataFrame()
        for epoch in range(1, args.max_epoch + 1):
            model.train()

            tl, ta = proto_epoch(train_sampler, train_set, model, args, train=True, segment=args.segment)
            print('epoch {}, train nb batches {}, loss={:.4f} acc={:.4f}, '
                  .format(epoch,
                          len(train_sampler),
                          # len(train_loader),
                          tl, ta))

            model.eval()
            vl, va = proto_epoch(valid_sampler, valid_set_fr_val, model, args, train=False, segment=args.segment)
            print('epoch {}, valid nb batches {}, loss={:.4f} acc={:.4f}, '
                  .format(epoch,
                          len(valid_sampler),
                          # len(train_loader),
                          vl, va))

            tel, tea = proto_epoch(test_sampler, test_set_val, model, args, train=False, segment=args.segment)
            print('epoch {}, test nb batches {}, loss={:.4f} acc={:.4f}, '
                  .format(epoch,
                          len(test_sampler),
                          # len(train_loader),
                          tel, tea))

            trlog['train_loss'] = tl
            trlog['train_acc'] = ta
            trlog['val_loss'] = vl
            trlog['val_acc'] = va
            trlog['test_loss'] = tel
            trlog['test_acc'] = tea

            state["epoch"] = epoch + 1
            state["model"]["state_dict"] = model.state_dict()
            state["optimizer"]["state_dict"] = optimizer.state_dict()
            state.update(trlog)

            print(f"Epoch: {epoch}/{args.max_epoch}")

            embed_dir = "stored_data/embeddings"
            embed_dir = os.path.join(embed_dir, "proto", "embeddings")
            create_folder(embed_dir)
            fig_dir = os.path.join(embed_dir, "figures")
            create_folder(fig_dir)
            name = "train" + str(epoch)
            measures_emb_train = measure_embeddings(train_dl_emb, model,
                                                    osp.join(embed_dir, name), osp.join(fig_dir, name), "train")
            name = "val" + str(epoch)
            measures_emb_valid = measure_embeddings(valid_dl_emb, model,
                                                    osp.join(embed_dir, name), osp.join(fig_dir, name), "val")
            name = "test" + str(epoch)
            measures_emb_test = measure_embeddings(test_dl_emb, model,
                                                   osp.join(embed_dir, name), osp.join(fig_dir, name), "test")

            trlog.update(measures_emb_train)
            trlog.update(measures_emb_valid)
            trlog.update(measures_emb_test)
            print_results = "\n"
            for k in trlog:
                print_results += "\t {}: {} \n".format(k, trlog[k])

            LOG.info(print_results)
            save_results = save_results.append(trlog, ignore_index=True)
            save_results.to_csv(os.path.join(log_directory, "results" + ".csv"),
                                sep="\t", header=True, index=False)

            if cfg.early_stopping is not None:
                if early_stopping_call.apply(va):
                    print("EARLY STOPPING")
                    break

            if cfg.model_checkpoint is not None:
                if epoch % cfg.model_checkpoint == cfg.model_checkpoint - 1:
                    save_model(state, osp.join(model_directory, 'epoch_{}'.format(epoch)))

            if cfg.save_best:
                if va is None:
                    va = 0
                if save_best_call.apply(va):
                    trlog["max_acc"] = epoch
                    save_model(state, osp.join(model_directory, 'max-acc'))

        save_model(state, osp.join(model_directory, 'epoch-last'))

    LOG.info("\nEVAL\n")

    if cfg.save_best:
        print(f"best model at epoch : {save_best_call.best_epoch} with validation loss {save_best_call.best_val}")
        print(osp.join(model_directory, 'max-acc'))
        model = load_model(osp.join(model_directory, 'max-acc'))

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    if eval_path:
        eval_df_fr = dfs.get("eval")
        eval_dl = DataLoadDf(eval_df_fr, many_hot_encoder.encode_weak,
                             transform=Compose(list_trans_val))
    else:
        eval_dl = None
    dataloaders = datasets_classif(model, train_dl_emb, valid_dl_emb, test_dl_emb, args, many_hot_encoder, classes,
                                   base_model_name, eval_dl=eval_dl)
    classif_model, classif_state = do_classif(dataloaders, many_hot_encoder, classes,
                                              save_model_dir=osp.join(model_directory, base_model_name + "classif"),
                                              result_path=osp.join(log_directory, base_model_name + "classif" + ".csv"))

    model = to_cuda_if_available(model)
    model.eval()
    if args.agg_time is not None:
        trans_embedding = [ToTensor(), View(-1)]
    else:
        trans_embedding = [ToTensor()]
    test_df1 = dfs["test1"]
    test_dl1 = DataLoadDf(test_df1, many_hot_encoder.encode_weak, transform=Compose(list_trans_val))
    embed_set1 = "final_test1"
    test_embed_dir1 = os.path.join(embed_dir, embed_set1)
    df_test_embed1, _ = calculate_embedding(test_dl1, model, savedir=test_embed_dir1, concatenate="append")
    test_embed1 = DataLoadDf(df_test_embed1, many_hot_encoder.encode_weak, transform=Compose(trans_embedding))
    test_embed_loader1 = DataLoader(test_embed1, batch_size=cfg.batch_size_classif, shuffle=False,
                                    num_workers=num_workers,
                                    drop_last=False)

    test_df10 = dfs["test10"]
    test_dl10 = DataLoadDf(test_df10, many_hot_encoder.encode_weak, transform=Compose(list_trans_val))
    embed_set10 = "final_test10"
    test_embed_dir10 = os.path.join(embed_dir, embed_set10)
    df_test_embed10, _ = calculate_embedding(test_dl10, model, savedir=test_embed_dir10,
                                             concatenate="append")
    test_embed10 = DataLoadDf(df_test_embed10, many_hot_encoder.encode_weak, transform=Compose(trans_embedding))
    test_embed_loader10 = DataLoader(test_embed10, batch_size=cfg.batch_size_classif, shuffle=False,
                                     num_workers=num_workers, drop_last=False)

    model = to_cpu(model)
    classif_model = to_cuda_if_available(classif_model)
    classif_model.eval()
    mean_test_results1 = measure_classif(classif_model, test_embed_loader1,
                                         classes=classes,
                                         suffix_print="test1")

    mean_test_results10 = measure_classif(classif_model, test_embed_loader10,
                                          classes=classes,
                                          suffix_print="test10")

    print(f"Time of the program: {time.time() - t}")
