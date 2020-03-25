#!/projects/pul51/shared/calcul/users/nturpault/anaconda3/envs/pytorch/bin/python
import argparse
import os
import os.path as osp
from copy import deepcopy
import time
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from DataLoad import DataLoadDf
from utils.Transforms import ApplyLog, Unsqueeze, ToTensor, View, Normalize, Compose
from utils.Samplers import CategoriesSampler
from pprint import pformat
import config as cfg
from DesedSynthetic import DesedSynthetic
from evaluation_measures import get_f_measure_by_class, measure_classif
from common import get_model, get_optimizer, shared_args, get_dfs, measure_embeddings
from models.FullyConnected import FullyConnected
from models.CombineModel import CombineModel
from utils.Logger import LOG
from utils.Scaler import ScalerSum
from utils.utils import ManyHotEncoder, create_folder, to_cuda_if_available, EarlyStopping, SaveBest, to_cpu, \
    load_model, save_model, ViewModule


if __name__ == '__main__':
    LOG.info(__file__)
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers_classif', type=int, default=1)
    parser.add_argument('--conv_dropout', type=float, default=cfg.conv_dropout)
    parser.add_argument('--dropout_classif', type=float, default=cfg.dropout_non_recurrent)
    parser.add_argument('--nb_layers', type=int, default=cfg.nb_layers)
    parser.add_argument('--pool_freq', type=int, default=cfg.pool_freq)
    parser.add_argument('--last_layer', type=int, default=cfg.last_layer)
    parser.add_argument('--epochs', type=float, default=cfg.n_epoch_classifier)

    parser = shared_args(parser)

    args = parser.parse_args()
    pformat(vars(args))

    n_epochs = round(args.epochs)
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

    model_directory, log_directory = cfg.get_dirs("baseline")
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
    LOG.info("train_classes repartition: \n {}".format(train_weak_df.event_labels.value_counts()))
    classes = dataset.classes
    many_hot_encoder = ManyHotEncoder(classes)

    # Model
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    list_trans_fr = [ApplyLog(), ToTensor(), Unsqueeze(0)]

    if args.segment:
        list_trans_fr.append(Unsqueeze(0))

    train_set = DataLoadDf(train_weak_df,
                           many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)
    if args.balance:
        train_sampler = CategoriesSampler(train_set.df.event_labels, classes, round(cfg.batch_size / len(classes)))
        train_load = DataLoader(train_set, num_workers=num_workers, batch_sampler=train_sampler)
    else:
        train_load = DataLoader(train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        train_sampler = train_load.batch_sampler
    LOG.debug("len train : {}".format(len(train_set)))
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
        if args.balance:
            val_sampler = CategoriesSampler(valid_set.df.event_labels, classes, round(cfg.batch_size / len(classes)))
            valid_load = DataLoader(valid_set, num_workers=num_workers, batch_sampler=val_sampler)
        else:
            valid_load = DataLoader(valid_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
            val_sampler = valid_load.batch_sampler

    test_df = dfs["test"]
    test_set = DataLoadDf(test_df,
                          many_hot_encoder.encode_weak, Compose(list_trans_fr), return_indexes=False)
    LOG.info("len test: {}".format(len(test_set)))
    if args.balance:
        test_sampler = CategoriesSampler(test_set.df.event_labels, classes, round(cfg.batch_size / len(classes)))
        test_load = DataLoader(test_set, num_workers=num_workers, batch_sampler=test_sampler)
    else:
        test_load = DataLoader(test_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
        test_sampler = test_load.batch_sampler

    trans_emb = deepcopy(list_trans_fr)
    if not args.segment:
        trans_emb.append(Unsqueeze(0))

    train_set_emb = DataLoadDf(train_weak_df, many_hot_encoder.encode_weak,
                               transform=Compose(trans_emb))
    valid_set_val = DataLoadDf(valid_weak_df, many_hot_encoder.encode_weak,
                               transform=Compose(trans_emb))
    test_set_val = DataLoadDf(test_df, many_hot_encoder.encode_weak,
                              transform=Compose(trans_emb))

    emb_state = {"scaler": scaler.state_dict(),
                 "many_hot_encoder": many_hot_encoder.state_dict()}
    emb_model, emb_state = get_model(emb_state, args)
    emb_model = to_cuda_if_available(emb_model)
    # Classif_model
    if args.segment:
        X, y = train_set[0]
    else:
        X, y = next(iter(train_load))
    X = to_cuda_if_available(X)
    emb = emb_model(X)
    LOG.info("shape input CNN: x {}, y {}".format(X.shape, y.shape))
    LOG.info("shape after CNN: {}".format(emb.shape))

    if args.n_layers_classif == 2:
        dimensions = [32, 16]
    elif args.n_layers_classif == 1:
        dimensions = [32]
    classif_args = (emb.shape[-1], dimensions, 10)
    if args.single_label:
        final_activation = "log_softmax"
    else:
        final_activation = "sigmoid"
    classif_kwargs = {"batch_norm": cfg.batch_norm, "activation": cfg.activation,
                      "dropout": args.dropout_classif, "final_activation": final_activation
                      }
    init_classif_name = os.path.join("stored_data", "model", "init_FC")
    # no_load = True
    # if os.path.exists(init_classif_name):
    #     try:
    #         classif_model, classif_state = load_model(init_classif_name, return_state=True)
    #         no_load = False
    #     except (RuntimeError, TypeError) as e:
    #         LOG.warn("Init model couldn't be load, rewritting the file")
    # if no_load:
    fc_model = FullyConnected(*classif_args, **classif_kwargs)
    fc_model = to_cuda_if_available(fc_model)
    # model.eval()
    # for i, child in enumerate(model.children()):
    #     for param in child.parameters():
    #         param.requires_grad = False

    if args.agg_time is not None:
        model = CombineModel(emb_model, fc_model)
    else:
        model = CombineModel(emb_model, ViewModule((-1, round(cfg.max_len_seconds / cfg.frames_in_sec), emb.shape[-1])),
                             fc_model)
    print(round(cfg.max_len_seconds / cfg.frames_in_sec))
    # for name, child in classif_model.named_children():
    #     for cname, param in child.named_parameters():
    #         print("{:30}, {:8}: requires_grad: {}".format(name, cname, param.requires_grad))

    state = {
        'model': {"name": [emb_model.__class__.__name__, fc_model.__class__.__name__],
                  'args': {
                      emb_model.__class__.__name__: emb_state["model"]["args"],
                      fc_model.__class__.__name__:  classif_args},
                  'kwargs': {
                      emb_model.__class__.__name__: emb_state["model"]["kwargs"],
                      fc_model.__class__.__name__: classif_kwargs},
                  'state_dict': model.state_dict()
                  },
    }

    optimizer, state = get_optimizer(model, state)

    criterion_bce = torch.nn.NLLLoss()  # torch.nn.BCELoss()
    model, criterion_bce = to_cuda_if_available(model, criterion_bce)
    LOG.info(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("number of parameters in the model: {}".format(pytorch_total_params))

    early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup",
                                        init_patience=cfg.first_early_wait)
    save_best_call = SaveBest(val_comp="sup")

    print(optimizer)

    save_results = pd.DataFrame()

    model_name_sup = osp.join(model_directory, "classif")
    create_folder(model_name_sup)
    if cfg.save_best:
        model_path_sup1 = os.path.join(model_name_sup, "best_model")
    else:
        model_path_sup1 = os.path.join(model_name_sup, "epoch_" + str(n_epochs))
    print("path of model : " + model_path_sup1)

    state['many_hot_encoder'] = many_hot_encoder.state_dict()

    def train_loop(train_load, model):
        loss_bce = []
        if args.segment:
            for cnt, indexes in enumerate(train_load.batch_sampler):
                optimizer.zero_grad()
                for j, ind in enumerate(indexes):
                    inputs, pred_labels = train_set[ind]
                    if cnt == 0 and epoch_ == 0:
                        LOG.debug("classif input shape: {}".format(inputs.shape))

                    # zero the parameter gradients
                    inputs, pred_labels = to_cuda_if_available(inputs, pred_labels)

                    # forward + backward + optimize
                    weak_out = model(inputs)
                    loss_bce = criterion_bce(weak_out, pred_labels.argmax(0, keepdim=True))
                    loss_bce.backward()
                    loss_bce.append(loss_bce.item())
                optimizer.step()
        else:
            for cnt, samples in enumerate(train_load):
                optimizer.zero_grad()
                inputs, pred_labels = samples
                if cnt == 0 and epoch_ == 0:
                    LOG.debug("classif input shape: {}".format(inputs.shape))

                # zero the parameter gradients
                inputs, pred_labels = to_cuda_if_available(inputs, pred_labels)

                # forward + backward + optimize
                weak_out = model(inputs)
                loss_bce = criterion_bce(weak_out, pred_labels)
                loss_bce.backward()
                loss_bce.append(loss_bce.item())
                optimizer.step()
        loss_bce = np.mean(loss_bce)
        print('[%d / %d, %5d] loss: %.3f' %
              (epoch_ + 1, n_epochs, cnt + 1, loss_bce))
        return loss_bce, model

    if not os.path.exists(model_path_sup1) or cfg.recompute_classif:
        for epoch_ in range(n_epochs):
            start = time.time()
            loss_mean_bce, model = train_loop(train_load, model)

            model.eval()
            macro_f_measure_train = get_f_measure_by_class(model, len(classes), train_set_emb, max=args.single_label)

            macro_f_measure_val = get_f_measure_by_class(model, len(classes), valid_set_val, max=args.single_label)
            mean_macro_f_measure = np.mean(macro_f_measure_val)

            macro_f_measure_test = get_f_measure_by_class(model, len(classes), test_set_val, max=args.single_label)
            model.train()
            print("Time to train an epoch: {}".format(time.time() - start))
            # print statistics

            results = {"train_loss": loss_mean_bce,
                       "macro_measure_train": np.mean(macro_f_measure_train),
                       "class_macro_train": np.array_str(macro_f_measure_train, precision=2),
                       "macro_measure_valid": mean_macro_f_measure,
                       "macro_measure_test": np.mean(macro_f_measure_test),
                       }
            for key in results:
                LOG.info("\t\t ---->  {} : {}".format(key, results[key]))

            save_results = save_results.append(results, ignore_index=True)
            save_results.to_csv(os.path.join(log_directory, "baseline" + ".csv"),
                                sep="\t", header=True, index=False)
            # scheduler.step(mean_macro_f_measure)

            # ##########
            # # Callbacks
            # ##########
            state["optimizer"]["state_dict"] = optimizer.state_dict()
            state['epoch'] = epoch_ + 1
            state["model"]["state_dict"] = model.state_dict()
            state["loss"] = loss_mean_bce
            state.update(results)

            if cfg.early_stopping is not None:
                if early_stopping_call.apply(mean_macro_f_measure):
                    print("EARLY STOPPING")
                    break

            if cfg.save_best and save_best_call.apply(mean_macro_f_measure):
                save_model(state, model_path_sup1)

            if cfg.model_checkpoint is not None:
                if epoch_ % cfg.model_checkpoint == cfg.model_checkpoint - 1:
                    model_path_chkpt = os.path.join(model_name_sup, "epoch_" + str(epoch_))
                    save_model(state, model_path_chkpt)

        if cfg.save_best:
            LOG.info(
                "best model at epoch : {} with macro {}".format(save_best_call.best_epoch, save_best_call.best_val))
            LOG.info("loading model from: {}".format(model_path_sup1))
            model, state = load_model(model_path_sup1, return_optimizer=False, return_state=True)
        else:
            model_path_sup1 = os.path.join(model_name_sup, "epoch_" + str(n_epochs))
            save_model(state, model_path_sup1)
        LOG.debug("model path: {}".format(model_path_sup1))
        LOG.debug('Finished Training')
    else:
        model, state = load_model(model_path_sup1, return_optimizer=False, return_state=True)
        LOG.info("Loaded model at epoch: {}".format(state.get("epoch")))
    LOG.info("#### End classif")
    # save_results.to_csv(os.path.join(log_directory, model_name_sup + ".csv"), sep="\t", header=True, index=False)

    model.eval()
    mean_test_results = measure_classif(model, test_set_val, classes=classes,
                                        suffix_print="test")
    if eval_path is not None:
        eval_df = dfs["eval"]
        eval_set = DataLoadDf(eval_df,
                              many_hot_encoder.encode_weak, Compose(trans_emb), return_indexes=False)
        mean_eval_results = measure_classif(model, eval_set, classes=classes,
                                            suffix_print="eval", single_label=args.single_label)

    test_set1 = DataLoadDf(dfs["test1"], many_hot_encoder.encode_weak, transform=Compose(trans_emb))
    test_set10 = DataLoadDf(dfs["test10"], many_hot_encoder.encode_weak, transform=Compose(trans_emb))
    mean_test_results1 = measure_classif(model, test_set1,
                                         classes=classes,
                                         suffix_print="test1", single_label=args.single_label)

    mean_test_results10 = measure_classif(model, test_set10,
                                          classes=classes,
                                          suffix_print="test10", single_label=args.single_label)

    # Test
    emb_model.eval()
    base_model_name = "baseline"
    embed_dir = "stored_data/embeddings"
    embed_dir = os.path.join(embed_dir, base_model_name)
    create_folder(embed_dir)
    fig_dir = os.path.join(embed_dir, "figures")
    create_folder(fig_dir)

    if args.agg_time is not None:
        trans_embedding = [ToTensor(), View((emb_state["nb_frames_staying"], -1))]
    else:
        trans_embedding = [ToTensor()]

    embed_set = "final"
    train_embed_dir = os.path.join(embed_dir, embed_set)

    measure_embeddings(train_set_emb, model.models[0], osp.join(embed_dir, "train"),
                       figure_path=osp.join(fig_dir, "train"), set_name="train")
    measure_embeddings(valid_set_val, model.models[0], osp.join(embed_dir, "valid"),
                       figure_path=osp.join(fig_dir, "valid"), set_name="valid")
    measure_embeddings(test_set_val, model.models[0], osp.join(embed_dir, "test"),
                       figure_path=osp.join(fig_dir, "test"), set_name="test")

    model.eval()

    print("Time of the program: {}".format(time.time() - t))
    from orion.client import report_results

    report_results(
        [dict(
            name="mean_test_results",
            type="objective",
            value=float(100 - state["macro_measure_valid"]*100)
        )
        ]
    )
# from orion.core.io.experiment_builder import ExperimentBuilder
# from orion.storage.base import Storage
# ep = ExperimentBuilder()
# aa = ep.fetch_full_config({}, use_db=False)
# st = Storage(of_type="legacy", config=aa)
# # get experiments
# st.fetch_experiments({})
# st._db.read("experiments")
# st._db.read("experiments", {'name': 'baselineoptasha3'})
# st._db.remove("experiments", {'name': 'baselineoptasha3'})
