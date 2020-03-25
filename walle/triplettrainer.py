import os
import time

import numpy as np
import pandas as pd
from torch import nn

import config as cfg
from evaluation_measures import get_f_measure_by_class
from utils.utils import create_folder, to_cuda_if_available, EarlyStopping, SaveBest, load_model, save_model, to_cpu

from utils.Logger import LOG


def train_classifier(train_loader, classif_model, optimizer_classif, many_hot_encoder=None,
                     valid_loader=None, state={},
                     dir_model="model", result_path="res", recompute=True):
    criterion_bce = nn.BCELoss()
    classif_model, criterion_bce = to_cuda_if_available(classif_model, criterion_bce)
    print(classif_model)

    early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup",
                                        init_patience=cfg.first_early_wait)
    save_best_call = SaveBest(val_comp="sup")

    # scheduler = ReduceLROnPlateau(optimizer_classif, 'max', factor=0.1, patience=cfg.reduce_lr,
    #                               verbose=True)
    print(optimizer_classif)

    save_results = pd.DataFrame()

    create_folder(dir_model)
    if cfg.save_best:
        model_path_sup1 = os.path.join(dir_model, "best_model")
    else:
        model_path_sup1 = os.path.join(dir_model, "epoch_" + str(cfg.n_epoch_classifier))
    print("path of model : " + model_path_sup1)

    state['many_hot_encoder'] = many_hot_encoder.state_dict()

    if not os.path.exists(model_path_sup1) or recompute:
        for epoch_ in range(cfg.n_epoch_classifier):
            print(classif_model.training)
            start = time.time()
            loss_mean_bce = []
            for i, samples in enumerate(train_loader):
                inputs, pred_labels = samples
                if i == 0:
                    LOG.debug("classif input shape: {}".format(inputs.shape))

                # zero the parameter gradients
                optimizer_classif.zero_grad()
                inputs = to_cuda_if_available(inputs)

                # forward + backward + optimize
                weak_out = classif_model(inputs)
                weak_out = to_cpu(weak_out)
                # print(output)
                loss_bce = criterion_bce(weak_out, pred_labels)
                loss_mean_bce.append(loss_bce.item())
                loss_bce.backward()
                optimizer_classif.step()

            loss_mean_bce = np.mean(loss_mean_bce)
            classif_model.eval()
            n_class = len(many_hot_encoder.labels)
            macro_f_measure_train = get_f_measure_by_class(classif_model, n_class,
                                                           train_loader)
            if valid_loader is not None:
                macro_f_measure = get_f_measure_by_class(classif_model, n_class,
                                                         valid_loader)
                mean_macro_f_measure = np.mean(macro_f_measure)
            else:
                mean_macro_f_measure = -1
            classif_model.train()
            print("Time to train an epoch: {}".format(time.time() - start))
            # print statistics
            print('[%d / %d, %5d] loss: %.3f' %
                  (epoch_ + 1, cfg.n_epoch_classifier, i + 1, loss_mean_bce))

            results = {"train_loss": loss_mean_bce,
                       "macro_measure_train": np.mean(macro_f_measure_train),
                       "class_macro_train": np.array_str(macro_f_measure_train, precision=2),
                       "macro_measure_valid": mean_macro_f_measure,
                       "class_macro_valid": np.array_str(macro_f_measure, precision=2),
                       }
            for key in results:
                LOG.info("\t\t ---->  {} : {}".format(key, results[key]))

            save_results = save_results.append(results, ignore_index=True)
            # scheduler.step(mean_macro_f_measure)

            # ##########
            # # Callbacks
            # ##########
            state['epoch'] = epoch_ + 1
            state["model"]["state_dict"] = classif_model.state_dict()
            state["optimizer"]["state_dict"] = optimizer_classif.state_dict()
            state["loss"] = loss_mean_bce
            state.update(results)

            if cfg.early_stopping is not None:
                if early_stopping_call.apply(mean_macro_f_measure):
                    print("EARLY STOPPING")
                    break

            if cfg.save_best and save_best_call.apply(mean_macro_f_measure):
                save_model(state, model_path_sup1)

        if cfg.save_best:
            LOG.info(
                "best model at epoch : {} with macro {}".format(save_best_call.best_epoch, save_best_call.best_val))
            LOG.info("loading model from: {}".format(model_path_sup1))
            classif_model, state = load_model(model_path_sup1, return_optimizer=False, return_state=True)
        else:
            model_path_sup1 = os.path.join(dir_model, "epoch_" + str(cfg.n_epoch_classifier))
            save_model(state, model_path_sup1)
        LOG.debug("model path: {}".format(model_path_sup1))
        LOG.debug('Finished Training')
    else:
        classif_model, state = load_model(model_path_sup1, return_optimizer=False, return_state=True)
    LOG.info("#### End classif")
    save_results.to_csv(result_path, sep="\t", header=True, index=False)

    return classif_model, state
