import os

import sklearn
from sklearn.manifold import TSNE

from torch import optim
from torch.utils.data import DataLoader

import config as cfg
from DataLoad import DataLoadDf
from Embedding import calculate_embedding, tsne_plots, scatter_ratio, proto_acc
from evaluation_measures import measure_classif
from models.CNN import CNN
from models.FullyConnected import FullyConnected
from triplettrainer import train_classifier
from utils.Logger import LOG
from utils.Samplers import CategoriesSampler
from utils.Transforms import ToTensor, View, Compose
from utils.utils import load_model, save_model, AdamW, create_folder, to_cuda_if_available, to_cpu


def get_model(state, args, init_model_name=None):
    if init_model_name is not None and os.path.exists(init_model_name):
        model, optimizer, state = load_model(init_model_name, return_optimizer=True, return_state=True)
    else:
        if "conv_dropout" in args:
            conv_dropout = args.conv_dropout
        else:
            conv_dropout = cfg.conv_dropout
        cnn_args = {1}

        if args.fixed_segment is not None:
            frames = cfg.frames
        else:
            frames = None

        nb_layers = 4
        cnn_kwargs = {"activation": cfg.activation, "conv_dropout": conv_dropout, "batch_norm": cfg.batch_norm,
                      "kernel_size": nb_layers * [3], "padding": nb_layers * [1],
                      "stride": nb_layers * [1], "nb_filters": [16, 16, 32, 65],
                      "pooling": [(2, 2), (2, 2), (1, 4), (1, 2)],
                      "aggregation": args.agg_time, "norm_out": args.norm_embed, "frames": frames,
                      }
        nb_frames_staying = cfg.frames // (2 ** 2)
        model = CNN(*cnn_args, **cnn_kwargs)
        # model.apply(weights_init)
        state.update({
            'model': {"name": model.__class__.__name__,
                      'args': cnn_args,
                      "kwargs": cnn_kwargs,
                      'state_dict': model.state_dict()},
            'nb_frames_staying': nb_frames_staying
        })
        if init_model_name is not None:
            save_model(state, init_model_name)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("number of parameters in the model: {}".format(pytorch_total_params))
    return model, state


def get_optimizer(model, state):
    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs)
    state['optimizer'] = {"name": optimizer.__class__.__name__,
                          'args': '',
                          "kwargs": optim_kwargs,
                          'state_dict': optimizer.state_dict()
                          }
    return optimizer, state


def shared_args(parser):
    parser.add_argument('--subpart_data', type=int, default=None, dest="subpart_data")
    parser.add_argument('--weak_file', type=str, default="weak")
    parser.add_argument('--agg_time', type=str, default="mean")
    parser.add_argument('--frames_in_sec', type=float, default=None)
    parser.add_argument('--segment', action="store_true", default=False)
    parser.add_argument('--dropna', action="store_true", default=False)
    parser.add_argument('--long_events', action="store_true", default=False)
    parser.add_argument('--norm_embed', action="store_true", default=False)
    parser.add_argument('--balance', action="store_true", default=False)
    parser.add_argument('--unique_fr', action="store_true", default=False)
    parser.add_argument('--fixed_segment', type=float, default=None)
    parser.add_argument('--single_label', action="store_true", default=False)
    return parser


def datasets_classif(model, train_weak_embed, valid_weak_dl_fr, test_dl_fr, args, many_hot_encoder, classes,
                     save_name="", eval_dl=None):
    encode_function_label = many_hot_encoder.encode_weak
    num_workers = cfg.num_workers
    model.eval()
    embed_dir = "stored_data/embeddings"
    embed_dir = os.path.join(embed_dir, save_name)
    create_folder(embed_dir)
    fig_dir = os.path.join(embed_dir, "figures")
    create_folder(fig_dir)

    if args.agg_time is not None:
        trans_embedding = [ToTensor(), View(-1)]
    else:
        trans_embedding = [ToTensor()]

    model = to_cuda_if_available(model)
    embed_set = "final"
    train_embed_dir = os.path.join(embed_dir, embed_set)
    df_weak, embed_weak = calculate_embedding(train_weak_embed, model, savedir=train_embed_dir,
                                              concatenate="append")
    weak_embed = DataLoadDf(df_weak, encode_function_label, transform=Compose(trans_embedding))
    LOG.info(f"len weak embed: {len(weak_embed)}")
    weak_embed.set_transform(Compose(trans_embedding))

    batch_size_classif = cfg.batch_size_classif
    df_valid, embed_valid = calculate_embedding(valid_weak_dl_fr, model, savedir=train_embed_dir,
                                                concatenate="append")

    valid_embed = DataLoadDf(df_valid, encode_function_label, transform=Compose(trans_embedding))
    embed_set = "final_test"
    test_embed_dir = os.path.join(embed_dir, embed_set)
    df_test_embed, emb_test = calculate_embedding(test_dl_fr, model, savedir=test_embed_dir,
                                                  concatenate="append")

    test_embed = DataLoadDf(df_test_embed, encode_function_label,
                            transform=Compose(trans_embedding))

    if args.balance:
        n_per_class = max(round(batch_size_classif / len(classes)), 1)
        weak_sampler = CategoriesSampler(weak_embed.df.event_labels, classes, n_per_class)
        weak_embed_loader = DataLoader(weak_embed, batch_sampler=weak_sampler, num_workers=num_workers)
        valid_sampler = CategoriesSampler(valid_embed.df.event_labels, classes, n_per_class)
        valid_embed_loader = DataLoader(valid_embed, batch_sampler=valid_sampler, num_workers=num_workers)
        test_sampler = CategoriesSampler(test_embed.df.event_labels, classes, n_per_class)
        test_embed_loader = DataLoader(test_embed, batch_sampler=test_sampler, num_workers=num_workers)
    else:
        weak_embed_loader = DataLoader(weak_embed, batch_size=batch_size_classif, num_workers=num_workers, shuffle=True,
                                       drop_last=True)
        valid_embed_loader = DataLoader(valid_embed, batch_size=batch_size_classif, shuffle=False,
                                        num_workers=num_workers,
                                        drop_last=False)
        test_embed_loader = DataLoader(test_embed, batch_size=batch_size_classif, shuffle=False,
                                       num_workers=num_workers,
                                       drop_last=False)

    if eval_dl is not None:
        model = to_cuda_if_available(model)
        embed_set = "final_eval"
        eval_embed_dir = os.path.join(embed_dir, embed_set)
        df_eval_embed, embed_eval = calculate_embedding(eval_dl, model, savedir=eval_embed_dir,
                                                        concatenate="append")

        eval_embed = DataLoadDf(df_eval_embed, encode_function_label,
                                transform=Compose(trans_embedding))
        if args.balance:
            eval_sampler = CategoriesSampler(eval_embed.df.event_labels, classes, n_per_class)
            eval_embed_loader = DataLoader(eval_embed, batch_sampler=eval_sampler, num_workers=num_workers)
        else:
            eval_embed_loader = DataLoader(eval_embed, batch_size=batch_size_classif, shuffle=False,
                                           num_workers=num_workers, drop_last=False)
    else:
        eval_embed_loader = None

    model = to_cpu(model)
    return {"train": weak_embed_loader,
            "valid": valid_embed_loader,
            "test": test_embed_loader,
            "eval": eval_embed_loader}


def do_classif(dataloaders, many_hot_encoder, classes,
               save_model_dir="", result_path="res.csv"):
    weak_embed_loader = dataloaders["train"]
    valid_embed_loader = dataloaders["valid"]
    test_embed_loader = dataloaders["test"]
    eval_embed_loader = dataloaders.get("eval")
    # ###########
    # Eval
    # ########
    # Define model and optimizer
    x, y = next(iter(weak_embed_loader))
    classif_args = (x.shape[-1], [32], 10)
    classif_kwargs = {"batch_norm": cfg.batch_norm, "activation": cfg.activation, "dropout": cfg.dropout_non_recurrent}
    optim_kwargs = {"lr": 0.001, "betas": (0.9, 0.999)}

    init_classif_name = os.path.join("stored_data", "model", "init_FC" + str(x.shape[-1]))
    no_load = True
    if os.path.exists(init_classif_name):
        try:
            LOG.info("load init model")
            classif_model, classif_state = load_model(init_classif_name, return_state=True)
            no_load = False
        except (RuntimeError, TypeError) as e:
            LOG.warn("Init model couldn't be load, rewritting the file")
            LOG.warn(e)
    if no_load:
        classif_model = FullyConnected(*classif_args, **classif_kwargs)
        classif_state = {
            'model': {"name": classif_model.__class__.__name__,
                      'args': classif_args,
                      "kwargs": classif_kwargs,
                      'state_dict': classif_model.state_dict()}
        }
        save_model(classif_state, init_classif_name)

    optimizer_classif = optim.Adam(filter(lambda p: p.requires_grad, classif_model.parameters()),
                                   **optim_kwargs)
    classif_state['optimizer'] = {"name": optimizer_classif.__class__.__name__,
                                  'args': '',
                                  "kwargs": optim_kwargs,
                                  'state_dict': optimizer_classif.state_dict()
                                  }
    classif_model, state = train_classifier(weak_embed_loader,
                                            classif_model,
                                            optimizer_classif,
                                            many_hot_encoder=many_hot_encoder,
                                            valid_loader=valid_embed_loader,
                                            state=classif_state,
                                            dir_model=save_model_dir,
                                            result_path=result_path,
                                            recompute=cfg.recompute_classif)
    classif_model.eval()

    mean_test_results = measure_classif(classif_model, test_embed_loader, classes=classes, suffix_print="test")
    state["mean_test_results"] = mean_test_results

    if eval_embed_loader is not None:
        mean_eval_results = measure_classif(classif_model, eval_embed_loader, classes=classes, suffix_print="eval")
        state["mean_eval_results"] = mean_eval_results

    return classif_model, state


def get_dfs(dataset, weak_path, test_path, eval_path=None, subpart_data=None, valid_list=None,
            frames_in_sec=None, segment=False, dropna=True,
            unique_fr=False, fixed_segment=False):
    weak_df_fr = dataset.get_df_feat_dir(weak_path, subpart_data=subpart_data, segment=segment,
                                         frames_in_sec=frames_in_sec, fixed_segment=fixed_segment)

    if unique_fr:
        if segment:
            raise NotImplementedError("cannot use unique fr with segment")

        def take_mid_fr(x):
            if len(x) > 2:
                x = x.iloc[1:-1]
            return x.sample(n=1)
        l_keep = weak_df_fr.groupby("raw_filename").apply(take_mid_fr).filename.tolist()
        weak_df_fr = weak_df_fr[weak_df_fr.filename.isin(l_keep)].reset_index(drop=True)

    if dropna:
        weak_df_fr = weak_df_fr.dropna().reset_index(drop=True)
        print("DROP NANS")
    valid_weak_df_fr = weak_df_fr[weak_df_fr.raw_filename.isin(valid_list)]
    train_weak_df_fr = weak_df_fr.drop(valid_weak_df_fr.index).reset_index(drop=True)
    valid_weak_df_fr = valid_weak_df_fr.reset_index(drop=True)
    valid_weak_df_fr = valid_weak_df_fr.dropna().reset_index(drop=True)
    valid_weak_df_fr = valid_weak_df_fr[~valid_weak_df_fr.event_labels.fillna("").str.contains(",")]
    valid_weak_df_fr = valid_weak_df_fr.reset_index(drop=True)

    LOG.debug("len weak df frames : {}".format(len(weak_df_fr)))
    LOG.debug("len train weak df frames : {}".format(len(train_weak_df_fr)))
    LOG.debug("len valid weak df frames : {}".format(len(valid_weak_df_fr)))

    # Todo, remove hard coded stuff
    test_df_fr = dataset.get_df_feat_dir(test_path, subpart_data=subpart_data, segment=segment,
                                         frames_in_sec=frames_in_sec,
                                         fixed_segment=0.2)
    test_df_fr = test_df_fr.dropna().reset_index(drop=True)
    test_df_1 = dataset.get_df_feat_dir(test_path, subpart_data=subpart_data, segment=segment,
                                        frames_in_sec=frames_in_sec, fixed_segment=1)
    test_df_1 = test_df_1.dropna().reset_index(drop=True)
    test_df_10 = dataset.get_df_feat_dir(test_path, subpart_data=subpart_data, segment=segment,
                                         frames_in_sec=frames_in_sec, fixed_segment=10)
    test_df_10 = test_df_10.dropna().reset_index(drop=True)

    print("drop test nans")
    if eval_path is not None:
        eval_df_fr = dataset.get_df_feat_dir(eval_path, subpart_data=subpart_data, segment=segment,
                                             frames_in_sec=frames_in_sec, fixed_segment=fixed_segment)
    else:
        eval_df_fr = None

    dfs = {
        "train": train_weak_df_fr,
        "valid": valid_weak_df_fr,
        "test": test_df_fr,
        "test1": test_df_1,
        "test10": test_df_10,
        "eval": eval_df_fr
    }
    return dfs


def measure_embeddings(set_embed, model, emb_path, figure_path, set_name=''):
    df, embed = calculate_embedding(set_embed, model,
                                    savedir=emb_path, concatenate="append")
    df = df.dropna()
    embed = embed[df.index]
    LOG.debug("embed shape: {}".format(embed.shape))
    LOG.debug("df shape: {}".format(df.shape))

    tsne_emb = TSNE().fit_transform(X=embed.reshape(embed.shape[0], -1))
    tsne_plots(tsne_emb, df, savefig=figure_path)
    scatter = scatter_ratio(embed.reshape(embed.shape[0], -1), df.reset_index())
    silhouette = sklearn.metrics.silhouette_score(
        embed.reshape(embed.shape[0], -1), df.event_labels, metric='euclidean')
    # Just informative
    LOG.info(f"{set_name} silhouette for all classes in 2D (tsne) : "
             f"{sklearn.metrics.silhouette_score(df[['X', 'Y']], df.event_labels, metric='euclidean')}")

    proto = proto_acc(embed.reshape(embed.shape[0], -1), df.reset_index())
    LOG.info("Proto accuracy {} : {}".format(set_name, proto))

    return {
        "scatter" + set_name: scatter,
        "silhouette" + set_name: silhouette,
        "proto" + set_name: proto
    }
