import argparse
import os
import time
from pprint import pformat

import scipy
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from DesedSynthetic import DesedSynthetic
from utils.Scaler import ScalerSum
from utils.utils import create_folder, to_cuda_if_available, to_cpu
from utils.Logger import LOG

if torch.cuda.is_available():
    from tsnecuda import TSNE
else:
    from sklearn.manifold import TSNE

plt.style.use("seaborn-darkgrid")


def get_embeddings_numpy(inputs, model, flatten=True):
    """ Get embeddings of a model. Assume inputs and model are in the same device"""
    try:
        embed = model.get_embedding(inputs)
    except AttributeError:
        embed = model(inputs)
    embed = to_cpu(embed)
    embed = embed.detach().numpy()
    if flatten:
        embed = embed.reshape(embed.shape[0], -1)
    return embed


def calculate_embedding(embedding_dl, model, savedir=None, concatenate=None, squeeze=True):
    # If frames, assume the savedir name or the filename is different than when it is not defined
    model.eval()
    if savedir is not None:
        create_folder(savedir)
    df = embedding_dl.df.copy()
    df.filename = df.filename.apply(lambda x: os.path.join(savedir, os.path.basename(x)))
    if savedir is not None:
        df.to_csv(os.path.join(savedir, "df"), sep="\t", index=False)
    if concatenate is not None:
        concat_embed = []
    for cnt, (data_in, y) in enumerate(embedding_dl):
        data_in = to_cuda_if_available(data_in)

        emb = get_embeddings_numpy(data_in, model, flatten=False)
        if cnt == 0:
            LOG.debug(f"shapes: input: {data_in.shape}, embed: {emb.shape}, dir: {savedir}")
        if squeeze:
            emb = np.squeeze(emb)
        if savedir is not None:
            np.save(df.iloc[cnt].filename, emb)

        if concatenate == "append":
            concat_embed.append(emb)
        elif concatenate == "extend":
            concat_embed.extend(emb)
        else:
            if concatenate is not None:
                raise NotImplementedError("Impossible to aggregate with this value")

    model.train()
    if concatenate is not None:
        concat_embed = np.array(concat_embed)
        return df, concat_embed
    return df


def get_embedding(model, data, frames):
    model.eval()
    embed = []
    for i, (x, y) in enumerate(data):
        x = x[:, :(x.shape[-2] - x.shape[-2] % frames), :]
        x = x.view(-1, 1, frames, x.shape[-1])
        inputs = x
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        # forward + backward + optimize
        outputs = model.cnn(inputs).cpu()
        outputs = outputs.squeeze(-1)
        outputs = outputs.permute(0, 2, 1)
        embed.extend(outputs.data.numpy())
    embed = np.array(embed)
    model.train()
    return embed


# Plots
def get_distance(a, b):
    d = (b - a) ** 2
    d = d.sum()
    return np.sqrt(d)


def get_triangle_area(cent1, cent2, cent3):
    dist1 = get_distance(cent1, cent2)
    dist2 = get_distance(cent1, cent3)
    dist3 = get_distance(cent2, cent3)
    s = (dist1 + dist2 + dist3) / 2
    area = np.sqrt(s * (s - dist1) * (s - dist2) * (s - dist3))
    return area


def get_centroids(df, classes):
    coordinates = []
    for c in classes:
        coor = df[df.event_labels.str.contains(c)][["X", "Y"]]
        dis = (((coor - coor.mean()) ** 2).sum(axis=1) ** 0.5)
        centroid = coor.iloc[dis.argsort()[:int(0.8 * len(dis))]].mean()
        coordinates.append(centroid)
    coordinates = np.array(coordinates)
    centroids = pd.DataFrame({"classe": classes, "X": coordinates[:, 0], "Y": coordinates[:, 1]})
    return centroids


def get_best_classes(centroids, classes):
    triangle_areas = []
    triplet_classes = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            for k in range(j + 1, len(classes)):
                c1_coor = centroids[centroids.classe == classes[i]][["X", "Y"]].values
                c2_coor = centroids[centroids.classe == classes[j]][["X", "Y"]].values
                c3_coor = centroids[centroids.classe == classes[k]][["X", "Y"]].values
                triangle_area = get_triangle_area(c1_coor, c2_coor, c3_coor)
                triangle_areas.append(triangle_area)
                triplet_classes.append([classes[i], classes[j], classes[k]])
    triplet_classes = np.array(triplet_classes)
    triangle_areas = np.array(triangle_areas)
    return triplet_classes[triangle_areas.argmax()]


def get_df_best_classes(df, triplet_classes):
    print(triplet_classes)
    df1 = df[df.event_labels.str.contains(triplet_classes[0])]
    df1.loc[:, "color"] = 0
    df1.loc[:, "X"] += 0.5
    df1.loc[:, "Y"] += 0.2
    df1.loc[:, "label"] = triplet_classes[0]

    df2 = df[df.event_labels.str.contains(triplet_classes[1])]
    df2.loc[:, "color"] = 1
    df2.loc[:, "X"] -= 0.5
    df2.loc[:, "Y"] += 0.2
    df2.loc[:, "label"] = triplet_classes[1]

    df3 = df[df.event_labels.str.contains(triplet_classes[2])]
    df3.loc[:, "color"] = 2
    # df1["X"] += 0.5
    df3.loc[:, "Y"] -= 0.5
    df3.loc[:, "label"] = triplet_classes[2]

    df = pd.concat([df1, df2, df3])

    return df


def plot_best_classes(df, classes, savefig=None):
    centroids = get_centroids(df, classes)
    best_classes = get_best_classes(centroids, classes)
    dfplot = get_df_best_classes(df, best_classes)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    for i, c in enumerate(best_classes):
        df1 = dfplot[dfplot.event_labels.str.contains(c)]
        ax.plot(df1.X, df1.Y, marker='o', linestyle='', ms=5, label=c, c=plt.cm.tab10(i))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)

    # ax.set_title("Model {}".format("title"), fontsize=20)
    ax.legend(fontsize=18)

    if savefig is not None:
        fig.savefig(savefig + ".png", bbox_inches="tight", transparent=True, format="png")
    else:
        plt.show()

    return dfplot


def tsne_plots(tsne_embedding, df, savefig=None):
    df.loc[:, "X"] = tsne_embedding[:, 0]
    df.loc[:, "Y"] = tsne_embedding[:, 1]

    fig = plt.figure(figsize=(20, 10))
    plt.margins(0.05)  # Optional, just adds 5% padding to the autoscaling
    df = df.dropna()

    classes = []
    for c in df.event_labels.unique():
        classes.extend(c.split(","))
    classes = list(set(classes))

    full_df = pd.DataFrame()
    for i, c in enumerate(classes):
        # if not c == "Speech":
        df1 = df[df.event_labels.str.contains(c)]
        df1.loc[:, "X"] += 0.5 * np.random.rand(1)[0]
        df1.loc[:, "Y"] += 0.2 * np.random.rand(1)[0]
        df1.loc[:, "label"] = c
        full_df = full_df.append(df1, ignore_index=True)

        plt.scatter(df1.X, df1.Y, marker='o', label=c)
        plt.setp(fig.gca().get_xticklabels(), visible=False)
        plt.setp(fig.gca().get_yticklabels(), visible=False)
    # ax.set_title("Model {}".format("title"), fontsize=20)
    fig.legend()
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
    #           ncol=len(classes) // 2, fancybox=True, shadow=True, fontsize=18)

    if savefig is not None:
        fig.savefig(savefig + ".png", bbox_inches="tight", transparent=True, format="png")
        plt.close(fig)
    else:
        plt.show()

    return full_df


def scatter_ratio(embed, df):
    classes = ['Vacuum_cleaner', 'Frying', 'Cat', 'Alarm_bell_ringing', 'Running_water', 'Speech',
               'Electric_shaver_toothbrush', 'Blender', 'Dog', 'Dishes']
    vector_embed = embed.reshape(embed.shape[0], -1)

    s_within = np.zeros((vector_embed.shape[-1], vector_embed.shape[-1]))
    s_between = np.zeros((vector_embed.shape[-1], vector_embed.shape[-1]))
    full_mean = np.mean(vector_embed, axis=0)
    for c in classes:
        class_df = df[df.event_labels.fillna("").str.contains(c)]
        if not class_df.empty:
            class_embed = vector_embed[class_df.index]
            mean_class = np.mean(class_embed, axis=0)
            s_within += np.dot((class_embed - mean_class).T, (class_embed - mean_class))
            s_between += np.dot((mean_class - full_mean).T, (mean_class - full_mean))
    return np.trace(s_within + s_between) / np.trace(s_within)


def proto_acc(embed, df):
    classes = ['Alarm_bell_ringing', 'Blender', 'Cat', 'Dishes', 'Dog', 'Electric_shaver_toothbrush', 'Frying',
               'Running_water', 'Speech', 'Vacuum_cleaner']
    vector_embed = embed.reshape(embed.shape[0], -1)
    classes_mean = np.zeros((10, embed.shape[-1]))
    for i, c in enumerate(classes):
        class_df = df[df.event_labels.fillna("").str.contains(c)]
        if not class_df.empty:
            class_embed = vector_embed[class_df.index]
            mean_class = np.mean(class_embed, axis=0)
            classes_mean[i] = mean_class

    acc_per_class = np.zeros((len(classes)))
    for i, c in enumerate(classes):
        class_df = df[df.event_labels.fillna("").str.contains(c)]
        if not class_df.empty:
            class_embed = vector_embed[class_df.index]
            distance_to_min = scipy.spatial.distance.cdist(class_embed, classes_mean)
            labels = distance_to_min.argmin(-1)
            acc_per_class[i] = (labels == i).mean()

    LOG.info(pd.DataFrame([classes, acc_per_class.tolist()]).transpose())
    return acc_per_class.mean()


if __name__ == '__main__':
    import config as cfg
    from DataLoad import DataLoadDf
    from utils.Transforms import ApplyLog, Unsqueeze, PadOrTrunc, ToTensor, Normalize, Compose
    from utils.utils import load_model, ManyHotEncoder

    # ###########
    # ## Argument
    # ###########
    t = time.time()
    print("Arguments have been set for a certain group of experiments, feel free to change it.")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--subpart_data', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--embed_name', type=str, default=None)
    # Experiences to compare the impact of number of labaled vs unlabeled triplets
    # Be careful if subpart data is not None!!!!!!
    f_args = parser.parse_args()
    LOG.info(pformat(vars(f_args)))
    model_path = f_args.model_path
    assert model_path is not None, "model_path has to be defined to compute an embedding"
    embed_name = f_args.embed_name
    if embed_name is None:
        embed_name = model_path.split("/")[-2]
    ############
    #  Parameters experiences
    ###########
    subpart_data = f_args.subpart_data
    dataset = DesedSynthetic("../dcase2019",
                             base_feature_dir="../dcase2019/features",
                             save_log_feature=False)
    emb_model, state = load_model(model_path, return_state=True)
    epoch_model = state["epoch"]
    LOG.info("model loaded at epoch: {}".format(epoch_model))
    if torch.cuda.is_available():
        emb_model = emb_model.cuda()
    emb_model.eval()

    many_hot_encoder = ManyHotEncoder.load_state_dict(state['many_hot_encoder'])
    encode_function_label = many_hot_encoder.encode_weak
    scaler = ScalerSum.load_state_dict(state['scaler'])

    frames_in_sec = cfg.frames_in_sec

    transf = Compose([ApplyLog(), PadOrTrunc(nb_frames=cfg.frames), ToTensor(), Unsqueeze(0),
                      Normalize(scaler), Unsqueeze(1)])
    test_fr = dataset.get_df_feat_dir(cfg.test2018, frames_in_sec=frames_in_sec, subpart_data=subpart_data)
    print(len(test_fr))

    test_dataset = DataLoadDf(test_fr, many_hot_encoder.encode_weak, transform=transf)

    embed_set = "embedding"
    embed_dir = "stored_data/embeddings"
    embed_dir = os.path.join(embed_dir, embed_name, "embeddings")
    create_folder(embed_dir)
    fig_dir = os.path.join(embed_dir, "figures")
    create_folder(fig_dir)

    df_emb, embeddings = calculate_embedding(test_dataset, emb_model,
                                             savedir=os.path.join(embed_dir, embed_set), concatenate="append")
    print(embeddings.mean())
    print(embeddings.var())
    embeddings = sklearn.preprocessing.StandardScaler().fit_transform(embeddings.reshape(embeddings.shape[0], -1))
    print("normalized")
    print(embeddings.mean())
    print(embeddings.var())
    df_emb = df_emb.fillna("")
    tsne = TSNE()
    tsne_emb = tsne.fit_transform(X=embeddings.reshape(embeddings.shape[0], -1))
    tsne_plots(tsne_emb, df_emb, savefig=os.path.join(fig_dir, embed_set))
    scater_valid_rat = scatter_ratio(embeddings.reshape(embeddings.shape[0], -1), df_emb.reset_index())
    silhouette_valid_score = sklearn.metrics.silhouette_score(
        embeddings.reshape(embeddings.shape[0], -1), df_emb.event_labels, metric='euclidean')
    LOG.info("Valid silhouette for all classes in 2D (tsne) : {}".format(
        sklearn.metrics.silhouette_score(df_emb[["X", "Y"]], df_emb.event_labels, metric='euclidean')))

    embed_dir = "stored_data/embeddings"
    embed_dir = os.path.join(embed_dir, embed_name)
    create_folder(embed_dir)
    np.save(os.path.join(embed_dir, "embed" + str(epoch_model)), embeddings)
    test_fr.to_csv(os.path.join(embed_dir, "df" + str(epoch_model)), sep="\t", index=False)
