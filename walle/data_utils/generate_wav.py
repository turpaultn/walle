# -*- coding: utf-8 -*-
#########################################################################
# Initial software
# Copyright Nicolas Turpault, Romain Serizel, 2019, v1.0
# This software is distributed under the terms of the License GPL
#########################################################################
import time
import argparse
import os
import os.path as osp
import glob
import sys

from pprint import pformat
from desed.generate_synthetic import generate_files_from_jams

sys.path.append("..")
import config as cfg


if __name__ == "__main__":
    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-jams", action="store_true", default=False)
    args = parser.parse_args()
    pformat(vars(args))
    
    # Training
    dataset_root = cfg.relative_data_path
    soundbank_root = cfg.synthetic_soundbank

    meta_train_folder = osp.join(dataset_root, "metadata", "train", "one_event_train")
    meta_eval_folder = osp.join(dataset_root, "metadata", "eval")

    audio_train_folder = osp.join(dataset_root, "audio", "train", "one_event_train")
    audio_eval_folder = osp.join(dataset_root, "audio", "eval")

    list_jams = glob.glob(osp.join(meta_train_folder, "*.jams"))[:10]

    fg_path_train = osp.join(soundbank_root, "audio", "train", "soundbank", "foreground")
    bg_path_train = osp.join(soundbank_root, "audio", "train", "soundbank", "background")
    generate_files_from_jams(list_jams, audio_train_folder, fg_path=fg_path_train, bg_path=bg_path_train,
                             overwrite_jams=args.overwrite_jams)
    
    # Eval
    # In the evaluation part, there multiple subsets which allows to check robustness of systems
    list_folders = [osp.join(meta_eval_folder, dI) for dI in os.listdir(meta_eval_folder) if osp.isdir(osp.join(meta_eval_folder, dI))]
    fg_path_eval = osp.join(soundbank_root, "audio", "eval", "soundbank", "foreground")
    bg_path_eval = osp.join(soundbank_root, "audio", "eval", "soundbank", "background")
    
    for folder in list_folders:
        print(folder)
        list_jams = glob.glob(osp.join(folder, "*.jams"))[:10]
        generate_files_from_jams(list_jams, folder, fg_path=fg_path_eval, bg_path=bg_path_eval,
                                 overwrite_jams=args.overwrite_jams)
