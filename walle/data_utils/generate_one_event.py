import numpy as np
import os
import json
import glob
import os.path as osp

from desed.utils import create_folder
from desed.post_process import rm_high_polyphony, post_process_txt_labels
from desed.generate_synthetic import SoundscapesGenerator


def choose_file(class_path):
    source_files = sorted(glob.glob(os.path.join(class_path, "*")))
    source_files = [f for f in source_files if os.path.isfile(f)]
    ind = np.random.randint(0, len(source_files))
    return source_files[ind]


def generate_training(n_soundscapes, fg_folder, bg_folder, param_file, outfolder, duration=10.0, ref_db=-50):
    create_folder(outfolder)

    with open(param_file) as json_file:
        params = json.load(json_file)

    sg = SoundscapesGenerator(duration, fg_folder, bg_folder, ref_db=ref_db)
    sg.generate_by_label_occurence(params, n_soundscapes, outfolder,
                                   min_events=1, max_events=1, pitch_shift=('uniform', -3, 3))


if __name__ == '__main__':
    base_soundbank = osp.join("..", "..", "synthetic")

    for subset in ["train", "eval"]:
        soundbank_path = osp.join(base_soundbank, "audio", subset, "soundbank")
        fg_folder = osp.join(soundbank_path, "foreground/")
        bg_folder = osp.join(soundbank_path, "background")

        dataset_path = osp.join("..", "..", "dataset")
        param_file = osp.join(dataset_path, "metadata", subset, f"event_occurences_{subset}.json")
        outfolder = osp.join(dataset_path, "audio", subset, "one_event_generated")
        out_tsv = osp.join(dataset_path, "audio", subset, "one_event_generated.tsv")

        generate_training(15, fg_folder, bg_folder, param_file, outfolder)
        rm_high_polyphony(outfolder, 3)
        post_process_txt_labels(outfolder, output_folder=outfolder, output_tsv=out_tsv)
