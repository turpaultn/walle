import math
import os.path as osp
from utils.utils import create_folder

workspace = ""

absolute_dir_path = osp.abspath(osp.dirname(__file__))
relative_data_path = osp.join(absolute_dir_path, "..", "dataset")

weak = osp.join('metadata', 'train', 'weak.csv')
unlabel = osp.join('metadata', 'train', 'unlabel_in_domain.tsv')
synthetic = osp.join('metadata', 'train', 'synthetic.tsv')
validation = osp.join('metadata', 'validation', 'validation.tsv')
test2018 = osp.join('metadata', 'validation', 'test_dcase2018.tsv')
eval2018 = osp.join('metadata', 'validation', 'eval_dcase2018.tsv')

one_event_train = osp.join('metadata', 'train', 'one_event_train.tsv')
one_event_valid_list = osp.join('metadata', 'train', 'one_event_valid_list')
one_event = osp.join('metadata', 'eval', 'one_event_eval.tsv')
one_event_train200 = osp.join('metadata', 'train', 'one_event_train_200ms.tsv')
one_event200 = osp.join('metadata', 'eval', 'one_event_eval_200ms.tsv')

base_feature_dir = osp.join(relative_data_path, "features")

n_layers_RNN = 2

# config
# prepare_data
sample_rate = 16000
n_window = round(0.025 * sample_rate)  # 25ms aligned with google paper
hop_length = round(0.010 * sample_rate)  # aligned with google Audioset paper (10ms)
n_mels = 64  # 128
max_len_seconds = 10.
max_frames = math.ceil(max_len_seconds * sample_rate / hop_length)

frames_in_sec = .20
frames = int(frames_in_sec * sample_rate // hop_length)
if frames_in_sec is None:
    max_frames = max_frames
else:
    # This is useful so we can go back to a full segment file or at least a truncated part of it
    max_frames = max_frames - max_frames % frames

f_min = 125
f_max = 7500

# Main
number_test = 300
num_workers = 12
batch_size = 10
batch_size_classif = batch_size * 2
n_epoch_embedding = 500
n_epoch_classifier = 200

recompute_embedding = True
recompute_classif = True

early_stopping = 30
first_early_wait = 50
reduce_lr = 10
model_checkpoint = 1
save_best = True

conv_dropout = 0.2
dropout_non_recurrent = 0.2
activation = "leakyrelu"
batch_norm = False
attention = False

pool_freq = 2
last_layer = 32
nb_layers = 4

rampup_margin_length = None


def get_dirs(common_dir, create=True):
    """"""
    model_dir = osp.join("stored_data", "model", common_dir)
    log_dir = osp.join("stored_data", "logs", common_dir)
    if create:
        create_folder(model_dir)
        create_folder(log_dir)
    return model_dir, log_dir
