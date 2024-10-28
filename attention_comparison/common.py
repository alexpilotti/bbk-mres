import json
import os
import random

import numpy as np
from sklearn import metrics
import torch


CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"
CHAIN_TYPES = [CHAIN_H, CHAIN_L, CHAIN_HL]

TEST = "test"
TRAIN = "train"
VALIDATION = "validation"

DATASET_COL_NAME = "dataset"
LABEL_COL_NAME = "label"

DEFAULT_SEED = 42


def get_best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = seed is not None
    torch.backends.cudnn.benchmark = seed is None
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed if seed else 0)


def save_json_file(data, path, indent=4):
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def specificity(y_true, y_pred):
    tn, fp, _, _ = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def format_label_counts(data):
    counts = data[LABEL_COL_NAME].value_counts(normalize=True) * 100
    counts = counts.apply(lambda x: '{:,.2f} %'.format(x))
    return counts.to_string(header=False)
