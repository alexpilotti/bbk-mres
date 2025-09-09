import hashlib
import json
import os
import pathlib
import random

import numpy as np
from sklearn import metrics
import torch
import transformers


# IMGT numbering
# https://www.imgt.org/IMGTScientificChart/Nomenclature/IMGT-FRCDRdefinition.html
REGIONS = {
    "FR1": [1, 26],
    "CDR1": [27, 38],
    "FR2": [39, 55],
    "CDR2": [56, 65],
    "FR3": [66, 104],
    "CDR3": [105, 117],
    "FR4": [118, 129]
    }

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

_device = None


def set_device(device):
    global _device
    _device = device


def get_device():
    return _device


def get_best_device():
    if torch.cuda.is_available():
        device = "cuda"
        # LOCAL_RANK is set when using DistributedDataParallel
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank and int(local_rank) >= 0:
            device = f"{device}:{local_rank}"
    else:
        device = "cpu"

    return device


def set_seed(seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = seed is not None
    torch.backends.cudnn.benchmark = seed is None
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed if seed else 0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(seed is not None)
    transformers.set_seed(seed)


def save_json_file(data, path, indent=4):
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def specificity(y_true, y_pred):
    tn, fp, _, _ = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp)


def false_positive_rate(y_true, y_pred):
    tn, fp, _, _ = metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn)


def format_label_counts(data):
    counts = data[LABEL_COL_NAME].value_counts(normalize=True) * 100
    counts = counts.apply(lambda x: '{:,.2f} %'.format(x))
    return counts.to_string(header=False)


def _sha256_file(file_path, chunk_size=(1 << 20)):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _list_files(dir_path):
    for p in pathlib.Path(dir_path).rglob("*"):
        if p.is_file():
            yield p


def compute_files_hash(dir_path):
    for file_path in sorted(_list_files(dir_path)):
        yield (file_path.parts[-1], _sha256_file(file_path))


def get_training_args(output_model_path, save_strategy, batch_size, epochs,
                      learning_rate):
    return transformers.TrainingArguments(
        output_model_path,
        optim="adamw_torch_fused",
        evaluation_strategy=save_strategy,
        save_strategy=save_strategy,
        logging_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        lr_scheduler_type='linear',
        seed=DEFAULT_SEED,
        data_seed=DEFAULT_SEED,
        # Set to 0 for determinism concerns
        dataloader_num_workers=0,
        fp16=True,
        auto_find_batch_size=True,
        ddp_find_unused_parameters=False
    )


def get_predict_training_args():
    return transformers.TrainingArguments(
        # output_dir not needed by predict, but it has to be a valid path
        output_dir=os.getenv("TMPDIR", "/tmp"),
        seed=DEFAULT_SEED,
        data_seed=DEFAULT_SEED,
        # Set to 0 for determinism concerns
        dataloader_num_workers=0,
        # Needed due to a configuration mismatch error when using DeepSpeed
        fp16=True
    )
