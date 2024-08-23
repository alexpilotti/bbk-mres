import collections
import logging

import datasets
import numpy as np
import pandas as pd
from sklearn import model_selection

LOG = logging.getLogger(__name__)


def load_data(data_path, chain, positive_labels):
    dat = pd.read_parquet(data_path)
    X = dat.loc[:, chain]
    y_groups = dat.subject.values
    y = np.isin(dat.label.values, positive_labels).astype(int)
    return X, y, y_groups


def process_data(data, fold_num):
    X, y, y_groups = data

    y_group_counts = collections.Counter(y_groups)
    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")
    LOG.info(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} "
             f"donors/studies.")
    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")

    n_splits_outer = 4
    n_splits_inner = 3
    random_state = 9
    outer_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    inner_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=1)
    outer_cv_groups = outer_cv.split(X, y, y_groups)
    i = 1

    while i <= fold_num:
        k, (train_index, test_index) = next(enumerate(outer_cv_groups))
        i += 1

    # for train_index, test_index in outer_cv_groups:
    LOG.info(f"##### Outer fold {i - 1} #####")
    # get the cross validation score on the test
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_groups_train = y_groups[train_index]
    LOG.info(f"Train size: {len(train_index)}, test size: {len(test_index)}")
    LOG.info(f"% positive train: {np.mean(y_train)}, "
             f"% positive test: {np.mean(y_test)}")

    # split validation data from the training data
    inner_cv_groups = inner_cv.split(X_train, y_train, y_groups_train)
    j, (inner_train_index, val_index) = next(enumerate(inner_cv_groups))
    X_inner_train, X_val = X.iloc[inner_train_index], X.iloc[val_index]
    y_inner_train, y_val = y[inner_train_index], y[val_index]
    train = pd.DataFrame(
        {'sequence': X_inner_train.values, 'labels': y_inner_train})
    val = pd.DataFrame({'sequence': X_val.values, 'labels': y_val})
    test = pd.DataFrame({'sequence': X_test.values, 'labels': y_test})
    LOG.info(f'Train data size: {train.shape[0]}')
    LOG.info(f'Validation data size: {val.shape[0]}')

    ab_dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_pandas(train),
        "validation": datasets.Dataset.from_pandas(val),
        "test": datasets.Dataset.from_pandas(test)
    })
    class_label = datasets.ClassLabel(2, names=[0, 1])
    return ab_dataset.map(
        lambda seq, labels: {
            "sequence": seq,
            "labels": class_label.str2int(labels)
        },
        input_columns=["sequence", "labels"], batched=True
    )
