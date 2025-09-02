import collections
import logging

import numpy as np
import pandas as pd
from sklearn import model_selection

import common

LOG = logging.getLogger(__name__)


def load_data(data_path, positive_labels):
    dat = pd.read_parquet(data_path)
    dat.reset_index(inplace=True)
    X = dat.loc[:, [common.CHAIN_H, common.CHAIN_L]]
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
    X_train, X_test = X.loc[train_index], X.loc[test_index]
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

    train = X_inner_train.copy()
    train[common.LABEL_COL_NAME] = y_inner_train
    val = X_val.copy()
    val[common.LABEL_COL_NAME] = y_val
    test = X_test.copy()
    test[common.LABEL_COL_NAME] = y_test

    LOG.info(f'Train data size: {train.shape[0]}')
    LOG.info(f'Validation data size: {val.shape[0]}')

    train[common.DATASET_COL_NAME] = common.TRAIN
    val[common.DATASET_COL_NAME] = common.VALIDATION
    test[common.DATASET_COL_NAME] = common.TEST

    combined_data = pd.concat([train, val, test], ignore_index=True)
    # TODO(alexpilotti): avoid introducing duplicates in inner_train_index
    combined_data = combined_data.drop_duplicates(
        subset=[common.CHAIN_H, common.CHAIN_L])
    return combined_data


def save_data(data, output_path):
    LOG.info(f"Saving data to \"{output_path}\"")
    data.to_parquet(output_path)
