import logging

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
import torch

import collections


LOG = logging.getLogger(__name__)


def _load_data(data_path, embeddings_path, positive_labels):
    y = pd.read_parquet(data_path)
    X = torch.load(embeddings_path, map_location=torch.device('cpu')).numpy()
    y_groups = y.subject.values
    y = np.isin(y.label.values, positive_labels).astype(int)
    assert X.shape[0] == len(y)
    return X, y, y_groups


def _compute_metrics(preds, probs, labs):
    return {
        "precision": metrics.precision_score(labs, preds, pos_label=1),
        "recall": metrics.recall_score(labs, preds, pos_label=1),
        "f1": metrics.f1_score(labs, preds, pos_label=1, average="weighted"),
        "apr": metrics.average_precision_score(labs, probs, pos_label=1),
        "balanced_accuracy": metrics.balanced_accuracy_score(labs, preds),
        "auc": metrics.roc_auc_score(labs, probs),
        "mcc": metrics.matthews_corrcoef(labs, preds),
    }


def compute_prediction(data_path, embeddings_path, output_path,
                       positive_labels, shuffle=False, random_state=9):

    X, y, y_groups = _load_data(data_path, embeddings_path, positive_labels)
    y_group_counts = collections.Counter(y_groups)

    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")
    LOG.info(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} "
             f"donors/studies.")
    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")

    if shuffle:
        LOG.info(f"Shuffling the embedding...")
        y = y[np.random.permutation(len(y))]

    n_splits_outer = 4
    n_splits_inner = 3

    outer_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=random_state)
    inner_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=1)

    p_grid = {"C": [1e-2, 1e-1, 10, 100]}

    outer_cv_w_groups = outer_cv.split(X, y, y_groups)

    metrics = []

    i = 1
    svc = svm.SVC(kernel="rbf", class_weight="balanced",
                  probability=True)
    rank = np.zeros((n_splits_outer, 4))

    for train_index, test_index in outer_cv_w_groups:
        LOG.info(f"##### Outer fold {i} #####")
        # get the cross validation score on the test
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        LOG.info(f"Train size: {len(train_index)}, "
                 f"test size: {len(test_index)}")
        LOG.info(f"% positive train: {np.mean(y_train)}, "
                 f"% positive test: {np.mean(y_test)}")
        # inner loop

        search = model_selection.GridSearchCV(estimator=svc,
                                              param_grid=p_grid,
                                              cv=inner_cv, scoring="roc_auc",
                                              n_jobs=-1,
                                              pre_dispatch="1*n_jobs")
        search.fit(X_train, y_train, groups=y_groups[train_index])
        rank[i-1, :] = search.cv_results_['rank_test_score']

        prediction = search.predict(X_test)
        prob = search.predict_proba(X_test)[:, -1]
        metric = _compute_metrics(prediction, prob, y_test)
        LOG.info(metric)
        metrics.append(metric)
        i += 1

    out_score = pd.DataFrame(metrics)
    out_score.to_csv(output_path)
