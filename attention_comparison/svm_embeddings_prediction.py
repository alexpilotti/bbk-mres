import collections
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
from sklearn import svm
import torch

import common


LOG = logging.getLogger(__name__)


def _load_build_model_data(data_path, embeddings_path, positive_labels):
    y = pd.read_parquet(data_path)
    X = torch.load(embeddings_path, map_location=torch.device('cpu'),
                   weights_only=False).numpy()
    y_groups = y.subject.values
    y = np.isin(y.label.values.astype(str), positive_labels).astype(int)
    assert X.shape[0] == len(y)
    return X, y, y_groups


def _load_predict_data(data_path, embeddings_path):
    df = pd.read_parquet(data_path)
    y_test = df[common.LABEL_COL_NAME].to_numpy()
    X_test = torch.load(embeddings_path,
                        map_location=torch.device('cpu'),
                        weights_only=False).numpy()
    assert len(X_test) == len(y_test)
    return X_test, y_test


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


def build_model(data_path, embeddings_path, positive_labels, output_model_path,
                output_score_path):
    X, y, y_groups = _load_build_model_data(data_path, embeddings_path,
                                            positive_labels)
    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")
    LOG.info(f"In total, {len(y)} sequences from {len(np.unique(y_groups))} "
             f"donors/studies.")
    LOG.info(f"Class size: {collections.Counter(np.sort(y)).most_common()}")

    n_splits_outer = 4
    n_splits_inner = 3

    outer_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_outer, shuffle=True, random_state=9)
    inner_cv = model_selection.StratifiedGroupKFold(
        n_splits=n_splits_inner, shuffle=True, random_state=1)

    svc = svm.SVC(kernel="rbf", class_weight="balanced",
                  probability=True)
    rank = np.zeros((n_splits_outer, 4))
    p_grid = {"C": [1e-2, 1e-1, 10, 100]}
    outer_cv_w_groups = outer_cv.split(X, y, y_groups)
    metrics = []

    i = 1
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

    if output_score_path:
        out_score = pd.DataFrame(metrics)
        out_score.to_csv(output_score_path)

    best_rank = np.argmin(rank.sum(axis=0))
    best_params = list(p_grid.values())[0][best_rank]
    LOG.info(f"Best hyperparameters: {best_params}")
    best_model = svm.SVC(kernel="rbf", class_weight="balanced",
                         probability=True, C=best_params)
    # Use the entire dataset
    best_model.fit(X, y)

    LOG.info(f"Saving the best model: {output_model_path}")
    joblib.dump(best_model, output_model_path)


def predict(model_path, data_path, embeddings_path, output_metrics_path):
    model = joblib.load(model_path)

    X_test, y_test = _load_predict_data(data_path, embeddings_path)

    prediction = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, -1]
    metrics = _compute_metrics(prediction, prob, y_test)

    LOG.info(f"Saving SVM prediction metrics to: {output_metrics_path}")
    common.save_json_file(metrics, output_metrics_path)
