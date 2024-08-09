import collections
import json
import logging
import os

import datasets
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import model_selection
import torch
import transformers

import common
import models


LOG = logging.getLogger(__name__)

SAVE_STRATEGIES = [s.value for s in transformers.IntervalStrategy]

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 30
DEFAULT_SAVE_STRATEGY = transformers.IntervalStrategy.NO

LR = 1e-5
SEED = 42


def load_data(data_path, chain, positive_labels):
    dat = pd.read_parquet(data_path)
    X = dat.loc[:, chain]
    y_groups = dat.subject.values
    y = np.isin(dat.label.values, positive_labels).astype(int)
    return X, y, y_groups


def _compute_metrics(p):
    predictions, labs = p

    probs = torch.softmax(torch.from_numpy(predictions),
                          dim=1).detach().numpy()[:, -1]
    # We run an argmax to get the label
    preds = np.argmax(predictions, axis=1)

    return {
        "precision": metrics.precision_score(labs, preds, pos_label=1),
        "recall": metrics.recall_score(labs, preds, pos_label=1),
        "f1": metrics.f1_score(labs, preds, pos_label=1, average="weighted"),
        "apr": metrics.average_precision_score(labs, probs, pos_label=1),
        "balanced_accuracy": metrics.balanced_accuracy_score(labs, preds),
        "auc": metrics.roc_auc_score(labs, probs),
        "mcc": metrics.matthews_corrcoef(labs, preds),
    }


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


def train(processed_data, model_name, model_path,
          use_default_model_tokenizer, frozen_layers,
          output_model_path, batch_size, epochs, save_strategy):

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model()

    device = common.get_best_device()
    model = model.to(device)

    model_loader.freeze_weights(model, frozen_layers)

    model_size = sum(p.numel() for p in model.parameters())
    LOG.info(f"Model size: {model_size / 1e6:.2f}M")

    max_length = model_loader.get_max_length()

    def _preprocess(batch):
        formatted_seq_batch = []
        for seq in batch['sequence']:
            formatted_seq_batch.append(
                # TODO: handle HL and L cases
                model_loader.format_sequence(seq, None))
        t_inputs = tokenizer(formatted_seq_batch,
                             padding="max_length",
                             truncation=True,
                             max_length=max_length,
                             return_special_tokens_mask=True)
        batch['input_ids'] = t_inputs.input_ids
        batch['attention_mask'] = t_inputs.attention_mask
        return batch

    ab_dataset_tokenized = processed_data.map(
        _preprocess,
        batched=True,
        remove_columns=['sequence']
    )

    common.set_seed(SEED)

    LOG.info(f"Saving model to {output_model_path}")
    training_args = transformers.TrainingArguments(
        output_model_path,
        evaluation_strategy=save_strategy,
        save_strategy=save_strategy,
        logging_strategy='epoch',
        learning_rate=LR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        warmup_ratio=0,
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        lr_scheduler_type='linear',
        seed=SEED
    )

    trainer = transformers.Trainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=ab_dataset_tokenized['train'],
        eval_dataset=ab_dataset_tokenized['validation'],
        compute_metrics=_compute_metrics
    )

    trainer.train()

    if save_strategy == transformers.IntervalStrategy.NO:
        trainer.save_model()

    model.eval()
    outputs = trainer.predict(ab_dataset_tokenized["test"])
    LOG.info(outputs.metrics)

    out = trainer.state.log_history
    out.append(outputs.metrics)

    out_file = os.path.join(output_model_path, "model_test_stats.json")

    with open(out_file, 'w') as f:
        json.dump(out, f, indent=4)
