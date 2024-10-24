from itertools import zip_longest
import logging
import os

import datasets
import numpy as np
import pandas as pd
from sklearn import metrics
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
        "specificity": common.specificity(labs, preds),
    }


def load_data(data_path):
    data = pd.read_parquet(data_path)
    data = data[[common.CHAIN_H, common.CHAIN_L, common.LABEL_COL_NAME,
                 common.DATASET_COL_NAME]]

    ab_dataset = datasets.DatasetDict()
    for ds in [common.TRAIN, common.VALIDATION, common.TEST]:
        df = data[
            data[common.DATASET_COL_NAME] == ds].drop(
                columns=[common.DATASET_COL_NAME])
        ab_dataset[ds] = datasets.Dataset.from_pandas(df)

    class_label = datasets.ClassLabel(2, names=[0, 1])
    return ab_dataset.map(
        lambda chain_h, chain_l, labels: {
            common.CHAIN_H: chain_h,
            common.CHAIN_L: chain_l,
            common.LABEL_COL_NAME: class_label.str2int(labels)
        },
        input_columns=[common.CHAIN_H, common.CHAIN_L,
                       common.LABEL_COL_NAME], batched=True
    )


def _get_dataset_tokenized(data, chain, tokenizer, model_loader):
    max_length = model_loader.get_max_length()

    def _preprocess(batch):
        formatted_seq_batch = []

        for chain_h, chain_l in zip_longest(
                *[batch[c] if chain in [c, common.CHAIN_HL] else []
                  for c in [common.CHAIN_H, common.CHAIN_L]]):
            formatted_seq_batch.append(
                model_loader.format_sequence(chain_h, chain_l))

        t_inputs = tokenizer(formatted_seq_batch,
                             padding="max_length",
                             truncation=True,
                             max_length=max_length,
                             return_special_tokens_mask=True)
        batch['input_ids'] = t_inputs.input_ids
        batch['attention_mask'] = t_inputs.attention_mask
        return batch

    return data.map(
        _preprocess,
        batched=True,
        remove_columns=[common.CHAIN_H, common.CHAIN_L]
    )


def train(data, chain, model_name, model_path, use_default_model_tokenizer,
          frozen_layers, output_model_path, batch_size, epochs, save_strategy):
    common.set_seed(SEED)

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model()

    device = common.get_best_device()
    LOG.info(f"Using device: {device}")
    model = model.to(device)

    model_loader.freeze_weights(model, frozen_layers)

    model_size = sum(p.numel() for p in model.parameters())
    LOG.info(f"Model size: {model_size / 1e6:.2f}M")

    ab_dataset_tokenized = _get_dataset_tokenized(
        data, chain, tokenizer, model_loader)

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
        train_dataset=ab_dataset_tokenized[common.TRAIN],
        eval_dataset=ab_dataset_tokenized[common.VALIDATION],
        compute_metrics=_compute_metrics
    )

    trainer.train()

    if save_strategy == transformers.IntervalStrategy.NO:
        trainer.save_model()

    model.eval()
    outputs = trainer.predict(ab_dataset_tokenized[common.TEST])
    LOG.info(outputs.metrics)

    out = trainer.state.log_history
    out.append(outputs.metrics)

    out_file = os.path.join(output_model_path, "model_test_stats.json")
    common.save_json_file(out, out_file)


def predict(data, chain, model_name, model_path, use_default_model_tokenizer):
    common.set_seed(SEED)

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model()

    device = common.get_best_device()
    LOG.info(f"Using device: {device}")
    model = model.to(device)

    ab_dataset_tokenized = _get_dataset_tokenized(
        data, chain, tokenizer, model_loader)

    trainer = transformers.Trainer(
        model,
        tokenizer=tokenizer
    )

    model.eval()
    outputs = trainer.predict(ab_dataset_tokenized[common.TEST])
    metrics = _compute_metrics((outputs.predictions, outputs.label_ids))
    probs = torch.softmax(
        torch.from_numpy(outputs.predictions), dim=1).detach().numpy()[:, -1]

    print(f"Predicted the binding probability of {probs.shape[0]} sequences")

    metrics['model_name'] = model_name
    metrics['model_path'] = model_path
    metrics['test_loss'] = outputs.metrics['test_loss']

    return probs, metrics
