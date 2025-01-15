import itertools
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

LR = 1e-5


def _compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for pred, lab in zip(prediction, label):
            if lab != -100:  # Exclude special tokens
                true_predictions.append(pred)
                true_labels.append(lab)

    report = metrics.classification_report(
        y_true=true_labels,
        y_pred=true_predictions,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    return report


def _prepare_dataset(model_loader, tokenizer, data, chain, ds_names):
    data["sequence"] = data[f"sequence_{chain}"]
    data["labels"] = data[f"labels_{chain}"]

    data["sequence"] = data["sequence"].apply(
        lambda x: model_loader.format_sequence(
                x if chain == 'H' else None,
                x if chain == 'L' else None))

    ds = datasets.DatasetDict()
    for ds_name in ds_names:
        df = data[data[common.DATASET_COL_NAME] == ds_name].drop(
                columns=[c for c in data.columns if c not in
                         ["sequence", "labels"]])
        df["labels"] = df["labels"].apply(
            lambda x: [int(x) for x in x.split(',')])
        ds[ds_name] = datasets.Dataset.from_pandas(df)

    def _tokenize_and_align_labels(batch_data):
        max_length = model_loader.get_max_length()

        tokenized_inputs = tokenizer(
            batch_data["sequence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            is_split_into_words=False,
            return_tensors="pt"
        )

        aligned_labels_batch = []
        for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            labels = batch_data["labels"][i]
            aligned_labels = []
            for token in tokens:
                if token in tokenizer.all_special_tokens:
                    aligned_labels.append(-100)
                else:
                    aligned_labels.append(labels.pop(0))
            aligned_labels_batch.append(aligned_labels)

        tokenized_inputs["labels"] = aligned_labels_batch
        return tokenized_inputs

    return ds.map(_tokenize_and_align_labels, batched=True)


def train(data, chain, model_name, model_path, use_default_model_tokenizer,
          frozen_layers, output_model_path, batch_size, epochs, save_strategy):
    common.set_seed()

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model_for_token_classification()

    model_loader.freeze_weights(model, frozen_layers)

    model_size = sum(p.numel() for p in model.parameters())
    LOG.info(f"Model size: {model_size / 1e6:.2f}M")

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
        seed=common.DEFAULT_SEED,
        fp16=True,
        auto_find_batch_size=True
    )

    tokenized_dataset = _prepare_dataset(
        model_loader, tokenizer, data, chain, [common.TRAIN])
    data_collator = transformers.DataCollatorForTokenClassification(tokenizer)

    trainer = transformers.Trainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset[common.TRAIN],
        data_collator=data_collator,
        compute_metrics=_compute_metrics
    )

    trainer.train()

    if save_strategy == transformers.IntervalStrategy.NO:
        trainer.save_model()


def predict_metrics(data, chain, model_name, model_path,
                    use_default_model_tokenizer):
    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model_for_token_classification()

    trainer = transformers.Trainer(
        model,
        tokenizer=tokenizer
    )

    tokenized_dataset = _prepare_dataset(
        model_loader, tokenizer, data, chain, [common.TEST])

    model.eval()
    outputs = trainer.predict(tokenized_dataset[common.TEST])
    metrics = _compute_metrics((outputs.predictions, outputs.label_ids))
    probs = torch.softmax(
        torch.from_numpy(outputs.predictions), dim=1).detach().numpy()[:, -1]

    print(f"Predicted the binding probability of {probs.shape[0]} sequences")

    metrics['model_name'] = model_name
    metrics['model_path'] = model_path
    metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
    metrics['test_loss'] = outputs.metrics['test_loss']

    return metrics


def predict_labels(data, chain, model_name, model_path,
                   use_default_model_tokenizer):
    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model_for_token_classification()

    data = data[data["dataset"] == common.TEST]

    data["sequence"] = data[f"sequence_{chain}"]
    data["labels"] = data[f"labels_{chain}"]

    all_predicted_labels = []
    for row in data.itertuples():
        seq = model_loader.format_sequence(
            row.sequence_H if 'H' in chain else None,
            row.sequence_L if 'L' in chain else None)

        inputs = tokenizer(seq, return_tensors="pt",
                           is_split_into_words=False)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        predicted_labels = logits.argmax(dim=2).tolist()[0]

        predicted_labels_non_special_tokens = []
        for i, token in enumerate(tokens):
            if "unk" in token.lower() or token == "X":
                predicted_labels_non_special_tokens.append(-100)
            elif token not in tokenizer.all_special_tokens:
                predicted_labels_non_special_tokens.append(predicted_labels[i])

        LOG.info("Expected labels / Predicted labels: "
                 f"{row.labels} {predicted_labels_non_special_tokens}")

        assert len(row.sequence) == len(predicted_labels_non_special_tokens)

        all_predicted_labels.append(
            ",".join(map(str, predicted_labels_non_special_tokens)))

    data["predicted_labels"] = all_predicted_labels
    return data


def load_data(data_path):
    return pd.read_parquet(data_path)
