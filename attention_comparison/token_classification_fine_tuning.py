import itertools
import logging
import os

import datasets
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from tqdm import tqdm
import transformers

import common
import models

LOG = logging.getLogger(__name__)

LR = 1e-5

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


def _compute_metrics(p):
    predictions, labels = p

    labels = torch.tensor(labels)
    probs = torch.softmax(torch.tensor(predictions), dim=-1)
    preds = torch.argmax(probs, dim=-1)

    mask = labels != -100
    probs_filtered = probs[mask][:, 1].numpy()
    preds_filtered = preds[mask].numpy()
    labs_filtered = labels[mask].numpy()

    report = metrics.classification_report(
        y_true=labs_filtered,
        y_pred=preds_filtered,
        zero_division=0,
        digits=4,
        output_dict=True
    )

    if len(set(labs_filtered)) > 1:
        # AUC is undefined if there's a single class
        auc = metrics.roc_auc_score(labs_filtered, probs_filtered)
    else:
        auc = 0

    report["apr"] = metrics.average_precision_score(
        labs_filtered, probs_filtered, pos_label=1)
    report["balanced_accuracy"] = metrics.balanced_accuracy_score(
        labs_filtered, preds_filtered)
    report["auc"] = auc
    report["mcc"] = metrics.matthews_corrcoef(labs_filtered, preds_filtered)
    report["fpr"] = common.false_positive_rate(labs_filtered, preds_filtered)
    report["f1_weighted"] = metrics.f1_score(
        labs_filtered, preds_filtered, average="weighted")

    return report


def _get_num_pos(pos):
    """Needed for positions with insertion codes, e.g. 110A."""
    return int("".join(c for c in pos if c.isdigit()))


def _filter_region_data(data, region):
    if region not in REGIONS:
        raise Exception(f'Invalid region name: "{region}". '
                        f'Valid options are: {", ".join(REGIONS.keys())}')

    region_range = range(*REGIONS[region])

    LOG.info(f"Retrieving residues in region {region}, "
             f"between positions: {region_range[0]}-{region_range[-1]}")

    data["positions_num"] = data["positions"].apply(
        lambda x: list(map(_get_num_pos, x.split(","))))

    data["sequence"] = data.apply(
        lambda row: "".join(
            row["sequence"][i] for i, pos in enumerate(row["positions_num"])
            if pos in region_range),
        axis=1)

    data["labels"] = data.apply(
        lambda row:
            [row["labels"][i] for i, pos in enumerate(row["positions_num"])
             if pos in region_range],
        axis=1)

    data["positions"] = data["positions"].apply(
        lambda x: ",".join(
            [pos for pos in x.split(",")
             if _get_num_pos(pos) in region_range]))

    return data


def _prepare_data(data, chain, region, ds_names):
    data = data[data[common.DATASET_COL_NAME].isin(ds_names)]

    data["sequence"] = data[f"sequence_{chain}"]
    data["labels"] = data[f"labels_{chain}"]
    data["positions"] = data[f"positions_{chain}"]

    data["labels"] = data["labels"].apply(
        lambda x: list(map(int, x.split(","))))

    if region:
        if chain == common.CHAIN_HL:
            raise Exception(
                "When specifying a region, the combined HL sequence is not "
                "supported")
        data = _filter_region_data(data, region)

    return data


def _prepare_dataset(model_loader, tokenizer, data, chain, region, ds_names):
    data = _prepare_data(data, chain, region, ds_names)

    if region:
        data["sequence"] = data["sequence"].apply(
            lambda x: model_loader.format_simple_sequence(x))
    else:
        data["sequence"] = data.apply(
            lambda row: model_loader.format_sequence(
                    row["sequence_H"] if common.CHAIN_H in chain else None,
                    row["sequence_L"] if common.CHAIN_L in chain else None),
            axis=1)

    ds = datasets.DatasetDict()
    for ds_name, ds_data in data.groupby(common.DATASET_COL_NAME):
        ds[ds_name] = datasets.Dataset.from_pandas(
            ds_data.drop(columns=[
                c for c in data.columns if c not in ["sequence", "labels"]]))

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


def train(data, chain, region, model_name, model_path,
          use_default_model_tokenizer, frozen_layers, output_model_path,
          batch_size, epochs, save_strategy):
    data = data.copy()
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
        auto_find_batch_size=True,
        ddp_find_unused_parameters=False
    )

    tokenized_dataset = _prepare_dataset(
        model_loader, tokenizer, data, chain, region, [common.TRAIN])
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


def predict_metrics(data, chain, region, model_name, model_path,
                    use_default_model_tokenizer):
    data = data.copy()

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model_for_token_classification()

    trainer = transformers.Trainer(
        model,
        tokenizer=tokenizer
    )

    tokenized_dataset = _prepare_dataset(
        model_loader, tokenizer, data, chain, region, [common.TEST])

    model.eval()
    outputs = trainer.predict(tokenized_dataset[common.TEST])
    metrics = _compute_metrics((outputs.predictions, outputs.label_ids))
    probs = torch.softmax(
        torch.from_numpy(outputs.predictions), dim=1).detach().numpy()[:, -1]

    LOG.info(
        f"Predicted the binding probability of {probs.shape[0]} sequences")

    metrics['model_name'] = model_name
    metrics['model_path'] = model_path
    metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
    metrics['test_loss'] = outputs.metrics['test_loss']

    return metrics


def predict_labels(data, chain, region, model_name, model_path,
                   use_default_model_tokenizer):
    data = data.copy()

    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)
    model, tokenizer = model_loader.load_model_for_token_classification()

    data = _prepare_data(data, chain, region, [common.TEST])
    data["labels"] = data["labels"].apply(lambda x: ",".join(map(str, x)))

    pbar = tqdm(total=len(data), desc="Processing labels")

    all_predicted_labels = []
    for row in data.itertuples():
        if region:
            seq = model_loader.format_simple_sequence(row.sequence)
        else:
            seq = model_loader.format_sequence(
                row.sequence_H if common.CHAIN_H in chain else None,
                row.sequence_L if common.CHAIN_L in chain else None)

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

        LOG.debug("Expected labels / Predicted labels: "
                  f"{row.labels} {predicted_labels_non_special_tokens}")
        assert len(row.sequence) == len(predicted_labels_non_special_tokens)

        all_predicted_labels.append(
            ",".join(map(str, predicted_labels_non_special_tokens)))
        pbar.update(1)

    data["predicted_labels"] = all_predicted_labels

    data = data.drop(
        columns=[c for c in data.columns if c not in
                 ["iden_code", "positions", "labels", "predicted_labels"]])

    return data


def load_data(data_path):
    return pd.read_parquet(data_path)
