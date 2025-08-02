import logging

import pandas as pd
import torch

import common
import models


LOG = logging.getLogger(__name__)


def get_sequences(data_path, chain, indexes=None):
    data = pd.read_parquet(data_path)
    if not indexes:
        indexes = data.index

    sequences = []
    for index in indexes:
        sequence = {}
        if chain in [common.CHAIN_H,  common.CHAIN_HL]:
            sequence[common.CHAIN_H] = data.loc[index, common.CHAIN_H]
        if chain in [common.CHAIN_L, common.CHAIN_HL]:
            sequence[common.CHAIN_L] = data.loc[index, common.CHAIN_L]
        sequences.append(sequence)
    return sequences


def get_embeddings(model_name, model_path, use_default_model_tokenizer,
                   sequences):
    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)

    LOG.info(f"Processing {len(sequences)} sequences")
    return model_loader.get_embeddings(sequences)


def save_embeddings(embeddings, output_path):
    LOG.info(f"Saving embeddings to \"{output_path}\"")
    torch.save(embeddings, output_path)
