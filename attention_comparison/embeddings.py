import logging

import torch

import models


LOG = logging.getLogger(__name__)


def get_embeddings(model_name, model_path, use_default_model_tokenizer,
                   sequences):
    model_loader = models.get_model_loader(
        model_name, model_path, use_default_model_tokenizer)

    LOG.info(f"Processing {len(sequences)} sequences")
    return model_loader.get_embeddings(sequences)


def save_embeddings(embeddings, output_path):
    LOG.info(f"Saving embeddings to \"{output_path}\"")
    torch.save(embeddings, output_path)
