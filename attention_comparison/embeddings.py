import torch

import models


def get_embeddings(model_name, model_path, sequences):
    model_loader = models.get_model_loader(
        model_name, model_path, False)

    return model_loader.get_embeddings(sequences)


def save_embeddings(embeddings, output_path):
    torch.save(embeddings, output_path)
