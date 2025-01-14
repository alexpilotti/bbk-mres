import abc
import logging
import math

import antiberty
import torch

import common


ANTIBERTA2_BATCH_SIZE = 128
ANTIBERTY_BATCH_SIZE = 500
BLAM_PAIRED_BATCH_SIZE = 64
ESM2_BATCH_SIZE = 50

LOG = logging.getLogger(__name__)


def _batch_loader(sequences, batch_size):
    len_sequences = len(sequences)
    for i in range(0, len_sequences, batch_size):
        end_idx = min(i + batch_size, len_sequences)
        yield i, end_idx, sequences[i:end_idx]


class BaseEmbeddigs(metaclass=abc.ABCMeta):
    def __init__(self, model_loader):
        self._model_loader = model_loader

    @abc.abstractmethod
    def _get_batch_size(self):
        pass

    def _format_sequences(self, sequences):
        return [self._model_loader.format_sequence(
            s.get(common.CHAIN_H),
            s.get(common.CHAIN_L))
            for s in sequences]

    def get_embeddings(self, sequences, device):
        max_length = self._model_loader.get_max_length()
        batch_size = self._get_batch_size()
        num_batches = math.ceil(len(sequences) / batch_size)

        model, tokenizer = self._model_loader.load_model_for_embeddings()

        if device:
            model = model.to(device)

        formatted_seqs = self._format_sequences(sequences)

        embeddings = None
        for i, (start, end, batch) in enumerate(_batch_loader(formatted_seqs,
                                                              batch_size), 1):
            LOG.info(f"Processing batch {i} of {num_batches}")
            x = torch.tensor([
                tokenizer.encode(
                    seq,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_special_tokens_mask=True)
                for seq in batch]).to(device)
            attention_mask = (x != tokenizer.pad_token_id).float().to(device)
            with torch.no_grad():
                outputs = model(x, attention_mask=attention_mask,
                                output_hidden_states=True)
                outputs = outputs.hidden_states[-1]
                outputs = list(outputs.detach())
            for j, a in enumerate(attention_mask):
                outputs[j] = outputs[j][a == 1, :].mean(0)
            if embeddings is None:
                embeddings = torch.empty((len(sequences), len(outputs[0])))
            embeddings[start:end] = torch.stack(outputs)
            del x
            del attention_mask
            del outputs

        return embeddings


class _AntiBERTyCustomRunner(antiberty.AntiBERTyRunner):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device


class AntiBERTyEmbeddings(BaseEmbeddigs):
    def _get_batch_size(self):
        return ANTIBERTY_BATCH_SIZE

    def get_embeddings(self, sequences, device):
        batch_size = self._get_batch_size()
        num_batches = math.ceil(len(sequences) / batch_size)

        model, tokenizer = self._model_loader.load_model_for_embeddings()

        if device:
            model = model.to(device)

        antiberty_runner = _AntiBERTyCustomRunner(model, tokenizer, device)

        formatted_seqs = self._format_sequences(sequences)

        embeddings = None
        for i, (start, end, batch) in enumerate(_batch_loader(formatted_seqs,
                                                              batch_size), 1):
            LOG.info(f"Processing batch {i} of {num_batches}")
            x = antiberty_runner.embed(batch)
            x = [a.mean(axis=0) for a in x]
            if embeddings is None:
                embeddings = torch.empty((len(sequences), len(x[0])))
            embeddings[start:end] = torch.stack(x)

        return embeddings


class ESM2Embeddings(BaseEmbeddigs):
    def _get_batch_size(self):
        return ESM2_BATCH_SIZE


class AntiBERTa2Embeddings(ESM2Embeddings):
    def _get_batch_size(self):
        return ANTIBERTA2_BATCH_SIZE


class BALMPairedEmbeddings(BaseEmbeddigs):
    def _get_batch_size(self):
        return BLAM_PAIRED_BATCH_SIZE
