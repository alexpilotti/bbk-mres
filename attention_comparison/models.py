import abc
import logging
import os

import accelerate
import transformers

import model_embeddings

LOG = logging.getLogger(__name__)

MODEL_BALM_PAIRED = "BALM-paired"
MODEL_ANTIBERTY = "AntiBERTy"
MODEL_ANTIBERTA2 = "AntiBERTa2"
MODEL_ESM2_15B = "ESM2-15B"
MODEL_ESM2_3B = "ESM2-3B"
MODEL_ESM2_650M = "ESM2-650M"
MODEL_ESM2_150M = "ESM2-150M"
MODEL_ESM2_35M = "ESM2-35M"
MODEL_ESM2_8M = "ESM2-8M"

MODELS = [
    MODEL_BALM_PAIRED,
    MODEL_ANTIBERTY,
    MODEL_ANTIBERTA2,
    MODEL_ESM2_15B,
    MODEL_ESM2_3B,
    MODEL_ESM2_650M,
    MODEL_ESM2_150M,
    MODEL_ESM2_35M,
    MODEL_ESM2_8M,
]

_DEFAULT_MODEL_HUB_PATHS = {
    MODEL_ESM2_15B: "facebook/esm2_t48_15B_UR50D",
    MODEL_ESM2_3B: "facebook/esm2_t36_3B_UR50D",
    MODEL_ESM2_650M: "facebook/esm2_t33_650M_UR50D",
    MODEL_ESM2_150M: "facebook/esm2_t30_150M_UR50D",
    MODEL_ESM2_35M: "facebook/esm2_t12_35M_UR50D",
    MODEL_ESM2_8M: "facebook/esm2_t6_8M_UR50D",
    MODEL_ANTIBERTA2: "alchemab/antiberta2",
}

MODEL_TYPE_SEQUENCE_CLASSIFICATION = 1
MODEL_TYPE_MASKED_LM = 2

_ANTIBERTA2_MAX_LENGTH = 256
_ANTIBERTY_MAX_LENGTH = 512 - 2
_BALM_MAX_LENGTH = 512 - 2
# Note(alexpilotti): didn't find any resource validating a max length for ESM2,
# expect discussions suggesting to limit it to 1024
_ESM2_MAX_LENGTH = 512

_DEFAULT_NUM_FROZEN_LAYERS = 3


def accelerated(func):
    def wrapper(*args, **kwargs):
        model, tokenizer = func(*args, **kwargs)
        accelerator = accelerate.Accelerator()
        model, tokenizer = accelerator.prepare(model, tokenizer)
        LOG.info(f"Model device: {model.device}")
        return model, tokenizer
    return wrapper


class BaseModelLoader(metaclass=abc.ABCMeta):
    def check_model_name(model_name):
        return False

    def _get_default_model_path(self, model_name):
        try:
            return _DEFAULT_MODEL_HUB_PATHS[model_name]
        except KeyError:
            raise Exception(
                f"Default path not available for model: {model_name}")

    def _set_model_paths(self, model_name, model_path,
                         use_default_model_tokenizer):
        if not model_path or use_default_model_tokenizer:
            default_model_path = self._get_default_model_path(model_name)

        self._model_path = model_path or default_model_path
        self._tokenizer_path = (
            default_model_path if use_default_model_tokenizer
            else self._model_path)

    def __init__(self, model_name, model_path, use_default_model_tokenizer):
        self._set_model_paths(
            model_name, model_path, use_default_model_tokenizer)
        self._cls_token = None

    @accelerated
    def load_model_for_sequence_classification(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForSequenceClassification.\
            from_pretrained(self._model_path, num_labels=2)
        return model, tokenizer

    @accelerated
    def load_model_for_token_classification(self, num_labels=2):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForTokenClassification.\
            from_pretrained(self._model_path, num_labels=num_labels)
        return model, tokenizer

    @accelerated
    def load_model_for_embeddings(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            self._model_path)
        return model, tokenizer

    def format_sequence(self, chain_h, chain_l):
        if chain_h and chain_l:
            return f"{chain_h}{self._cls_token}{self._cls_token}{chain_l}"
        else:
            return chain_h or chain_l

    @abc.abstractmethod
    def _get_model_embeddings(self):
        pass

    def get_embeddings(self, formatted_sequences, device):
        return self._get_model_embeddings().get_embeddings(
            formatted_sequences, device)

    @abc.abstractmethod
    def get_max_length(self):
        pass

    @abc.abstractmethod
    def _get_bare_model(self, model):
        pass

    def freeze_weights(self, model, layers):
        bare_model = self._get_bare_model(model)
        for param in bare_model.embeddings.parameters():
            param.requires_grad = False

        if not layers:
            num_layers = len(bare_model.encoder.layer)
            # Pick all layers minus the last _DEFAULT_NUM_FROZEN_LAYERS
            layers = range(0, num_layers - _DEFAULT_NUM_FROZEN_LAYERS)

        for layer in [bare_model.encoder.layer[i] for i in layers]:
            for param in layer.parameters():
                param.requires_grad = False


class BaseBERTModelLoader(BaseModelLoader):
    def format_sequence(self, chain_h, chain_l):
        if chain_h and chain_l:
            return (f"{' '.join(chain_h)} {self._cls_token} "
                    f"{self._cls_token} {' '.join(chain_l)}")
        else:
            return ' '.join(chain_h or chain_l)


class ESM2ModelLoader(BaseModelLoader):
    def check_model_name(model_name):
        return model_name in [
            MODEL_ESM2_15B,
            MODEL_ESM2_3B,
            MODEL_ESM2_650M,
            MODEL_ESM2_150M,
            MODEL_ESM2_35M,
            MODEL_ESM2_8M]

    def get_max_length(self):
        return _ESM2_MAX_LENGTH

    def _get_model_embeddings(self):
        return model_embeddings.ESM2Embeddings(self)

    def _get_bare_model(self, model):
        return model.esm


class AntiBERTa2ModelLoader(BaseBERTModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_ANTIBERTA2

    @accelerated
    def load_model_for_embeddings(self):
        tokenizer = transformers.RoFormerTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.RoFormerForMaskedLM.from_pretrained(
            self._model_path)
        return model, tokenizer

    def get_max_length(self):
        return _ANTIBERTA2_MAX_LENGTH

    def _get_model_embeddings(self):
        return model_embeddings.AntiBERTa2Embeddings(self)

    def _get_bare_model(self, model):
        return model.roformer


class AntiBERTyModelLoader(BaseBERTModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_ANTIBERTY

    def _set_model_paths(self, model_name, model_path,
                         use_default_model_tokenizer):
        if model_path:
            self._model_path = model_path
            self._vocab_txt_path = os.path.join(self._model_path, "vocab.txt")
        else:
            import antiberty
            antiberty_dir = os.path.dirname(
                os.path.realpath(antiberty.__file__))
            models_base_dir = os.path.join(antiberty_dir, "trained_models")
            self._model_path = os.path.join(
                models_base_dir, "AntiBERTy_md_smooth")
            self._vocab_txt_path = os.path.join(models_base_dir, "vocab.txt")

    @accelerated
    def load_model_for_sequence_classification(self):
        tokenizer = transformers.BertTokenizer(
            vocab_file=self._vocab_txt_path, do_lower_case=False)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForSequenceClassification.\
            from_pretrained(self._model_path, num_labels=2)
        return model, tokenizer

    @accelerated
    def load_model_for_token_classification(self, num_labels=2):
        tokenizer = transformers.BertTokenizer(
            vocab_file=self._vocab_txt_path, do_lower_case=False)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForTokenClassification.\
            from_pretrained(self._model_path, num_labels=num_labels)
        return model, tokenizer

    @accelerated
    def load_model_for_embeddings(self):
        return self.load_model_for_sequence_classification()

    def get_max_length(self):
        return _ANTIBERTY_MAX_LENGTH

    def _get_model_embeddings(self):
        return model_embeddings.AntiBERTyEmbeddings(self)

    def _get_bare_model(self, model):
        return model.bert


class BALMPairedModelLoader(BaseModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_BALM_PAIRED

    def _set_model_paths(self, model_name, model_path,
                         use_default_model_tokenizer):
        if not model_path:
            raise Exception(
                "BALM requires a local model path, e.g. "
                "\"BALM-paired_LC-coherence_90-5-5-split_122222\"")
        self._model_path = model_path
        self._tokenizer_path = self._model_path

    @accelerated
    def load_model_for_embeddings(self):
        tokenizer = transformers.RobertaTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.RobertaForMaskedLM.from_pretrained(
            self._model_path)
        return model, tokenizer

    @accelerated
    def load_model_for_token_classification(self, num_labels=2):
        tokenizer = transformers.RobertaTokenizer.from_pretrained(
            self._tokenizer_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.RobertaForTokenClassification.from_pretrained(
            self._model_path)
        return model, tokenizer

    def get_max_length(self):
        return _BALM_MAX_LENGTH

    def _get_model_embeddings(self):
        return model_embeddings.BALMPairedEmbeddings(self)

    def _get_bare_model(self, model):
        return model.roberta


def get_model_loader(model_name, model_path, use_default_model_tokenizer):
    MODEL_LOADERS = [AntiBERTa2ModelLoader,
                     AntiBERTyModelLoader,
                     BALMPairedModelLoader,
                     ESM2ModelLoader]

    for model_loader_class in MODEL_LOADERS:
        if model_loader_class.check_model_name(model_name):
            model_loader = model_loader_class(
                model_name, model_path, use_default_model_tokenizer)
            return model_loader

    raise Exception(f"Unknown model: {model_name}")
