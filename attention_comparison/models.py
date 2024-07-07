import transformers


MODEL_BALM_PAIRED = "BALM-paired"
MODEL_ANTIBERTY = "AntiBERTy"
MODEL_ANTIBERTA2 = "AntiBERTa2"
MODEL_ESM2_650M = "ESM2-650M"
MODEL_ESM2_150M = "ESM2-150M"
MODEL_ESM2_35M = "ESM2-35M"
MODEL_ESM2_8M = "ESM2-8M"
MODEL_ESM2_FT = "ESM2-650M-FT"

MODELS = [
    MODEL_BALM_PAIRED,
    MODEL_ANTIBERTY,
    MODEL_ANTIBERTA2,
    MODEL_ESM2_650M,
    MODEL_ESM2_150M,
    MODEL_ESM2_35M,
    MODEL_ESM2_8M,
    MODEL_ESM2_FT,
]

_DEFAULT_MODEL_HUB_PATHS = {
    MODEL_ESM2_650M: "facebook/esm2_t33_650M_UR50D",
    MODEL_ESM2_150M: "facebook/esm2_t30_150M_UR50D",
    MODEL_ESM2_35M: "facebook/esm2_t12_35M_UR50D",
    MODEL_ESM2_8M: "facebook/esm2_t6_8M_UR50D",
    MODEL_ANTIBERTA2: "alchemab/antiberta2",
}

# Note(alexpilotti): doubling the length due to the additional spaces
_ANTIBERTA2_MAX_LENGTH = 256 * 2
_ANTIBERTY_MAX_LENGTH = (512 - 2) * 2
_BALM_MAX_LENGTH = 512 - 2
# Note(alexpilotti): didn't find any resource validating a max length for ESM2,
# expect discussions suggesting to limit it to 1024
_ESM2_MAX_LENGTH = 512


class BaseModelLoader:
    def check_model_name(model_name):
        return False

    def __init__(self, model_path):
        self._model_path = model_path
        self._cls_token = None

    def load_model(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._model_path)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForSequenceClassification.\
            from_pretrained(self._model_path, num_labels=2)
        return model, tokenizer

    def format_sequence(self, chain_h, chain_l):
        if chain_h and chain_l:
            return f"{chain_h}{self._cls_token}{self._cls_token}{chain_l}"
        else:
            return chain_h or chain_l


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
            MODEL_ESM2_650M,
            MODEL_ESM2_150M,
            MODEL_ESM2_35M,
            MODEL_ESM2_8M]

    def format_sequence(self, chain_h, chain_l):
        return super().format_sequence(
            chain_h, chain_l)[:_ESM2_MAX_LENGTH]


class AntiBERTa2ModelLoader(BaseBERTModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_ANTIBERTA2

    def format_sequence(self, chain_h, chain_l):
        return super().format_sequence(
            chain_h, chain_l)[:_ANTIBERTA2_MAX_LENGTH]


class AntiBERTyModelLoader(BaseBERTModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_ANTIBERTY

    def load_model(self):
        tokenizer = transformers.BertTokenizer(
            vocab_file=self._model_path, do_lower_case=False)
        self._cls_token = tokenizer.cls_token
        model = transformers.AutoModelForSequenceClassification.\
            from_pretrained(self._model_path, num_labels=2)
        return model, tokenizer

    def format_sequence(self, chain_h, chain_l):
        return super().format_sequence(
            chain_h, chain_l)[:_ANTIBERTY_MAX_LENGTH]


class BALMPairedModelLoader(BaseModelLoader):
    def check_model_name(model_name):
        return model_name == MODEL_BALM_PAIRED

    def __init__(self, model_path):
        if not model_path:
            raise Exception(
                "BALM requires a local model path, e.g. "
                "\"BALM-paired_LC-coherence_90-5-5-split_122222\"")
        super().__init__(model_path)

    def format_sequence(self, chain_h, chain_l):
        return super().format_sequence(
            chain_h, chain_l)[:_BALM_MAX_LENGTH]


def load_model(model_name, model_path):
    MODEL_LOADERS = [ESM2ModelLoader,
                     AntiBERTa2ModelLoader,
                     BALMPairedModelLoader]

    for model_loader_class in MODEL_LOADERS:
        if model_loader_class.check_model_name(model_name):
            model_loader = model_loader_class(
                model_path or _DEFAULT_MODEL_HUB_PATHS[model_name])
            return model_loader.load_model() + (model_loader.format_sequence, )

    raise Exception(f"Unknown model: {model_name}")
