# AbFlow: AbLM/PLM command line tools and reports

This repository contains a series of command line utils and reports used as
part of a research pipeline for reseacrh based on Antibody Language Models
(AbLMs) and Protein Languague Models (PLM). The tools can either be directly
invoked on the command line or used as part of a workflow pipeline such as
[AbFlow](https://github.com/alexpilotti/bbk_mres_airflow).

## Requirements

- Python 3.11 or later
- R 4.3 or later (needed only for the R reports)

## Supported models

Currently the following models are supported. Additional models can be easily
added, especially [Huggingface](https://huggingface.co/) based ones.

- [ESM-2](https://github.com/facebookresearch/esm) (8M, 35M, 150M, 650M, 3B,
  15B)
- [AntiBERTa2](https://huggingface.co/alchemab/antiberta2)
- [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy)
- [BALM](https://github.com/brineylab/BALM-paper) (paired, unpaired)
- [ft-ESM2](https://github.com/brineylab/BALM-paper)

## Installation

Clone this repository:

```console
git clone https://github.com/alexpilotti/bbk-mres abflow-cli
cd abflow-cli
```

A virtual environment is recommended:

```console
python3 -m venv venv
source venv/bin/activate
```

Python requirements:

```console
pip install -r requirements.txt

# ANARCI cannot be installed via pip:
pushd ..
git clone https://github.com/oxpig/ANARCI
cd ANARCI
python setup.py install
popd
```

## Commands

### Command line help

Just run the following for a complete list of command line options:

```console
python abflow/cli.py
```

### Binary classification (antigen-specificity) fine-tuning

```console
python abflow/cli.py seq-fine-tuning [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH] [--use-default-model-tokenizer]
    -i INPUT -c {H,L,HL} [-d DEVICE] -o OUTPUT
    [--frozen-layers FROZEN_LAYERS [FROZEN_LAYERS ...]]
    [-b BATCH_SIZE] [-e EPOCHS]
    [-s {no,steps,epoch}]
```

### Binary classification (antigen-specificity) prediction

```console
python abflow/cli.py seq-prediction [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH] [--use-default-model-tokenizer]
    -i INPUT -c {H,L,HL} [-d DEVICE] -o OUTPUT
```

### Token classification (paratope prediction) fine-tuning

```console
python abflow/cli.py token-fine-tuning [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH]
    [--use-default-model-tokenizer] -i INPUT -c {H,L,HL}
    [-d DEVICE] -o OUTPUT
    [--frozen-layers FROZEN_LAYERS [FROZEN_LAYERS ...]]
    [-b BATCH_SIZE] [-e EPOCHS]
    [-s {no,steps,epoch}] [--region REGION]
```

### Token classification (paratope prediction) prediction

```console
python abflow/cli.py token-prediction [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH] [--use-default-model-tokenizer]
    -i INPUT -c {H,L,HL} [-d DEVICE] -o OUTPUT -P PREDICTION
    [--region REGION]
```

### Retrieve model attention values

```console
python abflow/cli.py attentions [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH] [--use-default-model-tokenizer]
    -i INPUT -c {H,L,HL} [-d DEVICE] -o OUTPUT
    [-l LAYERS [LAYERS ...]]
    [--scheme {m,c,k,imgt,kabat,chothia,martin,i,a,aho,wolfguy,w}]
    [-s SEQUENCE_INDEXES [SEQUENCE_INDEXES ...] |
    --max-sequences MAX_SEQUENCES]
```

### Retrieve model embeddings

```console
python abflow/cli.py embeddings [-h]
    -m {BALM-paired,BALM-unpaired,AntiBERTy,AntiBERTa2,ESM1b,ESM2-15B,ESM2-3B,
        ESM2-650M,ESM2-150M,ESM2-35M,ESM2-8M}
    [-p MODEL_PATH] [--use-default-model-tokenizer]
    -i INPUT -c {H,L,HL} [-d DEVICE] -o OUTPUT
    [-s SEQUENCE_INDEXES [SEQUENCE_INDEXES ...]]
```

### SVM (antigen-specificity) prediction

```console
python abflow/cli.py svm-embeddings-predict [-h]
    -m MODEL_PATH -i INPUT -e EMBEDDINGS -o OUTPUT
```

### Split datasets

Split training and evaluation datasets based based on stratified k-folds.

```console
python abflow/cli.py split-data [-h]
    -i INPUT -o OUTPUT -l POSITIVE_LABELS [POSITIVE_LABELS ...] -f FOLD
```

### Remove similar antibody seqeuences

This uses MMSeqs2's clustering to remove similar sequences from a dataset.

```console
python abflow/cli.py remove-similar-sequences [-h]
    -i INPUT [-t TARGET] -o OUTPUT -c {H,L,HL} -m MIN_SEQ_ID
```

### Undersample dataset

This is used to match a separate's dataset class distribution

```console
python abflow/cli.py undersample [-h]
    -i INPUT [-t TARGET] [--target-dataset TARGET_DATASET] -o OUTPUT
```

### Shuffle labels

Shuffle the labales in a dataset randomly.

```console
python abflow/cli.py shuffle [-h] -i INPUT [-c COLUMN] -o OUTPUT
```

## Acknowledgements

This project includes code snippets from the following work, with many thanks
to Dr. Mamie Wang.

M. Wang, J. Patsenker, H. Li, Y. Kluger, and S. H. Kleinstein, 'Supervised
finetuning of pre-trained antibody language models improves antigen specificity
prediction', May 13, 2024.
doi: [10.1101/2024.05.13.593807](https://www.biorxiv.org/content/10.1101/2024.05.13.593807v1)
