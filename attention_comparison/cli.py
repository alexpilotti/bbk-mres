import argparse
import pathlib

import attention_weights
import models
import numbering


def _valid_dir_arg(value):
    path = pathlib.Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f'Directory not found: {value}')
    elif not path.is_dir():
        raise argparse.ArgumentTypeError(f'Not a directory: {value}')
    return str(path)


def _valid_file_arg(value):
    path = pathlib.Path(value)
    if not path.exists():
        msg = f'File not found: {value}'
        raise argparse.ArgumentTypeError(msg)
    return str(path)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        choices=models.MODELS,
                        help="The model type")
    parser.add_argument("-p", "--model_path", required=False,
                        type=_valid_dir_arg,
                        help="The model directory")
    parser.add_argument('--use-default-model-tokenizer', required=False,
                        action='store_true',
                        help="When a custom model path which does not include "
                             "a tokenizer is provided, this option allows to "
                             "use the default tokenizer")
    parser.add_argument("-i", "--input", required=True,
                        type=_valid_file_arg,
                        help="Sequences data in Apache Parquet format path")
    parser.add_argument("-s", "--sequence-indexes", required=False,
                        nargs='+', type=int,
                        help="Indexes of the sequences in the data file. "
                             "Accepts multiple values"),
    parser.add_argument("-o", "--output", required=True,
                        type=pathlib.Path,
                        help="Path of the output file that will contain the "
                             "attentions weights for the sequences")
    parser.add_argument("-c", "--chain", required=True,
                        choices=attention_weights.CHAIN_TYPES,
                        help="The antibody chain(s), can be H, L, HL"),
    parser.add_argument("-l", "--layers", required=False,
                        nargs='+', type=int,
                        help="The model layers to get weights from")
    parser.add_argument("--scheme", required=False,
                        choices=numbering.SCHEME_NAMES,
                        default=numbering.SCHEME_IMGT,
                        help="The sequence numbering scheme to use")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    sequences = attention_weights.get_sequences(
        args.input, args.chain, args.sequence_indexes)

    sequences = numbering.get_adjusted_sequence_numbering(
        sequences, args.scheme)

    attentions = attention_weights.get_attention_weights(
        args.model, args.model_path, args.use_default_model_tokenizer,
        sequences, args.layers)

    attention_weights.save_attentions(attentions, args.output)
