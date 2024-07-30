import argparse
import logging
import pathlib
import sys

import attention_weights
import common
import embeddings
import models
import numbering
import svm_embeddings_prediction


CMD_ATTENTIONS = "attentions"
CMD_EMBEDDINGS = "embeddings"
CMD_SVM_EMBEDDINGS_PREDICTION = "svm-embeddings-prediction"


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


def _add_common_args(parser):
    parser.add_argument(
        "-m", "--model", required=True,
        choices=models.MODELS,
        help="The model type")
    parser.add_argument(
        "-p", "--model_path", required=False,
        type=_valid_dir_arg,
        help="The model directory")
    parser.add_argument(
        '--use-default-model-tokenizer', required=False,
        action='store_true',
        help="When a custom model path which does not include a tokenizer is "
        "provided, this option allows to use the default tokenizer")
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-s", "--sequence-indexes", required=False,
        nargs='+', type=int,
        help="Indexes of the sequences in the data file. Accepts multiple "
        "values"),
    parser.add_argument(
        "-c", "--chain", required=True,
        choices=common.CHAIN_TYPES,
        help="The antibody chain(s), can be H, L, HL"),


def _add_attentions_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the output file that will contain the attentions "
        "weights for the sequences")
    parser.add_argument(
        "-l", "--layers", required=False,
        nargs='+', type=int,
        help="The model layers to get weights from")
    parser.add_argument(
        "--scheme", required=False,
        choices=numbering.SCHEME_NAMES,
        default=numbering.SCHEME_IMGT,
        help="The sequence numbering scheme to use")


def _add_embeddings_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the output file that will contain the embeddings for "
        "the sequences")


def _add_svm_embeddings_prediction_args(parser):
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-e", "--embeddings", required=True,
        type=_valid_file_arg,
        help="Path of the embeddings file")
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the CSV output file that will contain the score")
    parser.add_argument(
        '--shuffle', required=False,
        action='store_true',
        help="Shuffle the embedding labels")


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='command', required=True, help="Available commands")

    attentions_parser = subparsers.add_parser(
        CMD_ATTENTIONS, help="Get attentions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(attentions_parser)
    _add_attentions_args(attentions_parser)

    embeddings_parser = subparsers.add_parser(
        CMD_EMBEDDINGS, help="Get embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(embeddings_parser)
    _add_embeddings_args(embeddings_parser)

    svm_embeddings_prediction_parser = subparsers.add_parser(
        CMD_SVM_EMBEDDINGS_PREDICTION, help="SVM embeddings prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_svm_embeddings_prediction_args(svm_embeddings_prediction_parser)

    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        for choice in subparsers.choices:
            print(f"\n{choice}\n")
            subparsers.choices[choice].print_help()
    else:
        return parser.parse_args()


def _process_attentions_command(args):
    sequences = attention_weights.get_sequences(
        args.input, args.chain, args.sequence_indexes)

    sequences = numbering.get_adjusted_sequence_numbering(
        sequences, args.scheme)

    attentions = attention_weights.get_attention_weights(
        args.model, args.model_path, args.use_default_model_tokenizer,
        sequences, args.layers)

    attention_weights.save_attentions(attentions, args.output)


def _process_embeddings_command(args):
    sequences = attention_weights.get_sequences(
        args.input, args.chain, args.sequence_indexes)

    emb = embeddings.get_embeddings(
        args.model, args.model_path, sequences)
    embeddings.save_embeddings(emb, args.output)


def _process_svm_embeddings_prediction_command(args):
    svm_embeddings_prediction.compute_prediction(
        args.input, args.embeddings, args.output, args.shuffle)


def _setup_logging():
    logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    args = _parse_args()
    if not args:
        sys.exit(0)

    _setup_logging()

    if args.command == CMD_ATTENTIONS:
        _process_attentions_command(args)
    elif args.command == CMD_EMBEDDINGS:
        _process_embeddings_command(args)
    elif args.command == CMD_SVM_EMBEDDINGS_PREDICTION:
        _process_svm_embeddings_prediction_command(args)
