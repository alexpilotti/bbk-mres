import argparse
import logging
import pathlib
import sys

import attention_weights
import common
import data_splitting
import distribution
import embeddings
import models
import numbering
import seq_classification_fine_tuning as seq_class_ft
import sequence_identity
import shuffle
import svm_embeddings_prediction
import token_classification_fine_tuning as token_class_ft


CMD_ATTENTIONS = "attentions"
CMD_EMBEDDINGS = "embeddings"
CMD_SEQ_FINE_TUNING = "seq-fine-tuning"
CMD_SEQ_PREDICT = "seq-prediction"
CMD_TOKEN_FINE_TUNING = "token-fine-tuning"
CMD_TOKEN_PREDICT = "token-prediction"
CMD_REMOVE_SIMILAR_SEQUENCES = "remove-similar-sequences"
CMD_SHUFFLE = "shuffle"
CMD_SPLIT_DATA = "split-data"
CMD_SVM_EMBEDDINGS_PREDICTION = "svm-embeddings-prediction"
CMD_UNDERSAMPLE = "undersample"

DEFAULT_MIN_SEQ_ID = 0.9


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
        "-p", "--model-path", required=False,
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
        "-c", "--chain", required=True,
        choices=common.CHAIN_TYPES,
        help="The antibody chain(s), can be H, L, HL")
    parser.add_argument(
        "-d", "--device", required=False,
        type=str,
        help="The device to use, e.g. cuda, cpu")


def _add_shuffle_data_args(parser):
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Data in Apache Parquet format path")
    parser.add_argument(
        "-c", "--column", required=False,
        type=str, default=common.LABEL_COL_NAME,
        help="Name of the column to shuffle")
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the output file that will contain the shuffled data")


def _add_split_data_args(parser):
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the output file that will contain the splitted data")
    parser.add_argument(
        "-l", "--positive-labels", required=True,
        nargs='+', type=str,
        help="List of positive labels")
    parser.add_argument(
        "-f", "--fold", required=True,
        type=int,
        help="Fold index")


def _add_fine_tuning_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Directory where the model will be saved")
    parser.add_argument(
        "--frozen-layers", required=False,
        nargs='+', type=int,
        help="The model layers to freeze")
    parser.add_argument(
        "-b", "--batch_size", required=False,
        type=int, default=seq_class_ft.DEFAULT_BATCH_SIZE,
        help="Batch size")
    parser.add_argument(
        "-e", "--epochs", required=False,
        type=int, default=seq_class_ft.DEFAULT_EPOCHS,
        help="Number of training epochs")
    parser.add_argument(
        "-s", "--save-strategy", required=False,
        choices=seq_class_ft.SAVE_STRATEGIES,
        default=seq_class_ft.DEFAULT_SAVE_STRATEGY,
        help="The model save strategy")


def _add_seq_predict_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the file containing the prediction metrics")


def _add_token_predict_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the file containing the prediction metrics")
    parser.add_argument(
        "-P", "--prediction", required=True,
        type=pathlib.Path,
        help="Path of the file containing the predicted labels")


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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s", "--sequence-indexes", required=False,
        nargs='+', type=int,
        help="Indexes of the sequences in the data file. Accepts multiple "
        "values")
    group.add_argument(
        "--max-sequences", required=False,
        type=int,
        help="Use only up to the first maximum number of sequences in the "
        "data file")


def _add_embeddings_args(parser):
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Path of the output file that will contain the embeddings for "
        "the sequences")
    parser.add_argument(
        "-s", "--sequence-indexes", required=False,
        nargs='+', type=int,
        help="Indexes of the sequences in the data file. Accepts multiple "
        "values")


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
        "-l", "--positive-labels", required=False,
        nargs='+', type=str, default=["1"],
        help="List of positive labels")


def _add_remove_similar_sequences_args(parser):
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-t", "--target", required=False,
        type=_valid_file_arg,
        help=("Target sequences data (e.g. training set) in Apache Parquet "
              "format path"))
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-c", "--chain", required=True,
        choices=common.CHAIN_TYPES,
        help="The antibody chain(s), can be H, L, HL")
    parser.add_argument(
        "-m", "--min-seq-id", required=True,
        type=float, default=DEFAULT_MIN_SEQ_ID,
        help="Mininum sequence identity")


def _add_undersample_args(parser):
    parser.add_argument(
        "-i", "--input", required=True,
        type=_valid_file_arg,
        help="Sequences data in Apache Parquet format path")
    parser.add_argument(
        "-t", "--target", required=False,
        type=_valid_file_arg,
        help=("Target distribution data (e.g. training set) in Apache Parquet "
              "format path"))
    parser.add_argument(
        "--target-dataset", required=False,
        type=str,
        help=("Filter target data by the provided dataset column value "
              "(e.g. \"train\")"))
    parser.add_argument(
        "-o", "--output", required=True,
        type=pathlib.Path,
        help="Sequences data in Apache Parquet format path")


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='command', required=True, help="Available commands")

    split_data_parser = subparsers.add_parser(
        CMD_SPLIT_DATA, help="Process and split the sequences for training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_split_data_args(split_data_parser)

    seq_fine_tuning_parser = subparsers.add_parser(
        CMD_SEQ_FINE_TUNING, help="Sequence classification fine tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(seq_fine_tuning_parser)
    _add_fine_tuning_args(seq_fine_tuning_parser)

    token_fine_tuning_parser = subparsers.add_parser(
        CMD_TOKEN_FINE_TUNING, help="Token classification fine tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(token_fine_tuning_parser)
    _add_fine_tuning_args(token_fine_tuning_parser)

    seq_predict_parser = subparsers.add_parser(
        CMD_SEQ_PREDICT, help="Sequence classification prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(seq_predict_parser)
    _add_seq_predict_args(seq_predict_parser)

    token_predict_parser = subparsers.add_parser(
        CMD_TOKEN_PREDICT, help="Token classification prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common_args(token_predict_parser)
    _add_token_predict_args(token_predict_parser)

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

    remove_similar_sequences_parser = subparsers.add_parser(
        CMD_REMOVE_SIMILAR_SEQUENCES, help="Remove similar sequences",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_remove_similar_sequences_args(remove_similar_sequences_parser)

    undersample_parser = subparsers.add_parser(
        CMD_UNDERSAMPLE, help="Undersample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_undersample_args(undersample_parser)

    shuffle_parser = subparsers.add_parser(
        CMD_SHUFFLE, help="Shuffle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_shuffle_data_args(shuffle_parser)

    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        for choice in subparsers.choices:
            print(f"\n{choice}\n")
            subparsers.choices[choice].print_help()
    else:
        return parser.parse_args()


def _process_split_data_command(args):
    data = data_splitting.load_data(args.input, args.positive_labels)
    data = data_splitting.process_data(data, args.fold)
    data_splitting.save_data(data, args.output)


def _process_seq_fine_tuning_command(args):
    data = seq_class_ft.load_data(args.input)
    seq_class_ft.train(data, args.chain, args.model, args.model_path,
                       args.use_default_model_tokenizer, args.frozen_layers,
                       args.output, args.batch_size, args.epochs,
                       args.save_strategy, args.device)


def _process_token_fine_tuning_command(args):
    data = token_class_ft.load_data(args.input)
    token_class_ft.train(data, args.chain, args.model, args.model_path,
                         args.use_default_model_tokenizer, args.frozen_layers,
                         args.output, args.batch_size, args.epochs,
                         args.save_strategy, args.device)


def _process_seq_predict_command(args):
    data = seq_class_ft.load_data(args.input)
    _, metrics = seq_class_ft.predict(data, args.chain, args.model,
                                      args.model_path,
                                      args.use_default_model_tokenizer,
                                      args.device)
    common.save_json_file(metrics, args.output)


def _process_token_predict_command(args):
    data = token_class_ft.load_data(args.input)

    data = token_class_ft.predict_labels(
        data, args.chain, args.model, args.model_path,
        args.use_default_model_tokenizer, args.device)
    metrics = token_class_ft.predict_metrics(
        data, args.chain, args.model, args.model_path,
        args.use_default_model_tokenizer, args.device)

    data.to_parquet(args.prediction)
    common.save_json_file(metrics, args.output)


def _process_attentions_command(args):
    sequences = attention_weights.get_sequences(
        args.input, args.chain, args.sequence_indexes, args.max_sequences)

    sequences = numbering.get_adjusted_sequence_numbering(
        sequences, args.scheme)

    attentions = attention_weights.get_attention_weights(
        args.model, args.model_path, args.use_default_model_tokenizer,
        sequences, args.layers, args.device)

    attention_weights.save_attentions(attentions, args.output)


def _process_embeddings_command(args):
    sequences = attention_weights.get_sequences(
        args.input, args.chain, args.sequence_indexes)

    emb = embeddings.get_embeddings(
        args.model, args.model_path, args.use_default_model_tokenizer,
        sequences, args.device)
    embeddings.save_embeddings(emb, args.output)


def _process_svm_embeddings_prediction_command(args):
    svm_embeddings_prediction.compute_prediction(
        args.input, args.embeddings, args.output, args.positive_labels)


def _process_remove_similar_sequences_command(args):
    input_data = sequence_identity.read_data(args.input)

    if args.target:
        target_data = sequence_identity.read_data(args.target)
    else:
        target_data = None

    output_data = sequence_identity.remove_similar_sequences(
        input_data, target_data, args.min_seq_id, args.chain)

    sequence_identity.save_data(output_data, args.output)


def _process_undersample_command(args):
    input_data = sequence_identity.read_data(args.input)

    if args.target:
        target_data = sequence_identity.read_data(args.target)

        output_data = distribution.match_target_data_distribution(
            input_data, target_data, args.target_dataset)
    else:
        output_data = distribution.set_equal_count(input_data)

    sequence_identity.save_data(output_data, args.output)


def _process_shuffle_command(args):
    input_data = shuffle.load_data(args.input)
    output_data = shuffle.shuffle_column_values(input_data, args.column)
    shuffle.save_data(output_data, args.output)


def _setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


if __name__ == '__main__':
    args = _parse_args()
    if not args:
        sys.exit(0)

    _setup_logging()

    if args.command == CMD_ATTENTIONS:
        _process_attentions_command(args)
    elif args.command == CMD_EMBEDDINGS:
        _process_embeddings_command(args)
    elif args.command == CMD_SEQ_FINE_TUNING:
        _process_seq_fine_tuning_command(args)
    elif args.command == CMD_TOKEN_FINE_TUNING:
        _process_token_fine_tuning_command(args)
    elif args.command == CMD_SEQ_PREDICT:
        _process_seq_predict_command(args)
    elif args.command == CMD_TOKEN_PREDICT:
        _process_token_predict_command(args)
    elif args.command == CMD_REMOVE_SIMILAR_SEQUENCES:
        _process_remove_similar_sequences_command(args)
    elif args.command == CMD_SHUFFLE:
        _process_shuffle_command(args)
    elif args.command == CMD_SPLIT_DATA:
        _process_split_data_command(args)
    elif args.command == CMD_SVM_EMBEDDINGS_PREDICTION:
        _process_svm_embeddings_prediction_command(args)
    elif args.command == CMD_UNDERSAMPLE:
        _process_undersample_command(args)
