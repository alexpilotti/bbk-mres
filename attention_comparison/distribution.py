import logging

import pandas as pd

import common

LOG = logging.getLogger(__name__)


def load_data(data_path):
    return pd.read_parquet(data_path)


def set_equal_count(input_data):
    LOG.info(f"Number of initial rows: {len(input_data)}")
    LOG.info(f"Initial counts:\n{common.format_label_counts(input_data)}")

    min_count = input_data[common.LABEL_COL_NAME].value_counts().min()
    output_data = input_data.groupby(
        common.LABEL_COL_NAME, group_keys=False).apply(
            lambda x: x.sample(
                min_count,
                random_state=common.DEFAULT_SEED)).reset_index(drop=True)

    LOG.info(f"Number of final rows: {len(output_data)}")
    return output_data


def match_target_data_distribution(input_data, target_data):
    input_data_labels = sorted(target_data[common.LABEL_COL_NAME].unique())
    target_data_labels = sorted(target_data[common.LABEL_COL_NAME].unique())

    if input_data_labels != target_data_labels:
        raise Exception(f"The input labels {input_data_labels} do not match "
                        f"the target ones {target_data_labels}")

    LOG.info(f"Reference counts:\n{common.format_label_counts(target_data)}")
    LOG.info(f"Number of initial rows: {len(input_data)}")
    LOG.info(f"Initial counts:\n{common.format_label_counts(input_data)}")

    ref_distribution = target_data[common.LABEL_COL_NAME].value_counts(
        normalize=True)

    data_counts = (ref_distribution * len(input_data)).round().astype(int)
    avail_counts = input_data[common.LABEL_COL_NAME].value_counts()

    scale_factor = min(avail_counts.get(label, 0) / data_counts[label]
                       for label in data_counts.index if
                       data_counts[label] > 0)

    adj_data_counts = (data_counts * scale_factor).astype(int)

    output_data = input_data.groupby(common.LABEL_COL_NAME).apply(
        lambda x: x.sample(adj_data_counts.loc[x.name],
                           random_state=common.DEFAULT_SEED)).reset_index(
                               drop=True)

    LOG.info(f"Number of final rows: {len(output_data)}")
    LOG.info(f"Final counts:\n{common.format_label_counts(output_data)}")
    return output_data


def save_data(data, output_path):
    LOG.info(f"Saving data to \"{output_path}\"")
    data.to_parquet(output_path)
