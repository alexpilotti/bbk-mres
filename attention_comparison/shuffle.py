import logging

import numpy as np
import pandas as pd

import common

LOG = logging.getLogger(__name__)


def load_data(data_path):
    LOG.info(f"Loading data dome \"{data_path}\"")
    return pd.read_parquet(data_path)


def shuffle_column_values(data, column_name):
    common.set_seed()
    LOG.info(f"Shuffling values in column \"{column_name}\"")
    data[column_name] = np.random.permutation(data[column_name].values)
    return data


def save_data(data, output_path):
    LOG.info(f"Saving data to \"{output_path}\"")
    data.to_parquet(output_path)
