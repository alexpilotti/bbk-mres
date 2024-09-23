import logging
import os
import tempfile

import pandas as pd

import common
import mmseq2_wrapper

LOG = logging.getLogger(__name__)

DB_NAME = "db"
CLUSTER_DB_NAME = "cluster_db"
REP_SEQ_DB_NAME = "rep_seq_db"
TMP_DIR = "tmp"
INPUT_FASTA_FILE = "input_data.fasta"
OUTPUT_FASTA_FILE = "output_data.fasta"
CLUSTERS_TSV_FILE = "clusters.tsv"


def _save_fasta(data, fasta_path, column_name):
    with open(fasta_path, 'w') as f:
        for index, row in data.iterrows():
            f.write(f">{index}\n{row[column_name]}\n")


def _read_fasta(fasta_path):
    with open(fasta_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    desc = [line[1:] for line in lines[::2]]
    seqs = lines[1::2]
    return (desc, seqs)


def _get_cluster_representative_sequences(input_data, min_seq_id, chain):
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_dir = os.path.join(tmp_dir, DB_NAME)
        os.makedirs(db_dir)
        db_path = os.path.join(db_dir, DB_NAME)

        input_fasta_path = os.path.join(tmp_dir, INPUT_FASTA_FILE)
        _save_fasta(input_data, input_fasta_path, chain)

        mmseq2_wrapper.create_db(input_fasta_path, db_path)

        cluster_db_dir = os.path.join(tmp_dir, CLUSTER_DB_NAME)
        os.makedirs(cluster_db_dir)
        cluster_db_path = os.path.join(cluster_db_dir, CLUSTER_DB_NAME)
        mmseq2_tmp_dir = os.path.join(tmp_dir, TMP_DIR)
        os.makedirs(mmseq2_tmp_dir)

        mmseq2_wrapper.cluster(db_path, cluster_db_path, mmseq2_tmp_dir,
                               min_seq_id)

        rep_seq_db_dir = os.path.join(tmp_dir, REP_SEQ_DB_NAME)
        os.makedirs(rep_seq_db_dir)
        rep_seq_db_path = os.path.join(rep_seq_db_dir, REP_SEQ_DB_NAME)

        mmseq2_wrapper.result2repseq(db_path, cluster_db_path, rep_seq_db_path)

        output_fasta_path = os.path.join(tmp_dir, OUTPUT_FASTA_FILE)

        mmseq2_wrapper.result2flat(db_path, db_path, rep_seq_db_path,
                                   output_fasta_path)

        ids, sequences = _read_fasta(output_fasta_path)
        cluster_reps = pd.DataFrame({chain: sequences}, index=map(int, ids))

        LOG.info("Total representative sequences identified: "
                 f"{len(cluster_reps)}")

        return _check_label_consistency(tmp_dir, input_data, cluster_reps,
                                        db_path, cluster_db_path)


def _check_label_consistency(tmp_dir, input_data, cluster_reps, db_path,
                             cluster_db_path):
    """
    Removes sequences from cluster_reps where labels in the related cluster
    do not have consistent values.
    """
    clusters_tsv_path = os.path.join(tmp_dir, CLUSTERS_TSV_FILE)
    mmseq2_wrapper.createtsv(db_path, db_path, cluster_db_path,
                             clusters_tsv_path)

    df = pd.read_csv(clusters_tsv_path, sep='\t', header=None,
                     names=['Rep', 'Seq'])
    clusters = df.groupby('Rep')['Seq']

    updated_cluster_reps = cluster_reps
    for rep, seqs in clusters:
        if len(input_data.loc[seqs]["label"].unique()) > 1:
            LOG.debug(f"Removing representative sequence with index {rep} due "
                      f"to inconsistent labels in the cluster")
            updated_cluster_reps = updated_cluster_reps.drop(rep)

    LOG.info("Total representative sequences removed due to inconsistent "
             "labels in the corresponding clusters: "
             f"{len(cluster_reps) - len(updated_cluster_reps)}")

    return updated_cluster_reps


def _drop_duplicates(input_data, chain):
    # Even if the dataset contains unique HL values, there might be
    # duplicated H or L chains
    subset = [chain.H, chain.L] if chain == common.CHAIN_HL else chain
    return input_data.drop_duplicates(subset=subset)


def remove_similar_sequences(input_data, min_seq_id, chain):
    LOG.info(f"Number of initial rows: {len(input_data)}")

    unique_input_data = _drop_duplicates(input_data, chain)
    LOG.info(f"Number of duplicate rows removed based on chain {chain}: "
             f"{len(input_data) - len(unique_input_data)}")

    identity_data = _get_cluster_representative_sequences(
        unique_input_data, min_seq_id, chain)

    output_data = unique_input_data.loc[identity_data.index]

    LOG.info(f"Number of final rows: {len(output_data)}")

    return output_data


def read_data(input_path):
    return pd.read_parquet(input_path)


def save_data(data, output_path):
    LOG.info(f"Saving data to \"{output_path}\"")
    data.to_parquet(output_path)
