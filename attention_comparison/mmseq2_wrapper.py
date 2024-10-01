import os
import subprocess

ALIGN_SCORE_COV_SEQID = 3


def _run_process(args):
    return subprocess.run(args, check=True)


def create_db(fasta_path, seq_db_path):
    _run_process(
        ["mmseqs", "createdb", fasta_path, seq_db_path])


def create_index(seq_db_path, tmp_path):
    _run_process(
        ["mmseqs", "createindex", seq_db_path, tmp_path])


def easy_search(target_db_path, fasta_path, m8_path, tmp_path, format_output,
                alignment_mode=ALIGN_SCORE_COV_SEQID):
    _run_process(
        ["mmseqs", "easy-search", "--alignment-mode", str(alignment_mode),
         "--format-output", ",".join(
             format_output), fasta_path, target_db_path, m8_path, tmp_path])


def cluster(seq_db_path, cluster_db_path, tmp_path, min_seq_id=0.9,
            cov_mode=0, cluster_mode=0):
    _run_process(
        ["mmseqs", "cluster", seq_db_path, cluster_db_path, tmp_path,
         "--min-seq-id", str(min_seq_id), "--cov-mode", str(cov_mode),
         "--cluster-mode", str(cluster_mode)])


def result2repseq(input_seq_db_path, result_db_path, output_seq_db_path):
    _run_process(
        ["mmseqs", "result2repseq", input_seq_db_path, result_db_path,
         output_seq_db_path])


def result2flat(query_db_path, target_db_path, result_db_path, fasta_path):
    _run_process(
        ["mmseqs", "result2flat", query_db_path, target_db_path,
         result_db_path, fasta_path, "--use-fasta-header"])


def createtsv(query_db_path, target_db_path, result_db_path, tsv_path):
    _run_process(
        ["mmseqs", "createtsv", query_db_path, target_db_path, result_db_path,
         tsv_path])
