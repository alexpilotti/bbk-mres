import logging
import os
import re
import tempfile

import anarci
from Bio.SeqUtils import IUPACData
import pandas as pd
import sqlalchemy

import vcab_utils

DEFAULT_D_SASA_TH = 15

LOG = logging.getLogger(__name__)


def _load_token_classification_data(vcab_db_path, csv_path, pops_dir,
                                    d_sasa_th):
    pattern = re.compile(
        r"([a-zA-Z0-9]+)\_([a-zA-Z0-9])([a-zA-Z0-9])-([a-zA-Z0-9])"
        r"([a-zA-Z0-9])\_deltaSASA\_rpopsResidue.txt")

    engine = sqlalchemy.create_engine(f'sqlite:///{vcab_db_path}', echo=False)
    with engine.connect() as conn:
        vcab = pd.read_csv(csv_path)
        vcab.to_sql("vcab", conn, if_exists='append', index=False)

        pops_results = os.listdir(pops_dir)
        for pr in pops_results:
            match = pattern.match(pr)
            if match:
                result = match.groups()
                LOG.info(f"{pr} - {result}")
                pdbid = result[0]
                iden_chain_h = result[1]
                iden_chain_l = result[2]
                ab_chain = result[3]
                antigen_chain = result[4]
                iden_code = f"{pdbid}_{iden_chain_h}{iden_chain_l}"

                pops_data = pd.read_csv(os.path.join(pops_dir, pr), sep=' ',
                                        index_col=0)
                if len(pops_data) == 0:
                    LOG.info("pops_data is empty")
                else:
                    pops_data = vcab_utils.replace_back_T_F_chain(pops_data)
                    pops_data = pops_data[pops_data["Chain"] == ab_chain]

                    if len(pops_data) == 0:
                        LOG.info("Filtered data by Chain is empty")

                    pops_data.insert(0, "PDBID", pdbid)
                    pops_data.insert(1, "iden_code", iden_code)
                    pops_data.insert(2, "iden_chain_H", iden_chain_h)
                    pops_data.insert(3, "iden_chain_L", iden_chain_l)
                    pops_data.insert(4, "ab_chain", ab_chain)
                    pops_data.insert(5, "antigen_chain", antigen_chain)
                    pops_data.insert(
                        6, "ResidNe_1", pops_data['ResidNe'].apply(
                            lambda x: IUPACData.protein_letters_3to1[
                                x.capitalize()]))

                    pops_data.reset_index()
                    pops_data.to_sql("pops_results", conn, if_exists='append',
                                     index=False)

                chain = "H" if iden_chain_h == ab_chain else "L"

                mapping_results = vcab_utils.mapping_positions(
                    iden_code, vcab, f"{chain}V_seq",
                    f"{chain}V_coordinate_seq",
                    f"{chain}V_PDB_numbering",
                    pops_dir, "antigen_chain",
                    chain, d_sasa_th)

                mapping_results.insert(0, "iden_code", iden_code)
                mapping_results.insert(1, "chain", chain)
                mapping_results.insert(2, "pdb_chain", ab_chain)

                mapping_results.to_sql(
                    "mapping", conn, if_exists='append', index=False)
            else:
                raise Exception(f"Can't match: {pr}")


def _get_adj_sequence_numbering(sequence, scheme):
    numbering, chain_type = anarci.number(sequence, scheme=scheme)

    LOG.info(f"Sequence: {sequence}")
    LOG.info(f"ANARCI sequence: {''.join([r[1] for r in numbering])}")

    adj_seq_numbering = []
    j = 0
    num_list = [(a, b, c) for ((a, b), c) in numbering if b != ' ']
    if len(num_list):
        pass

    for i, r1 in enumerate(sequence):
        while True:
            err_msg = ()
            if j >= len(numbering):
                LOG.warning(
                    f"Residue {r1} at position {i} is beyond the sequence "
                    "returned from ANARCI")
                adj_seq_numbering.append((None, None))
                break
            (pos, ins), r2 = numbering[j]
            if ins != ' ' or r1 == r2:
                ins = ins.strip()
                adj_seq_numbering.append((pos, ins if len(ins) else None))
                j += 1
                break
            elif r2 != "-":
                LOG.warning(f"Unrecognized residue {r1} at position {i}")
                adj_seq_numbering.append((None, None))
                break
            else:
                j += 1

    assert len(adj_seq_numbering) == len(sequence)
    return adj_seq_numbering, chain_type


def _add_numbering(vcab_db_path="vcab.db"):
    engine = sqlalchemy.create_engine(f'sqlite:///{vcab_db_path}', echo=False)
    with engine.connect() as conn:
        insp = sqlalchemy.inspect(engine)
        columns = [c["name"] for c in insp.get_columns("mapping")]
        if "anarci_pos" not in columns:
            conn.execute(sqlalchemy.text(
                "alter table mapping add column anarci_pos int"))
        else:
            conn.execute(sqlalchemy.text(
                "update mapping set anarci_pos = NULL"))

        if "anarci_ins" not in columns:
            conn.execute(sqlalchemy.text(
                "alter table mapping add anarci_ins char(1)"))
        else:
            conn.execute(sqlalchemy.text(
                "update mapping set anarci_ins = NULL"))
        conn.commit()

        update_cmd = sqlalchemy.text(
            "update mapping set anarci_pos = :anarci_pos, "
            "anarci_ins = :anarci_ins where iden_code = :iden_code and "
            "chain = :chain and text_numbering = :seq_pos")
        result = conn.execute(sqlalchemy.text(
            "select iden_code, hv_seq, lv_seq, h_seq, l_seq "
            "from vcab order by iden_code"))
        for row in result:
            for chain in ["H", "L"]:
                LOG.info(f"iden_code: {row.iden_code}, chain: {chain}")
                adj_seq_numbering, chain_type = _get_adj_sequence_numbering(
                    row.HV_seq if chain == "H" else row.LV_seq, "imgt")

                if chain_type != chain:
                    raise Exception(
                        "ANARCI chain mismatch: {chain_type} {chain}")

                for seq_pos, (anarci_pos, anarci_ins) in enumerate(
                        adj_seq_numbering):
                    LOG.debug(
                        f"{row.iden_code}, {chain}, {seq_pos}, "
                        f"{anarci_pos} {anarci_ins}")
                    conn.execute(
                        update_cmd, {"iden_code": row.iden_code,
                                     "chain": chain,
                                     "anarci_pos": anarci_pos,
                                     "anarci_ins": anarci_ins,
                                     "seq_pos": seq_pos})
                conn.commit()


def _assign_dataset_by_species(species):
    return "train" if species == "Homo_Sapiens" else "test"


def _create_dataset(vcab_db_path, out_dataset_path,
                    insert_missing_residues=True):
    engine = sqlalchemy.create_engine(f'sqlite:///{vcab_db_path}', echo=False)
    with engine.connect() as conn:
        df = pd.read_sql(
            "select m.iden_code, v.species, m.chain, m.anarci_pos, "
            "m.anarci_ins, m.residue, m.pdb_numbering, "
            "cast(m.if_interface_res as integer) as "
            "if_interface_res from mapping m join vcab v on m.iden_code = "
            "v.iden_code where m.anarci_pos is not null "
            "order by m.iden_code, m.chain, m.anarci_pos, m.anarci_ins", conn)

    df["dataset"] = df["Species"].apply(_assign_dataset_by_species)

    df['if_interface_res'] = df['if_interface_res'].astype('Int64')
    df.loc[df['if_interface_res'].isna(), 'if_interface_res'] = -100
    max_pos = df["anarci_pos"].max()
    full_range = pd.Series(range(1, max_pos + 1), name='anarci_pos')

    if insert_missing_residues:
        df3 = pd.DataFrame()

        unique_combinations = df[['iden_code', 'chain']].drop_duplicates()
        for row in unique_combinations.itertuples():
            LOG.info(f"{row.iden_code} {row.chain}")
            df1 = df[(df["iden_code"] == row.iden_code) &
                     (df["chain"] == row.chain)]
            missing_pos = full_range[~full_range.isin(df1["anarci_pos"])]
            if len(missing_pos) == 0:
                continue
            first_row = df1.iloc[0]
            df2 = pd.DataFrame(missing_pos)
            df2["residue"] = "X"
            df2["if_interface_res"] = -100
            for c in ['iden_code', 'Species', 'chain', 'dataset']:
                df2[c] = first_row[c]
            for c in [c for c in df1.columns if c not in df2.columns]:
                df2[c] = None
            LOG.info(f"{row.iden_code} {row.chain} inserting "
                     f"missing positions: {len(df2)}")
            df3 = pd.concat([df3, df2])

        df = pd.concat([df, df3])
        df = df.sort_values(by=['iden_code', 'chain', 'anarci_pos'],
                            ascending=[True, True, True])

    df["anarci_pos_ins"] = (df["anarci_pos"].astype(str) +
                            df["anarci_ins"].fillna(""))

    df["pdb_numbering"] = df["pdb_numbering"].fillna("")

    df = df.groupby(
        ["iden_code", "Species", "dataset", "chain"],
        group_keys=False).agg({
            'anarci_pos_ins': lambda x: ','.join(map(str, x)),
            'residue': lambda x: ''.join(x),
            'if_interface_res': lambda x: ','.join(map(str, x)),
            'pdb_numbering': lambda x: ','.join(map(str, x))
        }).reset_index()

    df = df.rename(columns={
        'anarci_pos_ins': 'positions',
        'if_interface_res': 'labels',
        'residue': 'sequence'})

    pivoted_df = df.pivot(index=['iden_code', 'Species', 'dataset'],
                          columns='chain',
                          values=['positions', 'pdb_numbering', 'sequence',
                                  'labels'])
    pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()
    df = pivoted_df

    LOG.info(f"Initial sequences: {len(df)}")

    # Remove improbable sequences
    df = df[~df["positions_H"].str.contains("O", na=False)]

    # Remove cases where either chain H or L is missing
    missing_a_chain = df["sequence_H"].isna() | df["sequence_L"].isna()
    LOG.info("Removing sequences with missing H or L chains: "
             f"{len(df[missing_a_chain])}")
    df = df[~missing_a_chain]

    df["sequence_HL"] = df["sequence_H"] + df["sequence_L"]
    df["labels_HL"] = df["labels_H"] + "," + df["labels_L"]

    positions_H = df["positions_H"].apply(
        lambda x: ",".join(f"H{x}" for x in x.split(",")))
    positions_L = df["positions_L"].apply(
        lambda x: ",".join(f"L{x}" for x in x.split(",")))

    df["positions_HL"] = positions_H + "," + positions_L

    duplicates = df['sequence_HL'].duplicated()
    LOG.info(f"Removing duplicated sequences: {len(df[duplicates])}")
    df = df[~duplicates]

    LOG.info(df.groupby(['dataset']).size().reset_index(name='Count'))

    LOG.info(f"Final number of sequences: {len(df)}")
    df.to_parquet(out_dataset_path)


def process_vcab_data(csv_path, pops_dir, out_dataset_path,
                      d_sasa_th=DEFAULT_D_SASA_TH):
    with tempfile.NamedTemporaryFile() as tmp:
        vcab_db_path = tmp.name
        _load_token_classification_data(
            vcab_db_path, csv_path, pops_dir, d_sasa_th)
        _add_numbering(vcab_db_path)
        _create_dataset(vcab_db_path, out_dataset_path)
