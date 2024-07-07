import anarci

SCHEME_NAMES = anarci.scheme_names
SCHEME_IMGT = "imgt"


def _get_adj_sequence_numbering(sequence, scheme):
    numbering, chain_type = anarci.number(sequence, scheme=scheme)

    adj_seq_numbering = []
    j = 0
    for i, r1 in enumerate(sequence):
        while True:
            (pos, ins), r2 = numbering[j]
            j += 1
            if ins != ' ' or r1 == r2:
                adj_seq_numbering.append(pos - 1)
                break
            if r2 != "-":
                raise Exception(f"Unrecognized residue at position {pos}")

    assert len(adj_seq_numbering) == len(sequence)
    return adj_seq_numbering, chain_type


def get_adjusted_sequence_numbering(sequences, scheme):
    adj_sequences = []

    for sequence_index, seq_dict in enumerate(sequences):
        print(f"Sequence {sequence_index} out of {len(sequences)}")
        seq_dict_new = {}
        try:
            for chain, sequence in seq_dict.items():
                adj_numbering, detected_chain = _get_adj_sequence_numbering(
                    sequence, scheme)
                assert chain == detected_chain
                seq_dict_new[chain] = (sequence, adj_numbering)
            adj_sequences.append(seq_dict_new)
        except Exception as ex:
            print(f"Skipping sequence {sequence_index}: {ex}")

    return adj_sequences
