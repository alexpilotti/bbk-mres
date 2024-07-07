import pandas as pd
import torch

import models

DEFAULT_NUM_LAYERS = 3

CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"
CHAIN_TYPES = [CHAIN_H, CHAIN_L, CHAIN_HL]


def get_sequences(data_path, chain, indexes):
    data = pd.read_parquet(data_path)
    if not indexes:
        indexes = data.index

    sequences = []
    for index in indexes:
        sequence = {}
        if chain in [CHAIN_H, CHAIN_HL]:
            sequence[CHAIN_H] = data.loc[index, CHAIN_H]
        if chain in [CHAIN_L, CHAIN_HL]:
            sequence[CHAIN_L] = data.loc[index, CHAIN_L]
        sequences.append(sequence)
    return sequences


def _get_best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_adjusted_sequence_numberings(chain_h_adj, chain_l_adj):
    if chain_h_adj:
        seq_adj = [(CHAIN_H, i) for i in chain_h_adj]
        if chain_l_adj:
            # TODO(alexpilotti): This might not be right in case of
            # deletions at the end
            chain_h_length = chain_h_adj[-1] + 1
            seq_adj += [(CHAIN_L, p + chain_h_length) for p in chain_l_adj]
    else:
        seq_adj = [(CHAIN_L, i) for i in chain_l_adj]
    return seq_adj


def get_attention_weights(model_name, model_path, sequences, layers):
    model, tokenizer, format_sequence = models.load_model(
        model_name, model_path)

    device = _get_best_device()
    model = model.to(device)

    attentions = []

    for sequence_index, sequence in enumerate(sequences):
        print(f"Sequence {sequence_index} out of {len(sequences)}")

        chain_h, chain_h_adj = sequence.get(CHAIN_H, [None, None])
        chain_l, chain_l_adj = sequence.get(CHAIN_L, [None, None])

        seq_adj = _get_adjusted_sequence_numberings(chain_h_adj, chain_l_adj)

        formatted_sequence = format_sequence(chain_h, chain_l)
        inputs = tokenizer(formatted_sequence, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        special_tokens = tokenizer.all_special_tokens

        seq_token_indexes = [
            i for i, token in enumerate(tokens) if token not in special_tokens]

        if not layers:
            num_layers = len(outputs.attentions)
            # Take DEFAULT_NUM_LAYERS starting from the end
            layers = range(num_layers - DEFAULT_NUM_LAYERS,
                           num_layers - 1)

        for layer in layers:
            attention_matrix = outputs.attentions[layer].cpu().detach().numpy()
            batch = 0
            batches, num_heads, seq_len1, seq_len2 = attention_matrix.shape

            assert batches == 1
            assert seq_len1 == seq_len2 == len(tokens)

            for head in range(0, num_heads):
                seq_len1_range = range(0, seq_len1)

                df = pd.DataFrame(
                    attention_matrix[batch, head],
                    index=seq_len1_range,
                    columns=seq_len1_range)

                # Filter out all special tokens
                df = df.loc[seq_token_indexes, seq_token_indexes]
                # Rename columns from the token indexes to the sequence indexes
                # using the adjusted sequence numbering
                df.rename(inplace=True, columns=dict(
                    [(t, seq_adj[i]) for i, t in
                     enumerate(seq_token_indexes)]))

                # Rename rows from the token indexes to the sequence indexes
                # using the adjusted sequence numbering
                df.index = pd.MultiIndex.from_tuples(
                    seq_adj, names=["Chain_1", "Seq_1"])

                df = df.stack().reset_index()

                # Split the 2nd sequnce (chain, residue) tuples in two columns.
                # Cannot use pd.MultiIndex.from_tuples() on the columns
                # before calling stack() due to duplicates
                df[['Chain_2', 'Seq_2']] = pd.DataFrame(
                    df['level_2'].tolist(), index=df.index)
                df.insert(2, 'Seq_2', df.pop('Seq_2'))
                df.insert(2, 'Chain_2', df.pop('Chain_2'))
                df.drop(columns=['level_2'], inplace=True)

                df.rename(inplace=True, columns={
                    'adj_index': 'Seq_1', 'level_1': 'Seq_2', 0: "Weight"})

                # Convert from float32 to float64
                # df['Weight'] = df['Weight'].astype(np.float64)

                df.insert(0, 'Head', head)
                df.insert(0, 'Layer', layer)
                df.insert(0, CHAIN_L, chain_l)
                df.insert(0, CHAIN_H, chain_h)

                df.set_index(
                    [CHAIN_H, CHAIN_L, 'Layer', 'Head', 'Seq_1', 'Seq_2'],
                    inplace=True)

                attentions.append(df)

    return pd.concat(attentions)


def save_attentions(attentions, output_path):
    attentions.to_parquet(output_path)
