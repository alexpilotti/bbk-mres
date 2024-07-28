import torch


CHAIN_H = "H"
CHAIN_L = "L"
CHAIN_HL = "HL"
CHAIN_TYPES = [CHAIN_H, CHAIN_L, CHAIN_HL]


def _get_best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"
