import torch
import start_fens

from alphazero_cpp import (
    Board as BoardCpp,
    Move,
    GameResult,
    PlayerColor,
    Board,
    piece_value,
    color_value,
    Player,
)
from line_profiler_pycharm import profile


class FourPlayerChess(BoardCpp):
    start_fen = start_fens.EIGHT_SIMPLE.replace("\n", "")

    batch_indices_tensor = None
    plane_indices_tensor = None
    row_indices_tensor = None
    col_indices_tensor = None
    legal_moves_masks = None

    @classmethod
    def init_tensors(cls, max_moves, batch_size, device):
        cls.batch_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.plane_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.row_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.col_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.legal_moves_masks = torch.zeros((batch_size, *cls.action_space_dims), dtype=torch.float32, device=device)

    @classmethod
    # @profile
    def get_legal_moves_mask(cls, states, device):
        batch_size = len(states)
        legal_moves = [state.GetLegalMoves() for state in states]
        num_moves = sum(len(moves) for moves in legal_moves)

        batch_indices, plane_indices, row_indices, col_indices = cls.GetLegalMovesIndices(legal_moves, num_moves)

        batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.int64, device=device)
        plane_indices_tensor = torch.tensor(plane_indices, dtype=torch.int64, device=device)
        row_indices_tensor = torch.tensor(row_indices, dtype=torch.int64, device=device)
        col_indices_tensor = torch.tensor(col_indices, dtype=torch.int64, device=device)

        legal_moves_masks = torch.zeros((batch_size, *cls.action_space_dims), dtype=torch.float32, device=device)

        # Setting the corresponding positions in the mask tensor to 1
        legal_moves_masks.index_put_(
            (batch_indices_tensor, plane_indices_tensor, row_indices_tensor, col_indices_tensor),
            torch.tensor(1, dtype=torch.float32, device=device)
        )

        return legal_moves_masks