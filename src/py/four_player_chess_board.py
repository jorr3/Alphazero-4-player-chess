import torch

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

from fourpchess_interface import FourPlayerChessInterface


class FourPlayerChess(BoardCpp):
    # start_fen = """R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,1,1,yB,yK,yQ,yB,1,1,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/1,bP,10,gP,1/1,bP,10,gP,1/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/1,bP,10,gP,1/1,bP,10,gP,1/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,1,1,rB,rQ,rK,rB,1,1,x,x,x"""
    start_fen = """R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"""

    board_size = 14
    num_state_channels = 24

    # Pre-calculate some values for performance
    state_space_size = num_state_channels * board_size ** 2
    num_action_channels = Move.num_queen_moves + Move.num_knight_moves
    action_space_size = num_action_channels * board_size ** 2
    action_space_dims = (num_action_channels, board_size, board_size)
    state_space_dims = (num_state_channels, board_size, board_size)

    colors = {PlayerColor.RED, PlayerColor.BLUE, PlayerColor.YELLOW, PlayerColor.GREEN}
    color_channel_offsets = {color: color_value(color) * 24 // 4 for color in colors}

    batch_indices_tensor = None
    plane_indices_tensor = None
    row_indices_tensor = None
    col_indices_tensor = None
    legal_moves_masks = None

    def __init__(self):
        super().__init__(*parse_board_from_fen(self.start_fen, self.board_size))
        self.state = parse_board_from_fen(self.start_fen, self.board_size)
        self.root = None
        self.node = None
        self.memory = []

    @classmethod
    def init_tensors(cls, max_moves, batch_size, device):
        cls.batch_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.plane_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.row_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.col_indices_tensor = torch.zeros(max_moves, dtype=torch.int64, device=device)
        cls.legal_moves_masks = torch.zeros((batch_size, *cls.action_space_dims), dtype=torch.float32, device=device)

    @classmethod
    @profile
    def get_legal_moves_mask(cls, states, device):
        batch_size = len(states)
        legal_moves = [state.GetLegalMoves() for state in states]
        num_moves = sum(len(moves) for moves in legal_moves)

        if cls.batch_indices_tensor is None or cls.batch_indices_tensor.size(0) < num_moves:
            cls.init_tensors(num_moves, batch_size, device)

        batch_indices, plane_indices, row_indices, col_indices = cls.GetLegalMovesIndices(legal_moves, num_moves)

        # Ensure the tensors are large enough to hold the indices
        if cls.batch_indices_tensor.size(0) < len(batch_indices):
            cls.init_tensors(len(batch_indices), batch_size, device)

        # Copy the indices data to the tensors using direct data access to avoid memory allocation
        cls.batch_indices_tensor[:len(batch_indices)].copy_(torch.tensor(batch_indices, dtype=torch.int64))
        cls.plane_indices_tensor[:len(plane_indices)].copy_(torch.tensor(plane_indices, dtype=torch.int64))
        cls.row_indices_tensor[:len(row_indices)].copy_(torch.tensor(row_indices, dtype=torch.int64))
        cls.col_indices_tensor[:len(col_indices)].copy_(torch.tensor(col_indices, dtype=torch.int64))

        # Reset the legal_moves_masks tensor
        cls.legal_moves_masks.zero_()

        # Setting the corresponding positions in the mask tensor to 1
        cls.legal_moves_masks.index_put_(
            (cls.batch_indices_tensor[:len(batch_indices)],
             cls.plane_indices_tensor[:len(plane_indices)],
             cls.row_indices_tensor[:len(row_indices)],
             cls.col_indices_tensor[:len(col_indices)]),
            torch.tensor(1, dtype=torch.float32, device=device)
        )

        return cls.legal_moves_masks

# @classmethod
    # def get_legal_moves_mask(cls, states, device):
    #     num_moves = sum(len(state.GetLegalMoves()) for state in states)
    #
    #     if cls.batch_indices_tensor is None or cls.batch_indices_tensor.size(0) < num_moves:
    #         cls.init_tensors(num_moves, len(states), device)
    #
    #     return cls.GetLegalMovesMask(
    #         states, device,
    #         cls.batch_indices_tensor,
    #         cls.plane_indices_tensor,
    #         cls.row_indices_tensor,
    #         cls.col_indices_tensor,
    #         cls.legal_moves_masks
    #     )

