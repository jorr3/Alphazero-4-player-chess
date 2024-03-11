from typing import SupportsIndex

import chess
import numpy as np
import torch
import wandb
import chess.engine

from alphazero_implementation.igame import IGame
from alphazero_implementation.node import Node
from alphazero_implementation.move import Move
from dualingDQN.agent import Agent
from utils import profile


class TwoPlayerChess(IGame):
    board_size = 8
    num_state_channels = 14
    state_space_size = num_state_channels * board_size ** 2
    num_queen_moves = len(Move.queen_move_offsets) * (board_size - 1)
    num_action_channels = num_queen_moves + len(Move.knight_move_offsets)
    action_space_size = num_action_channels * board_size ** 2

    def __init__(self):
        self.state = self.get_initial_state()
        self.root = None
        self.node = None
        self.memory = []

    def __repr__(self):
        return str(self.state)

    def get_initial_state(self) -> chess.Board:
        return chess.Board()

    @classmethod
    def take_action(cls, state: chess.Board, action: Move, player: int):
        """Takes actions on a batch of states and updates the game states post-action."""

        board = state.copy()
        board.turn = player == 0
        from_square = chess.square(action.from_col, action.from_row)
        to_square = chess.square(action.to_col, action.to_row)

        move = chess.Move(from_square, to_square)

        if move not in board.legal_moves:
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if move not in board.legal_moves:
                raise Exception("Illegal move attempted.")

        board.push(move)

        return board

    @classmethod
    def get_legal_actions_mask(cls, state: list[chess.Board], device: str) -> torch.tensor:
        """Returns a batch of binary vectors indicating valid moves for each state."""
        legal_moves_mask = torch.zeros(cls.num_action_channels * cls.board_size ** 2, dtype=torch.float32,
                                       device=device)
        # print(len(list(state.legal_moves)))

        for move in state.legal_moves:
            from_ = divmod(move.from_square, cls.board_size)
            to = divmod(move.to_square, cls.board_size)
            move_index = Move.get_flat_index(*from_, *to, cls.board_size)
            legal_moves_mask[move_index] = 1

        return legal_moves_mask

    @classmethod
    def get_terminated(cls, state: chess.Board) -> tuple[bool, float]:
        """Returns the value of the current state and a flag indicating if the game is over."""
        if state.is_checkmate():  # The player who made the last move is the winner.
            return (True, 1.0)
        elif state.is_stalemate() or state.is_insufficient_material() or state.can_claim_draw():
            return (True, 0.0)
        else:
            return (False, None)

    @classmethod
    def get_opponent(cls, player: int) -> int:
        return (player + 1) % 2

    @staticmethod
    def get_opponent_value(value: float) -> float:
        """Returns the value from the perspective of the opponent."""
        return -value

    @classmethod
    def get_encoded_states(cls, states: list[chess.Board], device: str) -> torch.Tensor:
        """
        Encode a batch of board states into a tensor suitable for the DQN, including attacked squares maps.
        Each board state is encoded from the perspective of the player whose turn it is.
        """
        batch_size = len(states)
        encoded_states = torch.zeros((batch_size, cls.num_state_channels, cls.board_size, cls.board_size),
                                     dtype=torch.float32, device=device)

        for i, state in enumerate(states):
            for square in chess.SQUARES:
                piece = state.piece_at(square)
                if piece:
                    row, col = divmod(square, cls.board_size)
                    if not state.turn:  # Flip the board for Black
                        row, col = cls.board_size - 1 - row, cls.board_size - 1 - col
                    idx = piece.piece_type - 1 + (6 if piece.color else 0)
                    encoded_states[i, idx, row, col] = 1

            encoded_states[i, -1, :, :] = cls.get_attacked_squares_map(state,
                                                                       chess.WHITE if state.turn else chess.BLACK,
                                                                       device)
            encoded_states[i, -2, :, :] = cls.get_attacked_squares_map(state,
                                                                       chess.BLACK if state.turn else chess.WHITE,
                                                                       device)
        return encoded_states

    @classmethod
    def get_attacked_squares_map(cls, board, color, device) -> torch.Tensor:
        """
        Generate a map of squares attacked by a given color using PyTorch tensors.
        """
        attacked_map = torch.zeros((cls.board_size, cls.board_size), dtype=torch.float32, device=device)
        for square in chess.SQUARES:
            if board.is_attacked_by(color, square):
                row, col = divmod(square, cls.board_size)
                attacked_map[row, col] = 1
        return attacked_map
