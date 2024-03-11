import sys
from typing import Tuple

import torch
import re
from chessenv import (
    GameResult,
    PlayerColor,
    PieceType,
    Board,
    piece_value,
    color_value,
    Player,
)

from alphazero_implementation.igame import IGame
from fen_parser import parse_board_from_fen
from move import Move
from four_player_chess_interface import FourPlayerChessInterface

from line_profiler_pycharm import profile


class FourPlayerChess(IGame):
    start_fen = """R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x"""
    board_size = 14
    num_state_channels = 24
    state_space_size = num_state_channels * board_size**2
    num_queen_moves = len(Move.queen_move_offsets) * (board_size - 1)
    num_action_channels = num_queen_moves + len(Move.knight_move_offsets)
    action_space_size = num_action_channels * board_size**2

    def __init__(self):
        self.state = parse_board_from_fen(self.start_fen, self.board_size)
        self.turn = PlayerColor.RED
        self.root = None
        self.node = None
        self.memory = []

    def __repr__(self):
        pass

    @classmethod
    def take_action(cls, state, action: Move, player, verbose=False):
        """Takes action and returns the game state post-action."""
        state_copy = Board(state)
        state_copy.MakeMove(action.to_cpp())
        return state_copy

    @classmethod
    def get_legal_actions_mask(cls, state: list, device: str) -> torch.Tensor:
        """Returns a binary vector indicating valid moves for the current state, adjusted for player perspective."""
        legal_moves_mask = torch.zeros(
            cls.num_action_channels * cls.board_size**2,
            dtype=torch.float32,
            device=device,
        )

        for move in state.GetLegalMoves():
            from_ = move.From().GetRow(), move.From().GetCol()
            to = move.To().GetRow(), move.To().GetCol()

            # Adjust for player perspective
            for _ in range(color_value(state.GetTurn().GetColor())):
                from_, to = cls.rotate_position(*from_), cls.rotate_position(*to)

            move_index = Move.get_flat_index(*from_, *to, cls.board_size)
            legal_moves_mask[move_index] = 1

        return legal_moves_mask

    @classmethod
    def rotate_position(cls, row, col):
        """Rotates a position 90 degrees clockwise in a square matrix of the given size."""
        return col, cls.board_size - 1 - row

    @classmethod
    def get_terminated(cls, state, last_player_to_make_move: Player) -> tuple[bool, None] | tuple[bool, float]:
        """Returns if the game is over and the value of the current state."""
        terminal_value_map = {
            GameResult.STALEMATE: 0.0,
            GameResult.WIN_RY: 1.0 if last_player_to_make_move.GetColor() in [PlayerColor.RED, PlayerColor.YELLOW] else -1.0,
            GameResult.WIN_BG: 1.0 if last_player_to_make_move.GetColor() in [PlayerColor.BLUE, PlayerColor.GREEN] else -1.0,
        }

        result_state = state.GetGameResult()

        if result_state == GameResult.IN_PROGRESS:
            if len(state.GetLegalMoves()) == 0:
                ui = FourPlayerChessInterface()
                ui.draw_board_state(state)
                return True, terminal_value_map[GameResult.STALEMATE]

            return False, None

        return True, terminal_value_map[result_state]

    @classmethod
    def get_opponent(cls, player: PlayerColor) -> int:
        """Returns the opponent of the given player."""
        opponent_idx = (color_value(player.GetColor()) + 1) % 4
        return Player(PlayerColor(opponent_idx))

    @staticmethod
    def get_opponent_value(value: float) -> float:
        """Returns the value from the perspective of the opponent."""
        return -value

    # TODO: add attacked squares to the state representation
    @classmethod
    @profile
    def get_encoded_states(cls, states: list, device: str) -> torch.Tensor:
        """Returns an encoded representation of the game state for neural network inputs."""
        batch_size = len(states)
        encoded_states = torch.zeros(
            (batch_size, cls.num_state_channels, cls.board_size, cls.board_size),
            dtype=torch.float32,
            device=device,
        )

        piece_pattern = re.compile(r"(\w+) (\w+) at (\w\d+)")

        for i, state in enumerate(states):
            for piece_list in state.GetPieces():
                for piece in piece_list:
                    color, piece_type, uci_position = piece_pattern.match(
                        str(piece)
                    ).groups()
                    idx = cls._color_to_channel_offset(
                        color
                    ) + cls._piece_to_channel_idx(piece_type)
                    row, col = cls._uci_to_location(uci_position)
                    encoded_states[i, idx, row, col] = 1

        return cls.change_perspective(encoded_states, state.GetTurn())

    @staticmethod
    def change_perspective(tensor: torch.Tensor, player: Player):
        return torch.rot90(tensor, k=color_value(player.GetColor()), dims=[-2, -1])

    @classmethod
    def _color_to_channel_offset(cls, color: str) -> int:
        color_map = {
            "red": PlayerColor.RED,
            "blue": PlayerColor.BLUE,
            "yellow": PlayerColor.YELLOW,
            "green": PlayerColor.GREEN,
        }
        return int(color_value(color_map[color.lower()]) * cls.num_state_channels / 4)

    @classmethod
    def _piece_to_channel_idx(cls, piece_type: str) -> int:
        piece_map = {
            "pawn": PieceType.PAWN,
            "rook": PieceType.ROOK,
            "knight": PieceType.KNIGHT,
            "bishop": PieceType.BISHOP,
            "queen": PieceType.QUEEN,
            "king": PieceType.KING,
        }
        return piece_value(piece_map[piece_type.lower()]) - 1

    @classmethod
    def _uci_to_location(cls, uci):
        col = ord(uci[0]) - ord("a")
        row = cls.board_size - int(uci[1:])
        return row, col
