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

from src.igame import IGame
from fen_parser import parse_board_from_fen
from move import Move
from fourpchess_interface import FourPlayerChessInterface

from line_profiler_pycharm import profile


class FourPlayerChess(IGame):
    start_fen = """R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,1,1,yB,yK,yQ,yB,1,1,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/1,bP,10,gP,1/1,bP,10,gP,1/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/1,bP,10,gP,1/1,bP,10,gP,1/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,1,1,rB,rQ,rK,rB,1,1,x,x,x"""
    board_size = 14
    num_state_channels = 24
    state_space_size = num_state_channels * board_size**2
    num_queen_moves = len(Move.queen_move_offsets) * (board_size - 1)
    num_action_channels = num_queen_moves + len(Move.knight_move_offsets)
    action_space_size = num_action_channels * board_size**2
    action_space_dims = (num_action_channels, board_size, board_size)
    state_space_dims = (num_state_channels, board_size, board_size)

    piece_pattern = re.compile(r"(\w+) (\w+) at (\w\d+)")

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
        # This is a check that will be removed in production
        # legal_moves = state.GetLegalMoves()
        # move = str(action.to_cpp())
        # if move not in [str(x) for x in legal_moves]:
        #     print(state)
        #     pass

        state_copy = Board(state)
        state_copy.MakeMove(action.to_cpp())
        return state_copy

    @classmethod
    @profile
    def get_legal_actions_mask(cls, states: list, device: str) -> torch.Tensor:
        """Returns a 4D tensor indicating valid moves for each state in the batch, in a 3D action space."""
        legal_moves_masks = torch.zeros(
            (len(states), *cls.action_space_dims), dtype=torch.float32, device=device
        )

        # Pre-calculate indices for batch updates
        batch_indices, plane_indices, row_indices, col_indices = [], [], [], []

        for batch_index, state in enumerate(states):
            for move_cpp in state.GetLegalMoves():
                move = Move.from_cpp(move_cpp, state.GetTurn())
                action_plane, from_row, from_col = move.get_index()

                batch_indices.append(batch_index)
                plane_indices.append(action_plane)
                row_indices.append(from_row)
                col_indices.append(from_col)

        batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
        plane_indices_tensor = torch.tensor(plane_indices, dtype=torch.long, device=device)
        row_indices_tensor = torch.tensor(row_indices, dtype=torch.long, device=device)
        col_indices_tensor = torch.tensor(col_indices, dtype=torch.long, device=device)


        legal_moves_masks[
            batch_indices_tensor,
            plane_indices_tensor,
            row_indices_tensor,
            col_indices_tensor,
        ] = 1

        return legal_moves_masks

    @classmethod
    def get_terminated(
        cls, state, last_player_to_make_move: Player
    ) -> tuple[bool, None] | tuple[bool, float]:
        """Returns if the game is over and the value of the current state."""
        terminal_value_map = {
            GameResult.STALEMATE: 0.0,
            GameResult.WIN_RY: (
                1.0
                if last_player_to_make_move.GetColor()
                in [PlayerColor.RED, PlayerColor.YELLOW]
                else -1.0
            ),
            GameResult.WIN_BG: (
                1.0
                if last_player_to_make_move.GetColor()
                in [PlayerColor.BLUE, PlayerColor.GREEN]
                else -1.0
            ),
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

        # Batch indexing is used for performance reasons
        batch_indices, channel_indices, row_indices, col_indices = [], [], [], []

        for i, state in enumerate(states):
            for piece_list in state.GetPieces():
                for piece in piece_list:
                    color, piece_type, uci_position = cls.piece_pattern.match(
                        str(piece)
                    ).groups()
                    idx = cls._color_to_channel_offset(
                        color
                    ) + cls._piece_to_channel_idx(piece_type)
                    row, col = cls._uci_to_location(uci_position)

                    batch_indices.append(i)
                    channel_indices.append(idx)
                    row_indices.append(row)
                    col_indices.append(col)

        batch_indices = torch.tensor(batch_indices, device=device)
        channel_indices = torch.tensor(channel_indices, device=device)
        row_indices = torch.tensor(row_indices, device=device)
        col_indices = torch.tensor(col_indices, device=device)

        encoded_states[batch_indices, channel_indices, row_indices, col_indices] = 1
        return encoded_states
        return cls.change_perspective(encoded_states, color_value(player.GetColor()))

    @staticmethod
    def change_perspective(tensor: torch.Tensor, rotation):
        return torch.rot90(tensor, k=rotation, dims=[-2, -1])

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

    @classmethod
    def parse_actionspace(cls, batch_policy_tensors, turn, device):
        reshaped_policies = torch.zeros((len(batch_policy_tensors), *cls.action_space_dims), dtype=torch.float32, device=device)

        for i, policy in enumerate(batch_policy_tensors):
            reshaped_policy = policy.view(FourPlayerChess.action_space_dims)
            reshaped_policies[i] = reshaped_policy

        policy_player_red_perspective = cls.change_perspective(reshaped_policies, -color_value(turn.GetColor()))
        return policy_player_red_perspective
