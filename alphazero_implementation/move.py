import numpy as np
from chessenv import BoardLocation, Move as CppMove, PlayerColor, color_value, Player


class Move:
    queen_move_offsets = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    knight_move_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    def __init__(self, move_index: int, board_size: int, player: Player):
        num_queen_moves_per_direction = board_size - 1
        total_queen_moves = len(self.queen_move_offsets) * num_queen_moves_per_direction
        total_knight_moves = len(self.knight_move_offsets)
        num_distinct_moves = total_queen_moves + total_knight_moves

        if move_index >= num_distinct_moves * board_size * board_size:
            raise ValueError("Move index out of valid range.")

        self.move_index = move_index

        move_type, pos = divmod(move_index, board_size * board_size)
        start_row, start_col = divmod(pos, board_size)

        self.from_row, self.from_col = self.to_default_perspective(start_row, start_col, board_size, player)

        if move_type < total_queen_moves:  # Queen moves
            direction_idx, distance = divmod(move_type, num_queen_moves_per_direction)
            delta_row, delta_col = self.queen_move_offsets[direction_idx]
            end_row = start_row + delta_row * (distance + 1)
            end_col = start_col + delta_col * (distance + 1)
        else:  # Knight moves
            knight_move_idx = move_type - total_queen_moves
            delta_row, delta_col = self.knight_move_offsets[knight_move_idx]
            end_row = start_row + delta_row
            end_col = start_col + delta_col

        self.to_row, self.to_col = self.to_default_perspective(end_row, end_col, board_size, player)

    def to_cpp(self):
        start_location = BoardLocation(self.from_row, self.from_col)
        end_location = BoardLocation(self.to_row, self.to_col)
        return CppMove(start_location, end_location)

    # TODO: Make this function applicable to games outside of 4 player chess.
    @staticmethod
    def to_default_perspective(row, col, board_size, player):
        """
        Convert coordinates to the perspective of player 0.
        """
        # Rotate 90 degrees clockwise for each player color value
        for _ in range(color_value(player.GetColor())):
            row, col = col, board_size - 1 - row
        return row, col

    @staticmethod
    def get_flat_index(from_row, from_col, to_row, to_col, board_size):
        dx = to_row - from_row
        dy = to_col - from_col

        if (dx, dy) in Move.knight_move_offsets:
            move_type = 'knight'
            direction_idx = Move.knight_move_offsets.index((dx, dy))
            distance = 0
        else:
            move_type = 'queen'
            direction = (np.sign(dx), np.sign(dy))
            direction_idx = Move.queen_move_offsets.index(direction)
            distance = max(abs(dx), abs(dy)) - 1

        num_queen_moves_per_direction = board_size - 1
        if move_type == 'queen':
            move_type_index = direction_idx * num_queen_moves_per_direction + distance
        else:
            total_queen_moves = len(Move.queen_move_offsets) * num_queen_moves_per_direction
            move_type_index = total_queen_moves + direction_idx

        flat_index = move_type_index * board_size ** 2 + from_row * board_size + from_col
        return flat_index
