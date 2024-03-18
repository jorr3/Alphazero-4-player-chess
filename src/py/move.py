import time

import numpy as np
from chessenv import BoardLocation, Move as CppMove, PlayerColor, color_value, Player
from algos import move_parameters_from_index


class Move:
    queen_move_offsets = [
        (0, -1), (-1, -1), (-1, 0), (-1, 1),
        (0, 1), (1, 1), (1, 0), (1, -1),
    ]
    knight_move_offsets = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]

    board_size = 14

    num_queen_moves_per_direction = board_size - 1
    total_queen_moves = len(queen_move_offsets) * num_queen_moves_per_direction

    def __init__(self, from_row, from_col, to_row, to_col, player):
        self.from_row, self.from_col, self.to_row, self.to_col = from_row, from_col, to_row, to_col
        self.player = player

    def get_index(self):
        """Returns the action plane index and the 'from' coordinates in the 3D action space."""
        dy = self.to_row - self.from_row
        dx = self.to_col - self.from_col

        if (dx, dy) in self.knight_move_offsets:
            offset_index = self.knight_move_offsets.index((dx, dy))
            action_plane_index = len(self.queen_move_offsets) * (self.board_size - 1) + offset_index
        else:
            direction = (np.sign(dx), np.sign(dy))
            direction_index = self.queen_move_offsets.index(direction)
            distance = max(abs(dx), abs(dy)) - 1
            action_plane_index = direction_index * (self.board_size - 1) + distance

        return action_plane_index, self.from_row, self.from_col

    def get_flat_index(self):
        """Calculate the flat index of the move in a 112x14x14 action space."""
        dy = self.to_row - self.from_row
        dx = self.to_col - self.from_col
        direction = (np.sign(dx), np.sign(dy))

        if direction in self.queen_move_offsets:
            distance = max(abs(dx), abs(dy)) - 1
            direction_idx = self.queen_move_offsets.index(direction)
            move_type_index = direction_idx * (self.board_size - 1) + distance
        elif (dx, dy) in self.knight_move_offsets:
            direction_idx = self.knight_move_offsets.index((dx, dy))
            # For knight moves, the index starts after all queen move planes
            move_type_index = len(self.queen_move_offsets) * (self.board_size - 1) + direction_idx
        else:
            raise ValueError("Invalid move coordinates for the action space.")

        flat_index = move_type_index * self.board_size ** 2 + self.from_row * self.board_size + self.from_col
        return flat_index

    @classmethod
    def from_cpp(cls, cpp_move, player):
        from_row, from_col = cpp_move.From().GetRow(), cpp_move.From().GetCol()
        to_row, to_col = cpp_move.To().GetRow(), cpp_move.To().GetCol()
        return cls(from_row, from_col, to_row, to_col, player)

    @classmethod
    def from_index(cls, action_plane, from_row, from_col, player):
        if action_plane < cls.total_queen_moves:
            # Queen move
            direction_idx, distance = divmod(action_plane, cls.num_queen_moves_per_direction)
            delta_col, delta_row = cls.queen_move_offsets[direction_idx]
            to_row = from_row + delta_row * (distance + 1)
            to_col = from_col + delta_col * (distance + 1)
        else:
            # Knight move
            knight_move_idx = action_plane - cls.total_queen_moves
            delta_col, delta_row = cls.knight_move_offsets[knight_move_idx]
            to_row = from_row + delta_row
            to_col = from_col + delta_col

        return cls(from_row, from_col, to_row, to_col, player)

    # @classmethod
    # def from_index(cls, action_plane, from_row, from_col, player):
    #     # Call the C++ function to get the move parameters
    #     from_row, from_col, to_row, to_col = move_parameters_from_index(action_plane, from_row, from_col)
    #
    #     # Instantiate and return the class instance with the calculated positions
    #     return cls(from_row, from_col, to_row, to_col, player)

    @classmethod
    def from_flat_index(cls, move_index, player):
        num_queen_moves_per_direction = cls.board_size - 1
        total_queen_moves = len(cls.queen_move_offsets) * num_queen_moves_per_direction

        move_type, pos = divmod(move_index, cls.board_size ** 2)
        from_row, from_col = divmod(pos, cls.board_size)

        if move_type < total_queen_moves:
            direction_idx, distance = divmod(move_type, num_queen_moves_per_direction)
            delta_col, delta_row = cls.queen_move_offsets[direction_idx]
            to_row = from_row + delta_row * (distance + 1)
            to_col = from_col + delta_col * (distance + 1)
        else:
            knight_move_idx = move_type - total_queen_moves
            delta_col, delta_row = cls.knight_move_offsets[knight_move_idx]
            to_row = from_row + delta_row
            to_col = from_col + delta_col

        return cls(from_row, from_col, to_row, to_col, player)

    
    def to_cpp(self):
        """Converts to C++ Move object."""
        start_location = BoardLocation(self.from_row, self.from_col)
        end_location = BoardLocation(self.to_row, self.to_col)
        return CppMove(start_location, end_location)

    def __str__(self):
        from_col_letter = chr(self.from_col + ord('a'))
        to_col_letter = chr(self.to_col + ord('a'))
        from_row_number = self.board_size - self.from_row
        to_row_number = self.board_size - self.to_row
        return f"Move: {from_col_letter}{from_row_number} -> {to_col_letter}{to_row_number}"

