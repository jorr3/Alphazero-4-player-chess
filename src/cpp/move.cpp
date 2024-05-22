#include "move.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fpchess
{
    const std::vector<std::pair<int, int>> Move::queen_move_offsets = {
        {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}};
    const std::vector<std::pair<int, int>> Move::knight_move_offsets = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
    const int Move::num_queen_moves_per_direction = Move::board_size - 1;
    const int Move::num_queen_moves = Move::queen_move_offsets.size() * Move::num_queen_moves_per_direction;
    const int Move::num_knight_moves = Move::knight_move_offsets.size();

    Move::Move(int action_plane, chess::BoardLocation from)
        : chess::Move(from, chess::BoardLocation::kNoLocation)
    {
        if (action_plane < num_queen_moves)
        {
            auto direction_idx = action_plane / num_queen_moves_per_direction;
            auto distance = action_plane % num_queen_moves_per_direction;
            auto [delta_col, delta_row] = queen_move_offsets[direction_idx];
            to_ = from_.Relative(delta_row * (distance + 1), delta_col * (distance + 1));
        }
        else
        {
            auto knight_move_idx = action_plane - num_queen_moves;
            auto [delta_col, delta_row] = knight_move_offsets[knight_move_idx];
            to_ = from_.Relative(delta_row, delta_col);
        }
    }

    Move::Move(int flat_index)
    {
        int total_positions = board_size * board_size;
        int move_type = flat_index / total_positions;
        int pos = flat_index % total_positions;
        from_ = chess::BoardLocation(pos / board_size, pos % board_size);

        if (move_type < num_queen_moves)
        {
            int direction_idx = move_type / num_queen_moves_per_direction;
            int distance = move_type % num_queen_moves_per_direction;
            auto [delta_col, delta_row] = queen_move_offsets[direction_idx];
            to_ = from_.Relative(delta_row * (distance + 1), delta_col * (distance + 1));
        }
        else
        {
            int knight_move_idx = move_type - num_queen_moves;
            auto [delta_col, delta_row] = knight_move_offsets[knight_move_idx];
            to_ = from_.Relative(delta_row, delta_col);
        }
    }

    std::tuple<int, int, int> Move::GetIndex() const
    {
        int dx = to_.GetCol() - from_.GetCol();
        int dy = to_.GetRow() - from_.GetRow();
        auto direction = CalculateDirection(dx, dy);
        int offset_index, action_plane_index;

        if (std::find(knight_move_offsets.begin(), knight_move_offsets.end(), std::make_pair(dx, dy)) != knight_move_offsets.end())
        {
            offset_index = IndexOfMoveOffset(knight_move_offsets, {dx, dy});
            action_plane_index = num_queen_moves + offset_index;
        }
        else
        {
            int direction_index = IndexOfMoveOffset(queen_move_offsets, direction);
            int distance = std::max(std::abs(dx), std::abs(dy)) - 1;
            action_plane_index = direction_index * num_queen_moves_per_direction + distance;
        }

        return {action_plane_index, from_.GetRow(), from_.GetCol()};
    }

    int Move::GetFlatIndex() const
    {
        auto [action_plane_index, _, __] = GetIndex();
        return action_plane_index * (board_size * board_size) + from_.GetRow() * board_size + from_.GetCol();
    }

    std::ostream &operator<<(std::ostream &os, const Move &move)
    {
        os << "Move: " << move.from_ << " -> " << move.to_;
        return os;
    }

    int Move::Sign(int value)
    {
        return (value > 0) - (value < 0);
    }

    int Move::IndexOfMoveOffset(const std::vector<std::pair<int, int>> &offsets, std::pair<int, int> offset)
    {
        auto it = std::find(offsets.begin(), offsets.end(), offset);
        if (it != offsets.end())
        {
            return std::distance(offsets.begin(), it);
        }
        throw std::invalid_argument("Offset not found in move offsets.");
    }

    std::pair<int, int> Move::CalculateDirection(int dx, int dy)
    {
        return {Sign(dx), Sign(dy)};
    }
}
