#include "move.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <unordered_map>

namespace fpchess
{
    constexpr int rows_ = chess::rows_;
    constexpr int cols_ = chess::cols_;

    const std::vector<std::pair<int, int>> Move::queen_move_offsets = {
        {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}};
    const std::vector<std::pair<int, int>> Move::knight_move_offsets = {
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
    // TODO: num_queen_moves_per_direction should be properly configured
    const int Move::num_queen_moves_per_direction = rows_ - 1;
    const int Move::num_queen_moves = Move::queen_move_offsets.size() * Move::num_queen_moves_per_direction;
    const int Move::num_knight_moves = Move::knight_move_offsets.size();
    std::unordered_map<std::pair<int, int>, int, pair_hash> move_index_map;

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
        int total_positions = rows_ * cols_;
        int move_type = flat_index / total_positions;
        int pos = flat_index % total_positions;
        from_ = chess::BoardLocation(pos / rows_, pos % cols_);

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

    void Move::InitializeMoveIndexMap()
    {
        // Populate for queen moves
        for (int i = 0; i < queen_move_offsets.size(); ++i)
        {
            for (int dist = 1; dist <= num_queen_moves_per_direction; ++dist)
            {
                int dx = queen_move_offsets[i].first * dist;
                int dy = queen_move_offsets[i].second * dist;
                int index = i * num_queen_moves_per_direction + (dist - 1);
                move_index_map[{dx, dy}] = index;
            }
        }
        // Populate for knight moves
        int baseIndex = num_queen_moves;
        for (int i = 0; i < knight_move_offsets.size(); ++i)
        {
            move_index_map[knight_move_offsets[i]] = baseIndex + i;
        }
    }

    std::tuple<int, int, int> Move::GetIndex() const
    {
        int dx = to_.GetCol() - from_.GetCol();
        int dy = to_.GetRow() - from_.GetRow();
        std::pair<int, int> delta = {dx, dy};

        auto it = move_index_map.find(delta);
        if (it != move_index_map.end())
        {
            int action_plane_index = it->second;
            return {action_plane_index, from_.GetRow(), from_.GetCol()};
        }

        throw std::invalid_argument("Invalid move: No corresponding action plane index found. Did you initialize move_index_map?");
    }

    int Move::GetFlatIndex() const
    {
        auto [action_plane_index, _, __] = GetIndex();
        return action_plane_index * (rows_ * cols_) + from_.GetRow() * rows_ + from_.GetCol();
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
