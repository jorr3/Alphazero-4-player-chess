#pragma once

#include <vector>
#include <utility>
#include <tuple>
#include <iostream>
#include "../../engine/board.h"

namespace fpchess
{

    class Move : public chess::Move
    {
    public:
        using chess::Move::Move;

        static constexpr int board_size = 14;
        static const int num_queen_moves_per_direction;
        static const int num_queen_moves;
        static const int num_knight_moves;

        Move(const chess::Move &c_move) : chess::Move(c_move) {}
        Move(int action_plane, chess::BoardLocation from);
        Move(int flat_index);

        std::tuple<int, int, int> GetIndex() const;
        int GetFlatIndex() const;

        friend std::ostream &operator<<(std::ostream &os, const Move &move);

    private:
        static const std::vector<std::pair<int, int>> queen_move_offsets;
        static const std::vector<std::pair<int, int>> knight_move_offsets;

        static int Sign(int value);
        static int IndexOfMoveOffset(const std::vector<std::pair<int, int>> &offsets, std::pair<int, int> offset);
        static std::pair<int, int> CalculateDirection(int dx, int dy);
    };
}
