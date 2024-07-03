#pragma once

#include <vector>
#include <utility>
#include <tuple>
#include <iostream>
#include <stdexcept>
#include "./engine/board.h"

namespace fpchess
{
    struct pair_hash
    {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2> &pair) const
        {
            auto hash1 = std::hash<T1>{}(pair.first);
            auto hash2 = std::hash<T2>{}(pair.second);
            return hash1 ^ hash2;
        }
    };

    class Move : public chess::Move
    {
    public:
        using chess::Move::Move;

        static constexpr int board_size = 14;
        static const int num_queen_moves_per_direction;
        static const int num_queen_moves;
        static const int num_knight_moves;
        static const std::vector<std::pair<int, int>> queen_move_offsets;
        static const std::vector<std::pair<int, int>> knight_move_offsets;
        static std::unordered_map<std::pair<int, int>, int, pair_hash> moveIndexMap;

        Move(const chess::Move &c_move) : chess::Move(c_move) {}
        Move(int action_plane, chess::BoardLocation from);
        Move(int flat_index);

        std::tuple<int, int, int> GetIndex() const;
        int GetFlatIndex() const;
        static void InitializeMoveIndexMap();

        friend std::ostream &operator<<(std::ostream &os, const Move &move);

    private:
        static int Sign(int value);
        static int IndexOfMoveOffset(const std::vector<std::pair<int, int>> &offsets, std::pair<int, int> offset);
        static std::pair<int, int> CalculateDirection(int dx, int dy);
    };
}
