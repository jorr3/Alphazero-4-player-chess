#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace py = pybind11;

const int board_size = 14;
const int num_queen_moves_per_direction = board_size - 1;
const int total_queen_moves = 8 * num_queen_moves_per_direction;

std::vector<std::pair<int, int>> queen_move_offsets = {
    {0, -1},
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, 1},
    {1, 1},
    {1, 0},
    {1, -1},
};

std::vector<std::pair<int, int>> knight_move_offsets = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};

std::tuple<int, int, int, int> move_parameters_from_index(int action_plane, int from_row, int from_col)
{
    int to_row, to_col;

    if (action_plane < total_queen_moves)
    {
        // Queen move
        int direction_idx = action_plane / num_queen_moves_per_direction;
        int distance = action_plane % num_queen_moves_per_direction;
        int delta_col = queen_move_offsets[direction_idx].first;
        int delta_row = queen_move_offsets[direction_idx].second;
        to_row = from_row + delta_row * (distance + 1);
        to_col = from_col + delta_col * (distance + 1);
    }
    else
    {
        // Knight move
        int knight_move_idx = action_plane - total_queen_moves;
        int delta_col = knight_move_offsets[knight_move_idx].first;
        int delta_row = knight_move_offsets[knight_move_idx].second;
        to_row = from_row + delta_row;
        to_col = from_col + delta_col;
    }

    return std::make_tuple(from_row, from_col, to_row, to_col);
}

int select_child(
    const std::vector<int> &children_visit_counts,
    const std::vector<double> &children_values,
    const std::vector<double> &children_priors,
    int parent_visit_count,
    double C)
{

    int best_child_index = -1;
    double best_ucb = -std::numeric_limits<double>::infinity();
    double parent_visit_count_sqrt = std::sqrt(parent_visit_count);

    for (size_t i = 0; i < children_visit_counts.size(); ++i)
    {
        double q = children_visit_counts[i] == 0 ? 0 : 1 - ((children_values[i] / children_visit_counts[i]) + 1) / 2;
        double ucb = q + C * (parent_visit_count_sqrt / (children_visit_counts[i] + 1)) * children_priors[i];

        if (ucb > best_ucb)
        {
            best_child_index = static_cast<int>(i);
            best_ucb = ucb;
        }
    }

    return best_child_index;
}

PYBIND11_MODULE(algos, m)
{
    m.def("select_child", &select_child, "A function to select the child with the highest UCB.");
    m.def("move_parameters_from_index", &move_parameters_from_index);
}
