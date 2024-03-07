#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <limits>

namespace py = pybind11;

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
}
