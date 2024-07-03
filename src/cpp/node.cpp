#include "Node.h"
#include "board.h"
#include <chrono>

namespace fpchess
{

    Node::Node(const double C, std::shared_ptr<Board> state, chess::PlayerColor turn,
               std::weak_ptr<Node> parent, std::shared_ptr<Move> action_taken, double prior, int visit_count)
        : C(C), state(state), turn(turn), parent(parent),
          move_made(action_taken), prior(prior), visit_count(visit_count), value_sum(0)
    {
    }

    bool Node::IsExpanded() const
    {
        return !children.empty();
    }

    std::shared_ptr<Node> Node::SelectChild()
    {
        std::vector<int> children_visit_counts;
        std::vector<double> children_values;
        std::vector<double> children_priors;

        // Gather visit counts, values, and priors for each child
        for (const auto &child : children)
        {
            children_visit_counts.push_back(child->visit_count);
            children_values.push_back(child->visit_count > 0 ? child->value_sum / child->visit_count : 0);
            children_priors.push_back(child->prior);
        }

        int best_child_index = -1;
        double best_ucb = -std::numeric_limits<double>::infinity();
        double parent_visit_count_sqrt = std::sqrt(static_cast<double>(visit_count));

        // Compute the UCB value for each child and find the child with the highest UCB value
        for (size_t i = 0; i < children_visit_counts.size(); ++i)
        {
            double q = children_visit_counts[i] == 0 ? 0 : children_values[i] / children_visit_counts[i];
            double ucb = q + C * std::sqrt(std::log(parent_visit_count_sqrt) / (1 + children_visit_counts[i])) * children_priors[i];

            if (ucb > best_ucb)
            {
                best_child_index = static_cast<int>(i);
                best_ucb = ucb;
            }
        }

        if (best_child_index == -1)
        {
            throw std::runtime_error("Failed to select a child.");
        }

        return children[best_child_index];
    }

    void Node::Expand(const torch::Tensor &policy, const std::vector<int64_t> &action_planes,
                      const std::vector<int64_t> &from_rows, const std::vector<int64_t> &from_cols,
                      const std::vector<double> &probs, BoardPool &pool)
    {
        int64_t num_moves = action_planes.size();

        for (int64_t i = 0; i < num_moves; ++i)
        {
            auto move = std::make_shared<Move>(
                action_planes[i],
                chess::BoardLocation(from_rows[i], from_cols[i]));
            chess::PlayerColor child_turn = Board::GetOpponent(turn);

            std::shared_ptr<Board> child_state = pool.acquire(*state);
            child_state->MakeMove(*move);

            auto child = std::make_shared<Node>(
                C, child_state, child_turn, shared_from_this(), move, probs[i]);
            children.push_back(child);
        }
    }

    void Node::ExpandNodes(std::vector<std::shared_ptr<Node>> &nodes,
                           const torch::Tensor &policy_batch,
                           const std::vector<std::vector<int64_t>> &non_zero_indices_batch,
                           const std::vector<double> &non_zero_values,
                           BoardPool &pool)
    {
        int batch_size = policy_batch.size(0);

        std::vector<std::vector<int64_t>> action_planes_batch(batch_size);
        std::vector<std::vector<int64_t>> from_rows_batch(batch_size);
        std::vector<std::vector<int64_t>> from_cols_batch(batch_size);
        std::vector<std::vector<double>> probs_batch(batch_size);

        for (size_t i = 0; i < non_zero_indices_batch.size(); ++i)
        {
            int batch_index = non_zero_indices_batch[i][0];
            int action_plane = non_zero_indices_batch[i][1];
            int from_row = non_zero_indices_batch[i][2];
            int from_col = non_zero_indices_batch[i][3];
            double prob = non_zero_values[i];

            action_planes_batch[batch_index].push_back(action_plane);
            from_rows_batch[batch_index].push_back(from_row);
            from_cols_batch[batch_index].push_back(from_col);
            probs_batch[batch_index].push_back(prob);
        }

        for (size_t i = 0; i < nodes.size(); ++i)
        {
            nodes[i]->Expand(policy_batch[i], action_planes_batch[i], from_rows_batch[i], from_cols_batch[i], probs_batch[i], pool);
        }
    }

    void Node::Backpropagate(float value)
    {
        value_sum += value;
        visit_count += 1;

        if (auto parent_ = parent.lock())
        {
            parent_->Backpropagate(-value);
        }
    }

    void Node::BackpropagateNodes(const std::vector<std::shared_ptr<Node>> &nodes, const torch::Tensor &values)
    {
        int num_nodes = nodes.size();
        auto values_cpu = values.to(torch::kCPU);
        auto values_acc = values_cpu.accessor<float, 1>();

        for (int i = 0; i < num_nodes; ++i)
        {
            nodes[i]->Backpropagate(values_acc[i]);
        }
    }
}
