#include "Node.h"

namespace fpchess
{

    Node::Node(const std::map<std::string, double> &args, std::shared_ptr<Board> state, chess::PlayerColor turn,
               std::weak_ptr<Node> parent, std::shared_ptr<Move> action_taken, double prior, int visit_count)
        : args(args), state(state), turn(turn), parent(parent),
          move_made(action_taken), prior(prior), visit_count(visit_count), value_sum(0)
    {
        simple_state = std::make_shared<chess::SimpleBoardState>(state->GetSimpleState());
    }

    bool Node::IsFullyExpanded() const
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
            double ucb = q + args.at("C") * std::sqrt(std::log(parent_visit_count_sqrt) / (1 + children_visit_counts[i])) * children_priors[i];

            if (ucb > best_ucb)
            {
                best_child_index = static_cast<int>(i);
                best_ucb = ucb;
            }
        }

        // Return the child with the highest UCB value, or nullptr if no such child exists
        return best_child_index != -1 ? children[best_child_index] : nullptr;
    }

    void Node::Expand(const torch::Tensor &policy)
    {
        auto non_zero_indices = torch::nonzero(policy).to(torch::kCPU);

        for (int64_t i = 0; i < non_zero_indices.size(0); ++i)
        {
            auto idx = non_zero_indices[i];
            int action_plane = idx[0].item<int64_t>();
            int from_row = idx[1].item<int64_t>();
            int from_col = idx[2].item<int64_t>();

            double prob = policy[action_plane][from_row][from_col].item<double>();

            auto move = std::make_shared<Move>(action_plane, chess::BoardLocation(from_row, from_col));
            chess::PlayerColor child_turn = Board::GetOpponent(turn);
            auto child_state = state->TakeAction(*move);

            auto child = std::make_shared<Node>(args, child_state, child_turn, shared_from_this(), move, prob);
            children.push_back(child);
        }

        // Remove reference to state to save memory
        state.reset();
    }

    void Node::Backpropagate(double value)
    {
        value_sum += value;
        visit_count += 1;

        if (auto parent_ = parent.lock())
        {
            parent_->Backpropagate(-value);
        }
    }
}
