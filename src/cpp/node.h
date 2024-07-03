#pragma once

#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <limits>

#include <torch/extension.h>

#include "board.h"
#include "move.h"
#include "./engine/board.h"

namespace fpchess
{

    class Node : public std::enable_shared_from_this<Node>
    {
    public:
        Node() = default;
        Node(const double C,
             std::shared_ptr<Board> state,
             chess::PlayerColor turn,
             std::weak_ptr<Node> parent = std::weak_ptr<Node>(),
             std::shared_ptr<Move> action_taken = nullptr,
             double prior = 0.0,
             int visit_count = 1);

        std::shared_ptr<Board> GetState()
        {
            return state;
        }

        std::shared_ptr<chess::SimpleBoardState> GetSimpleState()
        {
            return std::make_shared<chess::SimpleBoardState>(state->GetSimpleState());
        }

        chess::PlayerColor GetTurn()
        {
            return turn;
        }

        std::vector<std::shared_ptr<Node>> GetChildren()
        {
            return children;
        }

        std::shared_ptr<fpchess::Move> GetMoveMade()
        {
            return move_made;
        }

        int GetVisitCount()
        {
            return visit_count;
        }

        void SetVisitCount(int val)
        {
            visit_count = val;
        }

        bool IsExpanded() const;
        std::shared_ptr<Node> SelectChild();
        void Backpropagate(float value);
        static void BackpropagateNodes(const std::vector<std::shared_ptr<Node>> &nodes, const torch::Tensor &values);
        void Expand(const torch::Tensor &policy, const std::vector<int64_t> &action_planes,
                    const std::vector<int64_t> &from_rows, const std::vector<int64_t> &from_cols,
                    const std::vector<double> &probs, BoardPool &pool);

        static void ExpandNodes(std::vector<std::shared_ptr<Node>> &nodes,
                                const torch::Tensor &policy_batch,
                                const std::vector<std::vector<int64_t>> &non_zero_indices_batch,
                                const std::vector<double> &non_zero_values,
                                BoardPool &pool);

    private:
        double C;
        std::shared_ptr<Board> state;
        chess::PlayerColor turn;
        std::weak_ptr<Node> parent;
        std::shared_ptr<Move> move_made;
        double prior;
        int visit_count;
        double value_sum;
        std::vector<std::shared_ptr<Node>> children;
    };
}
