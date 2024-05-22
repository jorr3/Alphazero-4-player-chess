#pragma once

#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <limits>

#include <torch/extension.h>

#include "board.h"
#include "move.h"
#include "../../engine/board.h"

namespace fpchess
{

    class Node : public std::enable_shared_from_this<Node>
    {
    public:
        Node() = default;
        Node(const std::map<std::string, double> &args,
             std::shared_ptr<Board> state,
             chess::PlayerColor turn,
             std::weak_ptr<Node> parent = std::weak_ptr<Node>(),
             std::shared_ptr<Move> action_taken = nullptr,
             double prior = 0.0,
             int visit_count = 0);

        std::shared_ptr<fpchess::Board> GetState()
        {
            if (state == nullptr)
            {
                throw std::runtime_error("Full state not available. Use GetSimpleState() instead.");
            }
            return state;
        }

        std::shared_ptr<chess::SimpleBoardState> GetSimpleState()
        {
            return simple_state;
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

        bool IsFullyExpanded() const;

        std::shared_ptr<Node> SelectChild();

        void Expand(const torch::Tensor &policy);

        void Backpropagate(double value);

    private:
        std::map<std::string, double> args;
        std::shared_ptr<Board> state;
        std::shared_ptr<chess::SimpleBoardState> simple_state;
        chess::PlayerColor turn;
        std::weak_ptr<Node> parent;
        std::shared_ptr<Move> move_made;
        double prior;
        int visit_count;
        double value_sum;
        std::vector<std::shared_ptr<Node>> children;
    };
}
