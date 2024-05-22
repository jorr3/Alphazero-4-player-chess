#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <map>
#include <tuple>
#include <torch/torch.h>
#include "board.h"
#include "move.h"

namespace fpchess
{
    // Forward declaration
    class Node;

    class Board : public chess::Board
    {
    public:
        using chess::Board::Board;

        static constexpr int board_size = 14;
        static constexpr int num_state_channels = 24;
        static constexpr int kMateValue = 100000000;
        static const int state_space_size;
        static const int num_action_channels;
        static const int actionSpaceSize;
        static const std::tuple<int, int, int> action_space_dims;
        static const std::tuple<int, int, int> state_space_dims;
        static const std::string start_fen;
        static const std::map<chess::PlayerColor, int> color_channel_offsets;
        static std::vector<torch::Tensor> batch_indeces_tensor;

        Board(
            chess::Player turn,
            std::unordered_map<chess::BoardLocation, chess::Piece> locationToPiece,
            std::optional<std::unordered_map<chess::Player, chess::CastlingRights>> castlingRights,
            std::optional<chess::EnpassantInitialization> enp);

        std::shared_ptr<Node> GetRoot()
        {
            return root;
        }

        void SetRoot(std::shared_ptr<Node> rootPtr)
        {
            root = rootPtr;
        }

        std::shared_ptr<Node> GetNode()
        {
            return node;
        }

        void SetNode(std::shared_ptr<Node> nodePtr)
        {
            node = nodePtr;
        }

        std::shared_ptr<std::vector<std::tuple<chess::SimpleBoardState, torch::Tensor, chess::PlayerColor>>> GetMemory()
        {
            return memory;
        }

        void AppendToMemory(std::tuple<chess::SimpleBoardState, torch::Tensor, chess::PlayerColor> &memoryItem)
        {
            memory->push_back(memoryItem);
        }

        float GetOpponentValue(float val);
        const chess::SimpleBoardState GetSimpleState() const;
        bool IsMoveLegal(const Move &move);
        std::vector<Move> GetLegalMoves();
        std::tuple<int, std::optional<Move>, std::string> Eval(chess::IAlphaBetaPlayer &player, chess::EvaluationOptions options);
        std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> GetAttackedSquares() const;
        bool IsAttackedByPlayer(const chess::BoardLocation &location, chess::PlayerColor color) const;
        std::tuple<bool, float> GetTerminated();
        std::shared_ptr<Board> TakeAction(const Move &move);

        static chess::PlayerColor GetOpponent(const chess::PlayerColor &color);
        static chess::PlayerColor GetOpponent(const chess::Player &player);
        static torch::Tensor ChangePerspective(const torch::Tensor &tensor, int rotation);
        static torch::Tensor ParseActionspace(const torch::Tensor &actionspaces_1d, const chess::Player &turn);
        static torch::Tensor GetEncodedStates(const std::vector<std::shared_ptr<Board>> &states, const std::string &device);
        static torch::Tensor GetLegalMovesMask(const std::vector<std::shared_ptr<Board>> &states, const std::string &device);

    private:
        std::shared_ptr<Node> root;
        std::shared_ptr<Node> node;
        std::shared_ptr<std::vector<std::tuple<chess::SimpleBoardState, torch::Tensor, chess::PlayerColor>>> memory;

        bool IsKingSafeAfterMove(const Move &move);
        static torch::TensorOptions ConfigureDevice(const std::string &device);
    };

} // namespace fpchess
