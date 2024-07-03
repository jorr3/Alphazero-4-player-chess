#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <map>
#include <tuple>
#include <mutex>
#include <thread>
#include <torch/torch.h>
#include "move.h"

namespace fpchess
{
    class Node;
    struct MemoryEntry;

    class Board : public chess::Board
    {
    public:
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

        Board(
            chess::Player turn,
            std::unordered_map<chess::BoardLocation, chess::Piece> locationToPiece,
            std::optional<std::unordered_map<chess::Player, chess::CastlingRights>> castlingRights,
            std::optional<chess::EnpassantInitialization> enp);

        Board(const Board &other);
        Board();

        void CopyFrom(const Board &other);

        std::shared_ptr<Node> GetRootNode()
        {
            return rootNode;
        }

        void SetRootNode(std::shared_ptr<Node> rootPtr)
        {
            rootNode = rootPtr;
        }

        std::shared_ptr<Node> GetNode()
        {
            return node;
        }

        void SetNode(std::shared_ptr<Node> nodePtr)
        {
            node = nodePtr;
        }

        std::shared_ptr<Board> GetRootState()
        {
            if (rootState == nullptr)
            {
                return std::make_shared<Board>(*this);
            }

            return rootState;
        }

        void SetRootState(std::shared_ptr<Board> statePtr)
        {
            rootState = statePtr;
        }

        std::vector<MemoryEntry> &GetMemory()
        {
            if (rootState == nullptr)
            {
                return memory;
            }

            return rootState->memory;
        }

        void AppendToMemory(const MemoryEntry &entry)
        {
            if (rootState == nullptr)
            {
                memory.push_back(entry);
                return;
            }

            rootState->memory.push_back(entry);
        }
        const chess::SimpleBoardState GetSimpleState() const;
        bool IsMoveLegal(const Move &move);
        std::vector<Move> GetLegalMoves();
        std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> GetAttackedSquares() const;
        bool IsAttackedByPlayer(const chess::BoardLocation &location, chess::PlayerColor color) const;
        std::tuple<bool, float> GetTerminated();
        Board *TakeActionRaw(const Move &move);
        std::shared_ptr<Board> TakeAction(const Move &move);

        static float GetOpponentValue(float val);
        static chess::PlayerColor GetOpponent(const chess::PlayerColor &color);
        static chess::PlayerColor GetOpponent(const chess::Player &player);
        static torch::Tensor ChangePerspective(const torch::Tensor &tensor, int rotation);
        static torch::Tensor ParseActionspace(const torch::Tensor &actionspaces_1d, const chess::Player &turn);
        static torch::Tensor GetEncodedStates(const std::vector<std::shared_ptr<Board>> &states, const std::string &device);
        static torch::Tensor GetLegalMovesMask(const std::vector<std::shared_ptr<Board>> &states, const std::string &device,
                                               torch::Tensor &batch_indices_tensor,
                                               torch::Tensor &plane_indices_tensor,
                                               torch::Tensor &row_indices_tensor,
                                               torch::Tensor &col_indices_tensor,
                                               torch::Tensor &legal_moves_masks);
        static std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
        GetLegalMovesIndices(const std::vector<std::vector<std::shared_ptr<Move>>> &legal_moves, size_t num_moves);

    private:
        std::shared_ptr<Node> rootNode;
        std::shared_ptr<Node> node;
        std::shared_ptr<Board> rootState;
        std::vector<MemoryEntry> memory;

        bool IsKingSafeAfterMove(const Move &move);
        static torch::TensorOptions ConfigureDevice(const std::string &device);
        static void InitializeTensors(int size, torch::TensorOptions options);
    };

    struct MemoryEntry
    {
        Board state;
        at::Tensor action;
        chess::PlayerColor color;

        MemoryEntry(const Board &s, const at::Tensor &a, chess::PlayerColor c)
            : state(s), action(a), color(c) {}
    };

    class BoardPool
    {
    public:
        BoardPool(size_t poolSize)
            : poolSize(poolSize) {}

        // Destructor no longer needs to manually delete Boards
        ~BoardPool() = default;

        std::shared_ptr<Board> acquire(const Board &templateBoard)
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (pool.empty())
            {
                std::cout << "pool empty!!" << std::endl;
                refillPool();
            }

            std::shared_ptr<Board> board = pool.back();
            pool.pop_back();
            board->CopyFrom(templateBoard);

            return board;
        }

        void release(std::shared_ptr<Board> board)
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            pool.push_back(board);
        }

    private:
        size_t poolSize;
        std::vector<std::shared_ptr<Board>> pool;
        std::mutex pool_mutex;

        void refillPool()
        {
            auto thread_count = std::thread::hardware_concurrency();
            size_t batch_size = poolSize / thread_count;

            std::vector<std::vector<std::shared_ptr<Board>>> local_pools(thread_count);

            std::vector<std::thread> threads;
            for (size_t t = 0; t < thread_count; ++t)
            {
                threads.emplace_back([&local_pools, t, batch_size]()
                                     {
                    for (size_t i = 0; i < batch_size; ++i) {
                        local_pools[t].push_back(std::make_shared<Board>());
                    } });
            }

            for (auto &thread : threads)
            {
                thread.join();
            }

            // Merge local pools into the main pool
            for (size_t t = 0; t < thread_count; ++t)
            {
                pool.insert(pool.end(), local_pools[t].begin(), local_pools[t].end());
            }

            // Handle any remainder if poolSize is not perfectly divisible by thread count
            for (size_t i = thread_count * batch_size; i < poolSize; ++i)
            {
                pool.push_back(std::make_shared<Board>());
            }
        }
    };

} // namespace fpchess
