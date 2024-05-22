#include "board.h"
#include <tuple>
#include <iostream>
#include <memory>

namespace fpchess
{
    const int Board::state_space_size = num_state_channels * board_size * board_size;
    // TODO: Remove the hard coded values!
    const int Board::num_action_channels = 104 + 8;
    const int Board::actionSpaceSize = num_action_channels * board_size * board_size;
    const std::tuple<int, int, int> Board::action_space_dims = {num_action_channels, board_size, board_size};
    const std::tuple<int, int, int> Board::state_space_dims = {num_state_channels, board_size, board_size};
    const std::string Board::start_fen = "R-0,0,0,0-1,1,1,1-1,1,1,1-0,0,0,0-0-x,x,x,yR,yN,yB,yK,yQ,yB,yN,yR,x,x,x/x,x,x,yP,yP,yP,yP,yP,yP,yP,yP,x,x,x/x,x,x,8,x,x,x/bR,bP,10,gP,gR/bN,bP,10,gP,gN/bB,bP,10,gP,gB/bQ,bP,10,gP,gK/bK,bP,10,gP,gQ/bB,bP,10,gP,gB/bN,bP,10,gP,gN/bR,bP,10,gP,gR/x,x,x,8,x,x,x/x,x,x,rP,rP,rP,rP,rP,rP,rP,rP,x,x,x/x,x,x,rR,rN,rB,rQ,rK,rB,rN,rR,x,x,x";

    const std::map<chess::PlayerColor, int> Board::color_channel_offsets = {
        {chess::PlayerColor::RED, 0},
        {chess::PlayerColor::BLUE, 6},
        {chess::PlayerColor::YELLOW, 12},
        {chess::PlayerColor::GREEN, 18}};

    Board::Board(chess::Player turn,
                 std::unordered_map<chess::BoardLocation, chess::Piece> location_to_piece,
                 std::optional<std::unordered_map<chess::Player, chess::CastlingRights>> castling_rights = std::nullopt,
                 std::optional<chess::EnpassantInitialization> enp = std::nullopt)
        : chess::Board(turn, std::move(location_to_piece), castling_rights, enp),
          root(),
          node(),
          memory(std::make_shared<std::vector<std::tuple<chess::SimpleBoardState, torch::Tensor, chess::PlayerColor>>>())
    {
    }

    float Board::GetOpponentValue(float val)
    {
        return -val;
    }

    const chess::SimpleBoardState Board::GetSimpleState() const
    {
        return chess::SimpleBoardState{
            GetTurn(),
            GetPieces(),
            GetCastlingRights(),
            GetEnpassantInitialization(),
            GetAttackedSquares()};
    }

    bool Board::IsKingSafeAfterMove(const Move &move)
    {
        // MakeMove assigns turn_ to the next player so we store the current turn beforehand
        chess::Player currentTurn = turn_;
        MakeMove(move);
        bool isSafe = !IsKingInCheck(currentTurn);
        UndoMove();
        return isSafe;
    }

    bool Board::IsMoveLegal(const Move &move)
    {
        chess::MoveBuffer moveBuffer;
        moveBuffer.buffer = move_buffer_2_;
        moveBuffer.limit = move_buffer_size_;
        GetPseudoLegalMoves2(moveBuffer.buffer, moveBuffer.limit);
        bool isPseudoLegal = false;
        for (size_t i = 0; i < moveBuffer.pos; ++i)
        {
            if (moveBuffer.buffer[i] == move)
            {
                isPseudoLegal = true;
                break;
            }
        }
        if (!isPseudoLegal)
        {
            return false;
        }

        return IsKingSafeAfterMove(move);
    }

    std::vector<Move> Board::GetLegalMoves()
    {
        std::vector<Move> legalMoves;
        chess::Player TURN = turn_;

        chess::MoveBuffer moveBuffer;
        moveBuffer.buffer = move_buffer_2_;
        moveBuffer.limit = move_buffer_size_;
        moveBuffer.pos = 0;
        size_t numPseudoLegalMoves = GetPseudoLegalMoves2(moveBuffer.buffer, moveBuffer.limit);

        for (size_t i = 0; i < numPseudoLegalMoves; ++i)
        {
            chess::Move &c_move = moveBuffer.buffer[i];
            Move move(c_move);

            if (IsKingSafeAfterMove(move))
            {
                legalMoves.push_back(move);
            }
        }

        return legalMoves;
    }

    std::tuple<int, std::optional<Move>, std::string> Board::Eval(chess::IAlphaBetaPlayer &player, chess::EvaluationOptions options)
    {
        using namespace std::chrono;

        std::string dbg_str = "";
        chess::GameResult game_result = GetGameResult();

        if (game_result != chess::IN_PROGRESS)
        {
            dbg_str += "Game not in progress.";
            return {-1, std::nullopt, dbg_str};
        }

        auto start = system_clock::now();
        std::optional<Move> best_move;
        std::optional<time_point<system_clock>> deadline;

        if (options.timelimit.has_value())
        {
            deadline = start + milliseconds(options.timelimit.value());
        }

        int depth = 1;
        int score_centipawn = 0;
        while (depth < 100)
        {
            std::optional<milliseconds> time_limit;

            if (deadline.has_value())
            {
                time_limit = duration_cast<milliseconds>(deadline.value() - system_clock::now());
            }

            auto res = player.MakeMove(*this, time_limit, depth);

            if (res.has_value())
            {
                auto duration_ms = duration_cast<milliseconds>(system_clock::now() - start);
                score_centipawn = std::get<0>(res.value());

                best_move = std::get<1>(res.value());
                if (std::abs(score_centipawn) == kMateValue)
                {
                    break;
                }
            }
            else
            {
                break;
            }

            depth++;
        }

        dbg_str += "Depth: " + std::to_string(depth) + "\n";

        return {score_centipawn, best_move, dbg_str};
    }

    std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> Board::GetAttackedSquares() const
    {
        std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> attackedSquares;

        for (int color = chess::RED; color <= chess::GREEN; ++color)
        {
            chess::PlayerColor playerColor = static_cast<chess::PlayerColor>(color);
            for (int row = 0; row < 14; ++row)
            {
                for (int col = 0; col < 14; ++col)
                {
                    chess::BoardLocation location(row, col);
                    if (IsAttackedByPlayer(location, playerColor))
                    {
                        attackedSquares[playerColor].push_back(location);
                    }
                }
            }
        }
        return attackedSquares;
    }

    bool Board::IsAttackedByPlayer(const chess::BoardLocation &location, chess::PlayerColor color) const
    {
        // Helper function to check if a location is legal
        auto isLegalPosition = [&](const chess::BoardLocation &loc) -> bool
        {
            // Add your logic here to check if loc is a legal position on the board
            return loc.Present(); // This is a placeholder, replace with your actual logic
        };

        // Check for pawns
        for (const auto &pawnMove : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                     std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            chess::BoardLocation pawnLoc = location.Relative(pawnMove.first, pawnMove.second);
            if (isLegalPosition(pawnLoc) && GetPiece(pawnLoc).GetColor() == color && GetPiece(pawnLoc).GetPieceType() == chess::PAWN)
            {
                if (PawnAttacks(pawnLoc, color, location))
                    return true;
            }
        }

        // Check for knights
        for (const auto &knightMove : {std::make_pair(1, 2), std::make_pair(2, 1), std::make_pair(-1, -2), std::make_pair(-2, -1),
                                       std::make_pair(1, -2), std::make_pair(2, -1), std::make_pair(-1, 2), std::make_pair(-2, 1)})
        {
            chess::BoardLocation knightLoc = location.Relative(knightMove.first, knightMove.second);
            if (isLegalPosition(knightLoc) && GetPiece(knightLoc).GetColor() == color && GetPiece(knightLoc).GetPieceType() == chess::KNIGHT)
                return true;
        }

        // Check for bishops, rooks, and queens
        for (const auto &direction : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                      std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            chess::BoardLocation currentLoc = location;
            while (true)
            {
                currentLoc = currentLoc.Relative(direction.first, direction.second);
                if (!isLegalPosition(currentLoc))
                    break;
                if (GetPiece(currentLoc).Present())
                {
                    if (GetPiece(currentLoc).GetColor() != color)
                        break;

                    chess::PieceType currentPieceType = GetPiece(currentLoc).GetPieceType();
                    if (currentPieceType != chess::BISHOP && currentPieceType != chess::ROOK && currentPieceType != chess::QUEEN)
                        break;

                    if (currentPieceType == chess::BISHOP && (direction.first == 0 || direction.second == 0))
                        break;
                    if (currentPieceType == chess::ROOK && (direction.first != 0 && direction.second != 0))
                        break;
                    return true;
                }
            }
        }

        // Check for kings
        for (const auto &kingMove : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                     std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            chess::BoardLocation kingLoc = location.Relative(kingMove.first, kingMove.second);
            if (isLegalPosition(kingLoc) && GetPiece(kingLoc).GetColor() == color && GetPiece(kingLoc).GetPieceType() == chess::KING)
                return true;
        }

        return false;
    }

    std::tuple<bool, float> Board::GetTerminated()
    {
        chess::PlayerColor lastPlayer = GetTurn().GetColor();
        chess::GameResult result = GetGameResult();

        if (result == chess::IN_PROGRESS)
        {
            return std::make_tuple(false, 0);
        }

        float terminalValue = 0.0f;
        switch (result)
        {
        case chess::STALEMATE:
            terminalValue = 0.0f;
            break;
        case chess::WIN_RY:
            terminalValue = (lastPlayer == chess::RED || lastPlayer == chess::YELLOW) ? 1 : -1;
            break;
        case chess::WIN_BG:
            terminalValue = (lastPlayer == chess::BLUE || lastPlayer == chess::GREEN) ? 1 : -1;
            break;
        default:
            break;
        }

        return std::make_tuple(true, terminalValue);
    }

    std::shared_ptr<Board> Board::TakeAction(const Move &move)
    {
        Board boardCopy = *this;
        boardCopy.MakeMove(move);
        return std::make_shared<Board>(std::move(boardCopy));
    }

    chess::PlayerColor Board::GetOpponent(const chess::PlayerColor &color)
    {
        int opponentColorVal = (static_cast<int>(color) + 1) % 4;
        return chess::PlayerColor(opponentColorVal);
    }

    chess::PlayerColor Board::GetOpponent(const chess::Player &player)
    {
        return Board::GetOpponent(player.GetColor());
    }

    torch::Tensor Board::ChangePerspective(const torch::Tensor &tensor, int rotation)
    {
        return torch::rot90(tensor, rotation, {-2, -1});
    }

    torch::Tensor Board::ParseActionspace(const torch::Tensor &actionspaces_1d, const chess::Player &turn)
    {
        // Reshape the 1D tensor into a 3D tensor based on class static member action_space_dims
        torch::Tensor actionspaces_3d = actionspaces_1d.view({-1, std::get<0>(action_space_dims), std::get<1>(action_space_dims), std::get<2>(action_space_dims)});
        int rotation = -static_cast<int>(turn.GetColor());
        return ChangePerspective(actionspaces_3d, rotation);
    }

    torch::TensorOptions Board::ConfigureDevice(const std::string &device)
    {
        torch::TensorOptions options;

        if (device == "cpu")
        {
            options = options.device(torch::kCPU);
        }
        else if (device == "gpu" | device == "cuda")
        {
            options = options.device(torch::kCUDA);
        }
        else
        {
            throw std::invalid_argument("Invalid device argument.");
        }

        return options;
    }

    torch::Tensor Board::GetEncodedStates(const std::vector<std::shared_ptr<Board>> &states, const std::string &device)
    {
        int batch_size = states.size();
        auto tensorOpts = ConfigureDevice(device);

        torch::Tensor encoded_states = torch::zeros({batch_size, num_state_channels, board_size, board_size}, tensorOpts);

        // Preparing indices for batch operations
        std::vector<int64_t> batch_indices;
        std::vector<int64_t> channel_indices;
        std::vector<int64_t> row_indices;
        std::vector<int64_t> col_indices;

        for (int i = 0; i < batch_size; ++i)
        {
            auto &state = states[i];
            auto placed_pieces = state->GetPieces();

            for (const auto &placed_piece_vec : placed_pieces)
            {
                for (const auto &placed_piece : placed_piece_vec)
                {
                    const chess::Piece piece = placed_piece.GetPiece();
                    chess::BoardLocation location = placed_piece.GetLocation();
                    int piece_type_offset = piece.GetPieceType();
                    chess::PlayerColor color = piece.GetColor();

                    int row = location.GetRow();
                    int col = location.GetCol();

                    int plane_idx = color_channel_offsets.at(color) + piece_type_offset - 1;

                    batch_indices.push_back(i);
                    channel_indices.push_back(plane_idx);
                    row_indices.push_back(row);
                    col_indices.push_back(col);
                }
            }
        }

        // Convert indices to tensor
        auto batchIndices = torch::tensor(batch_indices, tensorOpts);
        auto channelIndices = torch::tensor(channel_indices, tensorOpts);
        auto rowIndices = torch::tensor(row_indices, tensorOpts);
        auto colIndices = torch::tensor(col_indices, tensorOpts);

        encoded_states.index_put_({batchIndices, channelIndices, rowIndices, colIndices}, 1);

        int amm_rotations = states[0]->GetTurn().GetColor();
        return ChangePerspective(encoded_states, amm_rotations);
    }

    torch::Tensor Board::GetLegalMovesMask(const std::vector<std::shared_ptr<Board>> &states, const std::string &device)
    {
        int batch_size = states.size();
        torch::TensorOptions options = ConfigureDevice(device);

        const auto [dim0, dim1, dim2] = action_space_dims;
        torch::Tensor legal_moves_masks = torch::zeros({batch_size, dim0, dim1, dim2}, options);

        std::vector<int64_t> batch_indices, plane_indices, row_indices, col_indices;

        for (size_t batch_index = 0; batch_index < batch_size; ++batch_index)
        {
            auto &state = states[batch_index];

            for (const auto &move : state->GetLegalMoves())
            {
                int action_plane, from_row, from_col;
                std::tie(action_plane, from_row, from_col) = move.GetIndex();

                batch_indices.push_back(batch_index);
                plane_indices.push_back(action_plane);
                row_indices.push_back(from_row);
                col_indices.push_back(from_col);
            }
        }

        // Creating tensors from indices
        auto plane_indices_tensor = torch::tensor(plane_indices, options);
        auto row_indices_tensor = torch::tensor(row_indices, options);
        auto col_indices_tensor = torch::tensor(col_indices, options);
        auto batch_indices_tensor = torch::tensor(batch_indices, options);

        // Setting the corresponding positions in the mask tensor to 1
        legal_moves_masks.index_put_({batch_indices_tensor, plane_indices_tensor, row_indices_tensor, col_indices_tensor}, 1);

        return legal_moves_masks;
    }

}
