#include "node.h"
#include "board.h"
#include <tuple>
#include <iostream>
#include <memory>

namespace fpchess
{
    const int Board::state_space_size = num_state_channels * chess::rows_ * chess::cols_;
    // TODO: Remove the hard coded values!
    const int Board::num_action_channels = 4 * chess::rows_ + 4 * chess::cols_ + 8;
    const int Board::actionSpaceSize = num_action_channels * chess::rows_ * chess::cols_;
    const std::tuple<int, int, int> Board::action_space_dims = {num_action_channels, chess::rows_, chess::cols_};
    const std::tuple<int, int, int> Board::state_space_dims = {num_state_channels, chess::rows_, chess::cols_};

    Board::Board(chess::Player turn,
                 std::unordered_map<chess::BoardLocation, chess::Piece> location_to_piece,
                 std::optional<std::unordered_map<chess::Player, chess::CastlingRights>> castling_rights = std::nullopt,
                 std::shared_ptr<Board> rootState = nullptr)
        : chess::Board(turn, std::move(location_to_piece), castling_rights),
          rootNode(),
          rootState(rootState)
    {
    }

    // Board::Board(const Board &other)
    //     : chess::Board(other),
    //       rootNode(other.rootNode),
    //       rootState(other.rootState)
    // {
    // }

    Board::Board()
        : chess::Board(chess::Player(), {}, std::nullopt)
    {
    }

    void Board::CopyFrom(const Board &other)
    {
        chess::Board::operator=(other);

        rootNode = other.rootNode;
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
            GetAttackedSquaresPlayers()};
    }

    bool Board::IsKingSafeAfterMove(const Move &move)
    {
        // MakeMove assigns turn_ to the next player so we store the current turn beforehand
        chess::Player currentTurn = turn_;
        MakeMove(move);
        bool isSafe = !IsKingInCheck(currentTurn);
        UndoMove();
        return isSafe;
        // return DiscoversCheck(GetKingLocation(turn_.GetColor()), move.From(), move.To(), turn_.GetTeam());
    }

    bool Board::IsMoveLegal(const Move &move)
    {
        Move buffer[300];
        chess::MoveBuffer moveBuffer;
        moveBuffer.buffer = buffer;
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

        Move buffer[300];
        chess::MoveBuffer moveBuffer;
        moveBuffer.buffer = buffer;
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

    std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> Board::GetAttackedSquaresPlayers() const
    {
        std::unordered_map<chess::PlayerColor, std::vector<chess::BoardLocation>> attackedSquares;

        for (int color = chess::RED; color <= chess::GREEN; ++color)
        {
            chess::PlayerColor playerColor = static_cast<chess::PlayerColor>(color);
            for (int row = 0; row < nRows(); ++row)
            {
                for (int col = 0; col < nCols(); ++col)
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

    std::unordered_map<chess::Team, std::vector<chess::BoardLocation>> Board::GetAttackedSquaresTeams() const
    {
        std::unordered_map<chess::Team, std::vector<chess::BoardLocation>> attackedSquares;

        for (int team = 0; team < chess::TEAM_COUNT; ++team)
        {
            chess::Team currentTeam = static_cast<chess::Team>(team);
            for (int row = 0; row < nRows(); ++row)
            {
                for (int col = 0; col < nCols(); ++col)
                {
                    chess::BoardLocation location(row, col);
                    if (IsAttackedByTeam(currentTeam, location))
                    {
                        attackedSquares[currentTeam].push_back(location);
                    }
                }
            }
        }
        return attackedSquares;
    }

    std::shared_ptr<Board> Board::TakeAction(const Move &move)
    {
        std::shared_ptr<Board> result = std::make_shared<Board>(*this);
        result->MakeMove(move);
        return result;
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

    std::map<chess::PlayerColor, int> Board::GenerateColorChannelOffsets(chess::PlayerColor color)
    {
        std::vector<chess::PlayerColor> colors = {chess::PlayerColor::RED, chess::PlayerColor::BLUE, chess::PlayerColor::YELLOW, chess::PlayerColor::GREEN};
        std::rotate(colors.begin(), std::find(colors.begin(), colors.end(), color), colors.end());

        std::map<chess::PlayerColor, int> offsets;
        for (int i = 0; i < colors.size(); ++i)
        {
            offsets[colors[i]] = i * 6;
        }
        return offsets;
    }

    torch::Tensor Board::GetEncodedState(const Board state, const std::string &device)
    {
        std::vector<std::shared_ptr<Board>> state_vec;
        state_vec.push_back(std::make_shared<Board>(state));
        return GetEncodedStates(state_vec, device);
    }

    torch::Tensor Board::GetEncodedStates(const std::vector<std::shared_ptr<Board>> &states, const std::string &device)
    {
        int batch_size = states.size();
        auto tensorOpts = ConfigureDevice(device);

        torch::Tensor encoded_states = torch::zeros({batch_size, num_state_channels, chess::rows_, chess::cols_}, tensorOpts);

        // Preparing indices for batch operations
        std::vector<int64_t> batch_indices;
        std::vector<int64_t> channel_indices;
        std::vector<int64_t> row_indices;
        std::vector<int64_t> col_indices;

        for (int i = 0; i < batch_size; ++i)
        {
            auto &state = states[i];
            auto placed_pieces = state->GetPieces();
            auto color_channel_offsets = GenerateColorChannelOffsets(state->GetTurn().GetColor());

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

    torch::Tensor Board::GetLegalMovesMask(const std::vector<std::shared_ptr<Board>> &states, const std::string &device,
                                           torch::Tensor &batch_indices_tensor,
                                           torch::Tensor &plane_indices_tensor,
                                           torch::Tensor &row_indices_tensor,
                                           torch::Tensor &col_indices_tensor,
                                           torch::Tensor &legal_moves_masks)
    {
        int batch_size = states.size();
        torch::TensorOptions options = ConfigureDevice(device);

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

        // Ensure the tensors are large enough to hold the indices
        size_t num_moves = batch_indices.size();
        if (batch_indices_tensor.size(0) < num_moves)
        {
            batch_indices_tensor = torch::zeros({(int64_t)num_moves}, torch::kInt64).to(device);
            plane_indices_tensor = torch::zeros({(int64_t)num_moves}, torch::kInt64).to(device);
            row_indices_tensor = torch::zeros({(int64_t)num_moves}, torch::kInt64).to(device);
            col_indices_tensor = torch::zeros({(int64_t)num_moves}, torch::kInt64).to(device);
        }

        // Copy the indices data to the tensors using direct data access
        auto batch_indices_data = batch_indices_tensor.data_ptr<int64_t>();
        auto plane_indices_data = plane_indices_tensor.data_ptr<int64_t>();
        auto row_indices_data = row_indices_tensor.data_ptr<int64_t>();
        auto col_indices_data = col_indices_tensor.data_ptr<int64_t>();

        for (size_t i = 0; i < num_moves; ++i)
        {
            batch_indices_data[i] = batch_indices[i];
            plane_indices_data[i] = plane_indices[i];
            row_indices_data[i] = row_indices[i];
            col_indices_data[i] = col_indices[i];
        }

        // Reset the legal_moves_masks tensor
        legal_moves_masks.zero_();

        // Setting the corresponding positions in the mask tensor to 1
        legal_moves_masks.index_put_(
            {batch_indices_tensor.slice(0, 0, num_moves),
             plane_indices_tensor.slice(0, 0, num_moves),
             row_indices_tensor.slice(0, 0, num_moves),
             col_indices_tensor.slice(0, 0, num_moves)},
            torch::tensor(1, options).to(device));

        return legal_moves_masks;
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
    Board::GetLegalMovesIndices(const std::vector<std::vector<std::shared_ptr<Move>>> &legal_moves, size_t num_moves)
    {
        std::vector<int64_t> batch_indices(num_moves);
        std::vector<int64_t> plane_indices(num_moves);
        std::vector<int64_t> row_indices(num_moves);
        std::vector<int64_t> col_indices(num_moves);

        size_t index = 0;
        for (size_t batch_index = 0; batch_index < legal_moves.size(); ++batch_index)
        {
            for (const auto &move : legal_moves[batch_index])
            {
                int action_plane, from_row, from_col;
                std::tie(action_plane, from_row, from_col) = move->GetIndex();

                batch_indices[index] = batch_index;
                plane_indices[index] = action_plane;
                row_indices[index] = from_row;
                col_indices[index] = from_col;
                ++index;
            }
        }

        return {batch_indices, plane_indices, row_indices, col_indices};
    }
}
