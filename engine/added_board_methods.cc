#include "board.h"

namespace chess
{
    constexpr int kMateValue = 100000000;

    bool Board::IsMoveLegal(const Move &move)
    {
        MoveBuffer moveBuffer;
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

        MakeMove(move);
        bool isKingInCheckAfterMove = IsKingInCheck(turn_);
        UndoMove();

        return !isKingInCheckAfterMove;
    }

    std::vector<Move> Board::GetLegalMoves()
    {
        std::vector<Move> legalMoves;
        chess::Player TURN = turn_;

        MoveBuffer moveBuffer;
        moveBuffer.buffer = move_buffer_2_;
        moveBuffer.limit = move_buffer_size_;
        moveBuffer.pos = 0;
        size_t numPseudoLegalMoves = GetPseudoLegalMoves2(moveBuffer.buffer, moveBuffer.limit);

        for (size_t i = 0; i < numPseudoLegalMoves; ++i)
        {
            Move move = moveBuffer.buffer[i];

            MakeMove(move);
            bool isKingInCheckAfterMove = IsKingInCheck(TURN);
            UndoMove();

            if (!isKingInCheckAfterMove)
            {
                legalMoves.push_back(move);
            }
        }

        return legalMoves;
    }

    std::tuple<int, std::optional<Move>, std::string> Board::Eval(IAlphaBetaPlayer &player, EvaluationOptions options)
    {
        using namespace std::chrono;

        std::string dbg_str = "";
        GameResult game_result = GetGameResult();

        if (game_result != IN_PROGRESS)
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

    std::unordered_map<PlayerColor, std::vector<BoardLocation>> Board::GetAttackedSquares() const
    {
        std::unordered_map<PlayerColor, std::vector<BoardLocation>> attackedSquares;

        for (int color = RED; color <= GREEN; ++color)
        {
            PlayerColor playerColor = static_cast<PlayerColor>(color);
            for (int row = 0; row < 14; ++row)
            {
                for (int col = 0; col < 14; ++col)
                {
                    BoardLocation location(row, col);
                    if (IsAttackedByPlayer(location, playerColor))
                    {
                        attackedSquares[playerColor].push_back(location);
                    }
                }
            }
        }
        return attackedSquares;
    }

    bool Board::IsAttackedByPlayer(const BoardLocation &location, PlayerColor color) const
    {
        // Helper function to check if a location is legal
        auto isLegalPosition = [&](const BoardLocation &loc) -> bool
        {
            // Add your logic here to check if loc is a legal position on the board
            return loc.Present(); // This is a placeholder, replace with your actual logic
        };

        // Check for pawns
        for (const auto &pawnMove : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                     std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            BoardLocation pawnLoc = location.Relative(pawnMove.first, pawnMove.second);
            if (isLegalPosition(pawnLoc) && GetPiece(pawnLoc).GetColor() == color && GetPiece(pawnLoc).GetPieceType() == PAWN)
            {
                if (PawnAttacks(pawnLoc, color, location))
                    return true;
            }
        }

        // Check for knights
        for (const auto &knightMove : {std::make_pair(1, 2), std::make_pair(2, 1), std::make_pair(-1, -2), std::make_pair(-2, -1),
                                       std::make_pair(1, -2), std::make_pair(2, -1), std::make_pair(-1, 2), std::make_pair(-2, 1)})
        {
            BoardLocation knightLoc = location.Relative(knightMove.first, knightMove.second);
            if (isLegalPosition(knightLoc) && GetPiece(knightLoc).GetColor() == color && GetPiece(knightLoc).GetPieceType() == KNIGHT)
                return true;
        }

        // Check for bishops, rooks, and queens
        for (const auto &direction : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                      std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            BoardLocation currentLoc = location;
            while (true)
            {
                currentLoc = currentLoc.Relative(direction.first, direction.second);
                if (!isLegalPosition(currentLoc))
                    break;
                if (GetPiece(currentLoc).Present())
                {
                    if (GetPiece(currentLoc).GetColor() != color)
                        break;

                    PieceType currentPieceType = GetPiece(currentLoc).GetPieceType();
                    if (currentPieceType != BISHOP && currentPieceType != ROOK && currentPieceType != QUEEN)
                        break;

                    if (currentPieceType == BISHOP && (direction.first == 0 || direction.second == 0))
                        break;
                    if (currentPieceType == ROOK && (direction.first != 0 && direction.second != 0))
                        break;
                    return true;
                }
            }
        }

        // Check for kings
        for (const auto &kingMove : {std::make_pair(1, 0), std::make_pair(0, 1), std::make_pair(-1, 0), std::make_pair(0, -1),
                                     std::make_pair(1, 1), std::make_pair(1, -1), std::make_pair(-1, 1), std::make_pair(-1, -1)})
        {
            BoardLocation kingLoc = location.Relative(kingMove.first, kingMove.second);
            if (isLegalPosition(kingLoc) && GetPiece(kingLoc).GetColor() == color && GetPiece(kingLoc).GetPieceType() == KING)
                return true;
        }

        return false;
    }

} // namespace chess
