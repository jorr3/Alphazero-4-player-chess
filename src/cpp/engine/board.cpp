#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "board.h"

namespace chess
{

  constexpr int kMobilityMultiplier = 5;
  Piece Piece::kNoPiece = Piece();
  BoardLocation BoardLocation::kNoLocation = BoardLocation();
  CastlingRights CastlingRights::kMissingRights = CastlingRights();

  const BoardLocation kRedInitialRookLocationKingside(13, 10);
  const BoardLocation kRedInitialRookLocationQueenside(13, 3);
  const BoardLocation kBlueInitialRookLocationKingside(10, 0);
  const BoardLocation kBlueInitialRookLocationQueenside(3, 0);
  const BoardLocation kYellowInitialRookLocationKingside(0, 3);
  const BoardLocation kYellowInitialRookLocationQueenside(0, 10);
  const BoardLocation kGreenInitialRookLocationKingside(3, 13);
  const BoardLocation kGreenInitialRookLocationQueenside(10, 13);

  const Player kRedPlayer = Player(RED);
  const Player kBluePlayer = Player(BLUE);
  const Player kYellowPlayer = Player(YELLOW);
  const Player kGreenPlayer = Player(GREEN);

  const Piece kRedPawn(kRedPlayer, PAWN);
  const Piece kRedKnight(kRedPlayer, KNIGHT);
  const Piece kRedBishop(kRedPlayer, BISHOP);
  const Piece kRedRook(kRedPlayer, ROOK);
  const Piece kRedQueen(kRedPlayer, QUEEN);
  const Piece kRedKing(kRedPlayer, KING);

  const Piece kBluePawn(kBluePlayer, PAWN);
  const Piece kBlueKnight(kBluePlayer, KNIGHT);
  const Piece kBlueBishop(kBluePlayer, BISHOP);
  const Piece kBlueRook(kBluePlayer, ROOK);
  const Piece kBlueQueen(kBluePlayer, QUEEN);
  const Piece kBlueKing(kBluePlayer, KING);

  const Piece kYellowPawn(kYellowPlayer, PAWN);
  const Piece kYellowKnight(kYellowPlayer, KNIGHT);
  const Piece kYellowBishop(kYellowPlayer, BISHOP);
  const Piece kYellowRook(kYellowPlayer, ROOK);
  const Piece kYellowQueen(kYellowPlayer, QUEEN);
  const Piece kYellowKing(kYellowPlayer, KING);

  const Piece kGreenPawn(kGreenPlayer, PAWN);
  const Piece kGreenKnight(kGreenPlayer, KNIGHT);
  const Piece kGreenBishop(kGreenPlayer, BISHOP);
  const Piece kGreenRook(kGreenPlayer, ROOK);
  const Piece kGreenQueen(kGreenPlayer, QUEEN);
  const Piece kGreenKing(kGreenPlayer, KING);

  namespace
  {

    int64_t rand64()
    {
      int32_t t0 = rand();
      int32_t t1 = rand();
      return (((int64_t)t0) << 32) + (int64_t)t1;
    }

    void AddPawnMoves2(
        MoveBuffer &moves,
        const BoardLocation &from,
        const BoardLocation &to,
        const PlayerColor color,
        const Piece capture = Piece::kNoPiece,
        const BoardLocation en_passant_location = BoardLocation::kNoLocation,
        const Piece en_passant_capture = Piece::kNoPiece)
    {
      bool is_promotion = false;

      constexpr int kRedPromotionRow = 3;
      constexpr int kYellowPromotionRow = 10;
      constexpr int kBluePromotionCol = 10;
      constexpr int kGreenPromotionCol = 3;

      switch (color)
      {
      case RED:
        is_promotion = to.GetRow() == kRedPromotionRow;
        break;
      case BLUE:
        is_promotion = to.GetCol() == kBluePromotionCol;
        break;
      case YELLOW:
        is_promotion = to.GetRow() == kYellowPromotionRow;
        break;
      case GREEN:
        is_promotion = to.GetCol() == kGreenPromotionCol;
        break;
      default:
        assert(false);
        break;
      }

      if (is_promotion)
      {
        moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, KNIGHT);
        moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, BISHOP);
        moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, ROOK);
        moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, QUEEN);
      }
      else
      {
        moves.emplace_back(from, to, capture, en_passant_location, en_passant_capture, NO_PIECE);
      }
    }

  } // namespace

  void Board::GetPawnMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {
    PlayerColor color = piece.GetColor();
    Team team = piece.GetTeam();

    // Move forward
    int delta_rows = 0;
    int delta_cols = 0;
    bool not_moved = false;
    switch (color)
    {
    case RED:
      delta_rows = -1;
      not_moved = from.GetRow() == 12;
      break;
    case BLUE:
      delta_cols = 1;
      not_moved = from.GetCol() == 1;
      break;
    case YELLOW:
      delta_rows = 1;
      not_moved = from.GetRow() == 1;
      break;
    case GREEN:
      delta_cols = -1;
      not_moved = from.GetCol() == 12;
      break;
    default:
      assert(false);
      break;
    }

    BoardLocation to = from.Relative(delta_rows, delta_cols);
    if (IsLegalLocation(to))
    {
      Piece other_piece = GetPiece(to);
      if (other_piece.Missing())
      {
        // Advance once square
        AddPawnMoves2(moves, from, to, piece.GetColor());
        // Initial move (advance 2 squares)
        if (not_moved)
        {
          to = from.Relative(delta_rows * 2, delta_cols * 2);
          other_piece = GetPiece(to);
          if (other_piece.Missing())
          {
            AddPawnMoves2(moves, from, to, piece.GetColor());
          }
        }
      } // NO EN PASSANT
      // else
      // {

      //   // En-passant
      //   if (other_piece.GetPieceType() == PAWN && piece.GetTeam() != other_piece.GetTeam())
      //   {

      //     int n_turns = (4 + piece.GetColor() - other_piece.GetColor()) % 4;
      //     const Move *other_player_move = nullptr;
      //     if (n_turns > 0 && n_turns <= (int)moves_.size())
      //     {
      //       other_player_move = &moves_[moves_.size() - n_turns];
      //     }
      //     else if (n_turns < 4)
      //     {
      //       const auto &enp_move = enp_.enp_moves[other_piece.GetColor()];
      //       if (enp_move.has_value())
      //       {
      //         other_player_move = &enp_move.value();
      //       }
      //     }

      //     if (other_player_move != nullptr && other_player_move->To() == to
      //         // TODO: Refactor this with 'enp' locations
      //         && other_player_move->ManhattanDistance() == 2 && (other_player_move->From().GetRow() == other_player_move->To().GetRow() || other_player_move->From().GetCol() == other_player_move->To().GetCol()))
      //     {
      //       const BoardLocation &moved_from = other_player_move->From();
      //       int delta_row = to.GetRow() - moved_from.GetRow();
      //       int delta_col = to.GetCol() - moved_from.GetCol();
      //       BoardLocation enpassant_to = moved_from.Relative(
      //           delta_row / 2, delta_col / 2);
      //       // there may be both en-passant and piece capture in the same move
      //       auto existing = GetPiece(enpassant_to);
      //       if (existing.Missing() || existing.GetTeam() != piece.GetTeam())
      //       {
      //         AddPawnMoves2(moves, from, enpassant_to, piece.GetColor(),
      //                       existing, to, other_piece);
      //       }
      //     }
      //   }
      // }
    }

    // Non-enpassant capture
    bool check_cols = team == RED_YELLOW;
    int capture_row, capture_col;
    for (int incr = 0; incr < 2; ++incr)
    {
      capture_row = from.GetRow() + delta_rows;
      capture_col = from.GetCol() + delta_cols;
      if (check_cols)
      {
        capture_col += incr == 0 ? -1 : 1;
      }
      else
      {
        capture_row += incr == 0 ? -1 : 1;
      }
      if (IsLegalLocation(capture_row, capture_col))
      {
        auto other_piece = GetPiece(capture_row, capture_col);
        if (other_piece.Present() && other_piece.GetTeam() != team)
        {
          AddPawnMoves2(moves, from, BoardLocation(capture_row, capture_col),
                        piece.GetColor(), other_piece);
        }
      }
    }
  }

  void Board::GetKnightMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {

    int delta_row, delta_col;
    for (int pos_row_sign = 0; pos_row_sign < 2; ++pos_row_sign)
    {
      for (int abs_delta_row = 1; abs_delta_row < 3; ++abs_delta_row)
      {
        delta_row = pos_row_sign > 0 ? abs_delta_row : -abs_delta_row;
        for (int pos_col_sign = 0; pos_col_sign < 2; ++pos_col_sign)
        {
          int abs_delta_col = abs_delta_row == 1 ? 2 : 1;
          delta_col = pos_col_sign > 0 ? abs_delta_col : -abs_delta_col;
          BoardLocation to = from.Relative(delta_row, delta_col);
          if (IsLegalLocation(to))
          {
            const auto capture = GetPiece(to);
            if (capture.Missing() || capture.GetTeam() != piece.GetTeam())
            {
              moves.emplace_back(from, to, capture);
            }
          }
        }
      }
    }
  }

  void Board::AddMovesFromIncrMovement2(
      MoveBuffer &moves,
      const Piece &piece,
      const BoardLocation &from,
      int incr_row,
      int incr_col,
      CastlingRights initial_castling_rights,
      CastlingRights castling_rights) const
  {
    BoardLocation to = from.Relative(incr_row, incr_col);
    while (IsLegalLocation(to))
    {
      const auto capture = GetPiece(to);
      if (capture.Missing())
      {
        moves.emplace_back(from, to, Piece::kNoPiece, initial_castling_rights,
                           castling_rights);
      }
      else
      {
        if (capture.GetTeam() != piece.GetTeam())
        {
          moves.emplace_back(from, to, capture, initial_castling_rights,
                             castling_rights);
        }
        break;
      }
      to = to.Relative(incr_row, incr_col);
    }
  }

  void Board::GetBishopMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {

    for (int pos_row = 0; pos_row < 2; ++pos_row)
    {
      for (int pos_col = 0; pos_col < 2; ++pos_col)
      {
        AddMovesFromIncrMovement2(
            moves, piece, from, pos_row ? 1 : -1, pos_col ? 1 : -1);
      }
    }
  }

  void Board::GetRookMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {

    // Update castling rights
    CastlingRights initial_castling_rights;
    CastlingRights castling_rights;
    std::optional<CastlingType> castling_type = GetRookLocationType(
        piece.GetPlayer(), from);
    if (castling_type.has_value())
    {
      const auto &curr_rights = castling_rights_[piece.GetColor()];
      if (curr_rights.Kingside() || curr_rights.Queenside())
      {
        if (castling_type == KINGSIDE)
        {
          if (curr_rights.Kingside())
          {
            initial_castling_rights = curr_rights;
            castling_rights = CastlingRights(false, curr_rights.Queenside());
          }
        }
        else
        {
          if (curr_rights.Queenside())
          {
            initial_castling_rights = curr_rights;
            castling_rights = CastlingRights(curr_rights.Kingside(), false);
          }
        }
      }
    }

    for (int do_pos_incr = 0; do_pos_incr < 2; ++do_pos_incr)
    {
      int incr = do_pos_incr > 0 ? 1 : -1;
      for (int do_incr_row = 0; do_incr_row < 2; ++do_incr_row)
      {
        int incr_row = do_incr_row > 0 ? incr : 0;
        int incr_col = do_incr_row > 0 ? 0 : incr;
        AddMovesFromIncrMovement2(moves, piece, from, incr_row, incr_col,
                                  initial_castling_rights, castling_rights);
      }
    }
  }

  void Board::GetQueenMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {
    GetBishopMoves2(moves, from, piece);
    GetRookMoves2(moves, from, piece);
  }

  void Board::GetKingMoves2(
      MoveBuffer &moves,
      const BoardLocation &from,
      const Piece &piece) const
  {

    const CastlingRights &initial_castling_rights = castling_rights_[piece.GetColor()];
    CastlingRights castling_rights(false, false);

    for (int delta_row = -1; delta_row < 2; ++delta_row)
    {
      for (int delta_col = -1; delta_col < 2; ++delta_col)
      {
        if (delta_row == 0 && delta_col == 0)
        {
          continue;
        }
        BoardLocation to = from.Relative(delta_row, delta_col);
        if (IsLegalLocation(to))
        {
          const auto capture = GetPiece(to);
          if (capture.Missing() || capture.GetTeam() != piece.GetTeam())
          {
            moves.emplace_back(from, to, capture, initial_castling_rights,
                               castling_rights);
          }
        }
      }
    }

    Team other_team = OtherTeam(piece.GetTeam());
    for (int is_kingside = 0; is_kingside < 2; ++is_kingside)
    {
      bool allowed = is_kingside ? initial_castling_rights.Kingside() : initial_castling_rights.Queenside();
      if (allowed)
      {
        std::vector<BoardLocation> squares_between;
        BoardLocation rook_location;

        switch (piece.GetColor())
        {
        case RED:
          if (is_kingside)
          {
            squares_between = {
                from.Relative(0, 1),
                from.Relative(0, 2),
            };
            rook_location = from.Relative(0, 3);
          }
          else
          {
            squares_between = {
                from.Relative(0, -1),
                from.Relative(0, -2),
                from.Relative(0, -3),
            };
            rook_location = from.Relative(0, -4);
          }
          break;
        case BLUE:
          if (is_kingside)
          {
            squares_between = {
                from.Relative(1, 0),
                from.Relative(2, 0),
            };
            rook_location = from.Relative(3, 0);
          }
          else
          {
            squares_between = {
                from.Relative(-1, 0),
                from.Relative(-2, 0),
                from.Relative(-3, 0),
            };
            rook_location = from.Relative(-4, 0);
          }
          break;
        case YELLOW:
          if (is_kingside)
          {
            squares_between = {
                from.Relative(0, -1),
                from.Relative(0, -2),
            };
            rook_location = from.Relative(0, -3);
          }
          else
          {
            squares_between = {
                from.Relative(0, 1),
                from.Relative(0, 2),
                from.Relative(0, 3),
            };
            rook_location = from.Relative(0, 4);
          }
          break;
        case GREEN:
          if (is_kingside)
          {
            squares_between = {
                from.Relative(-1, 0),
                from.Relative(-2, 0),
            };
            rook_location = from.Relative(-3, 0);
          }
          else
          {
            squares_between = {
                from.Relative(1, 0),
                from.Relative(2, 0),
                from.Relative(3, 0),
            };
            rook_location = from.Relative(4, 0);
          }
          break;
        default:
          assert(false);
          break;
        }

        // Make sure the rook is present
        const auto rook = GetPiece(rook_location);
        if (rook.Missing() || rook.GetPieceType() != ROOK || rook.GetTeam() != piece.GetTeam())
        {
          continue;
        }

        // Make sure that there are no pieces between the king and rook
        bool piece_between = false;
        for (const auto &loc : squares_between)
        {
          if (GetPiece(loc).Present())
          {
            piece_between = true;
            break;
          }
        }

        if (!piece_between)
        {
          // Make sure the king is not currently in or would pass through check
          if (!IsAttackedByTeam(other_team, squares_between[0]) && !IsAttackedByTeam(other_team, from))
          {
            // Additionally move the castle
            SimpleMove rook_move(rook_location, squares_between[0]);
            moves.emplace_back(from, squares_between[1], rook_move,
                               initial_castling_rights, castling_rights);
          }
        }
      }
    }
  }

  bool Board::RookAttacks(
      const BoardLocation &rook_loc,
      const BoardLocation &other_loc) const
  {
    if (rook_loc.GetRow() == other_loc.GetRow())
    {
      bool piece_between = false;
      for (int col = std::min(rook_loc.GetCol(), other_loc.GetCol()) + 1;
           col < std::max(rook_loc.GetCol(), other_loc.GetCol());
           ++col)
      {
        if (GetPiece(rook_loc.GetRow(), col).Present())
        {
          piece_between = true;
          break;
        }
      }
      if (!piece_between)
      {
        return true;
      }
    }
    if (rook_loc.GetCol() == other_loc.GetCol())
    {
      bool piece_between = false;
      for (int row = std::min(rook_loc.GetRow(), other_loc.GetRow()) + 1;
           row < std::max(rook_loc.GetRow(), other_loc.GetRow());
           ++row)
      {
        if (GetPiece(row, rook_loc.GetCol()).Present())
        {
          piece_between = true;
          break;
        }
      }
      if (!piece_between)
      {
        return true;
      }
    }
    return false;
  }

  bool Board::BishopAttacks(
      const BoardLocation &bishop_loc,
      const BoardLocation &other_loc) const
  {
    int delta_row = bishop_loc.GetRow() - other_loc.GetRow();
    int delta_col = bishop_loc.GetCol() - other_loc.GetCol();
    if (std::abs(delta_row) == std::abs(delta_col))
    {
      int row;
      int col;
      int col_incr;
      int row_max;
      if (bishop_loc.GetRow() < other_loc.GetRow())
      {
        row = bishop_loc.GetRow();
        col = bishop_loc.GetCol();
        row_max = other_loc.GetRow();
        col_incr = bishop_loc.GetCol() < other_loc.GetCol() ? 1 : -1;
      }
      else
      {
        row = other_loc.GetRow();
        col = other_loc.GetCol();
        row_max = bishop_loc.GetRow();
        col_incr = other_loc.GetCol() < bishop_loc.GetCol() ? 1 : -1;
      }
      row++;
      col += col_incr;
      bool piece_between = false;
      while (row < row_max)
      {
        if (GetPiece(row, col).Present())
        {
          piece_between = true;
          break;
        }

        ++row;
        col += col_incr;
      }
      return !piece_between;
    }
    return false;
  }

  bool Board::QueenAttacks(
      const BoardLocation &queen_loc,
      const BoardLocation &other_loc) const
  {
    return RookAttacks(queen_loc, other_loc) || BishopAttacks(queen_loc, other_loc);
  }

  bool Board::KingAttacks(
      const BoardLocation &king_loc,
      const BoardLocation &other_loc) const
  {
    if ((std::abs(king_loc.GetRow() - other_loc.GetRow()) + std::abs(king_loc.GetCol() - other_loc.GetCol())) < 2)
    {
      return true;
    }
    return false;
  }

  bool Board::KnightAttacks(
      const BoardLocation &knight_loc,
      const BoardLocation &other_loc) const
  {
    int abs_row_diff = std::abs(knight_loc.GetRow() - other_loc.GetRow());
    int abs_col_diff = std::abs(knight_loc.GetCol() - other_loc.GetCol());
    return (abs_row_diff == 1 && abs_col_diff == 2) || (abs_row_diff == 2 && abs_col_diff == 1);
  }

  bool Board::PawnAttacks(
      const BoardLocation &pawn_loc,
      PlayerColor pawn_color,
      const BoardLocation &other_loc) const
  {
    int row_diff = other_loc.GetRow() - pawn_loc.GetRow();
    int col_diff = other_loc.GetCol() - pawn_loc.GetCol();
    switch (pawn_color)
    {
    case RED:
      return row_diff == -1 && (std::abs(col_diff) == 1);
    case BLUE:
      return col_diff == 1 && (std::abs(row_diff) == 1);
    case YELLOW:
      return row_diff == 1 && (std::abs(col_diff) == 1);
    case GREEN:
      return col_diff == -1 && (std::abs(row_diff) == 1);
    default:
      assert(false);
      return false;
    }
  }

  size_t Board::GetAttackers2(
      PlacedPiece *buffer, size_t limit,
      Team team, const BoardLocation &location) const
  {
    assert(limit > 0);
    size_t pos = 0;

#define ADD_ATTACKER(row, col, piece)                          \
  buffer[pos++] = PlacedPiece(BoardLocation(row, col), piece); \
  if (pos == limit)                                            \
  {                                                            \
    return limit;                                              \
  }

    int loc_row = location.GetRow();
    int loc_col = location.GetCol();

    // Rooks & queens
    for (int do_incr_row = 0; do_incr_row < 2; ++do_incr_row)
    {
      for (int pos_incr = 0; pos_incr < 2; ++pos_incr)
      {
        int row_incr = do_incr_row ? (pos_incr ? 1 : -1) : 0;
        int col_incr = do_incr_row ? 0 : (pos_incr ? 1 : -1);
        int row = loc_row + row_incr;
        int col = loc_col + col_incr;
        while (row >= 0 && row < 14 && col >= 0 && col < 14)
        {
          const auto piece = GetPiece(row, col);
          if (piece.Present())
          {
            if (piece.GetTeam() == team && (piece.GetPieceType() == ROOK || piece.GetPieceType() == QUEEN))
            {
              ADD_ATTACKER(row, col, piece);
            }
            break;
          }
          row += row_incr;
          col += col_incr;
        }
      }
    }

    // Bishops & queens
    for (int pos_row = 0; pos_row < 2; ++pos_row)
    {
      int row_incr = pos_row ? 1 : -1;
      for (int pos_col = 0; pos_col < 2; ++pos_col)
      {
        int col_incr = pos_col ? 1 : -1;
        int row = loc_row + row_incr;
        int col = loc_col + col_incr;
        while (IsLegalLocation(row, col))
        {
          const auto piece = GetPiece(row, col);
          if (piece.Present())
          {
            if (piece.GetTeam() == team && (piece.GetPieceType() == BISHOP || piece.GetPieceType() == QUEEN))
            {
              ADD_ATTACKER(row, col, piece);
            }
            break;
          }
          row += row_incr;
          col += col_incr;
        }
      }
    }

    // Knights
    for (int row_less = 0; row_less < 2; ++row_less)
    {
      for (int pos_row = 0; pos_row < 2; ++pos_row)
      {
        int row = loc_row + (row_less ? (pos_row ? 1 : -1) : (pos_row ? 2 : -2));
        for (int pos_col = 0; pos_col < 2; ++pos_col)
        {
          int col = loc_col + (row_less ? (pos_col ? 2 : -2) : (pos_col ? 1 : -1));
          if (IsLegalLocation(row, col))
          {
            const auto piece = GetPiece(row, col);
            if (piece.Present() && piece.GetTeam() == team && piece.GetPieceType() == KNIGHT)
            {
              ADD_ATTACKER(row, col, piece);
            }
          }
        }
      }
    }

    // Pawns
    for (int pos_row = 0; pos_row < 2; ++pos_row)
    {
      int row = pos_row ? loc_row + 1 : loc_row - 1;
      if (row >= 0 && row < 14)
      {
        for (int pos_col = 0; pos_col < 2; ++pos_col)
        {
          int col = pos_col ? loc_col + 1 : loc_col - 1;
          if (col >= 0 && col < 14)
          {
            const auto piece = GetPiece(row, col);
            if (piece.Present() && piece.GetTeam() == team && piece.GetPieceType() == PAWN)
            {
              bool attacks = false;
              switch (piece.GetColor())
              {
              case RED:
                if (pos_row)
                {
                  attacks = true;
                }
                break;
              case BLUE:
                if (!pos_col)
                {
                  attacks = true;
                }
                break;
              case YELLOW:
                if (!pos_row)
                {
                  attacks = true;
                }
                break;
              case GREEN:
                if (pos_col)
                {
                  attacks = true;
                }
                break;
              default:
                assert(false);
                break;
              }

              if (attacks)
              {
                ADD_ATTACKER(row, col, piece);
              }
            }
          }
        }
      }
    }

    // Kings
    for (int delta_row = -1; delta_row < 2; ++delta_row)
    {
      int row = loc_row + delta_row;
      for (int delta_col = -1; delta_col < 2; ++delta_col)
      {
        if (delta_row == 0 && delta_col == 0)
        {
          continue;
        }
        int col = loc_col + delta_col;
        if (IsLegalLocation(row, col))
        {
          const auto piece = GetPiece(row, col);
          if (piece.Present() && piece.GetTeam() == team && piece.GetPieceType() == KING)
          {
            ADD_ATTACKER(row, col, piece);
          }
        }
      }
    }

#undef ADD_ATTACKER

    return pos;
  }

  bool Board::IsAttackedByTeam(Team team, const BoardLocation &location) const
  {
    PlacedPiece attackers[1];
    size_t pos = GetAttackers2(attackers, 1, team, location);
    return pos > 0;

    // auto attackers = GetAttackers(team, location, /*return_early=*/true);
    // return attackers.size() > 0;
  }

  bool Board::DiscoversCheck(
      const BoardLocation &king_location,
      const BoardLocation &move_from,
      const BoardLocation &move_to,
      Team attacking_team) const
  {
    int delta_row = move_from.GetRow() - king_location.GetRow();
    int delta_col = move_from.GetCol() - king_location.GetCol();
    if (std::abs(delta_row) != std::abs(delta_col) && delta_row != 0 && delta_col != 0)
    {
      return false;
    }

    int incr_col = delta_col == 0 ? 0 : delta_col > 0 ? 1
                                                      : -1;
    int incr_row = delta_row == 0 ? 0 : delta_row > 0 ? 1
                                                      : -1;
    int row = king_location.GetRow() + incr_row;
    int col = king_location.GetCol() + incr_col;
    while (IsLegalLocation(row, col))
    {
      if (row != move_from.GetRow() || col != move_from.GetCol())
      {
        if (row == move_to.GetRow() && col == move_to.GetCol())
        {
          return false;
        }
        const auto piece = GetPiece(row, col);
        if (piece.Present())
        {
          if (piece.GetTeam() == attacking_team)
          {
            if (delta_row == 0 || delta_col == 0)
            {
              if (piece.GetPieceType() == QUEEN || piece.GetPieceType() == ROOK)
              {
                return true;
              }
            }
            else
            {
              if (piece.GetPieceType() == QUEEN || piece.GetPieceType() == BISHOP)
              {
                return true;
              }
            }
          }
          break;
        }
      }

      row += incr_row;
      col += incr_col;
    }
    return false;
  }

  size_t Board::GetPseudoLegalMoves2(Move *buffer, size_t limit)
  {
    MoveBuffer move_buffer;
    move_buffer.buffer = buffer;
    move_buffer.limit = limit;

    BoardLocation king_location = GetKingLocation(turn_.GetColor());
    if (!king_location.Present())
    {
      return 0;
    }

    for (const auto &placed_piece : piece_list_[turn_.GetColor()])
    {
      const auto &location = placed_piece.GetLocation();
      const auto &piece = placed_piece.GetPiece();
      switch (piece.GetPieceType())
      {
      case PAWN:
        GetPawnMoves2(move_buffer, location, piece);
        break;
      case KNIGHT:
        GetKnightMoves2(move_buffer, location, piece);
        break;
      case BISHOP:
        GetBishopMoves2(move_buffer, location, piece);
        break;
      case ROOK:
        GetRookMoves2(move_buffer, location, piece);
        break;
      case QUEEN:
        GetQueenMoves2(move_buffer, location, piece);
        break;
      case KING:
        GetKingMoves2(move_buffer, location, piece);
        king_location = location;
        break;
      default:
        assert(false);
      }
    }

    return move_buffer.pos;
  }

  GameResult Board::GetGameResult(std::optional<Player> opt_player)
  {
    Player player = opt_player.value_or(turn_); // Use the provided player or default to turn_

    if (!GetKingLocation(player.GetColor()).Present())
    {
      // other team won
      return player.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
    }

    Move move_buffer[300];
    size_t num_moves = GetPseudoLegalMoves2(move_buffer, move_buffer_size_);

    for (size_t i = 0; i < num_moves; i++)
    {
      const auto &move = move_buffer[i];
      MakeMove(move);
      GameResult king_capture_result = CheckWasLastMoveKingCapture();
      if (king_capture_result != IN_PROGRESS)
      {
        UndoMove();
        return king_capture_result;
      }
      bool legal = !IsKingInCheck(player);
      UndoMove();
      if (legal)
      {
        return IN_PROGRESS;
      }
    }

    if (!IsKingInCheck(player))
    {
      return STALEMATE;
    }

    // No legal moves
    PlayerColor color = player.GetColor();
    if (color == RED || color == YELLOW)
    {
      return WIN_BG;
    }
    return WIN_RY;
  }

  bool Board::IsKingInCheck(const Player &player) const
  {
    const auto king_location = GetKingLocation(player.GetColor());

    if (!king_location.Present())
    {
      return false;
    }

    return IsAttackedByTeam(OtherTeam(player.GetTeam()), king_location);
  }

  bool Board::IsKingInCheck(Team team) const
  {
    if (team == RED_YELLOW)
    {
      return IsKingInCheck(Player(RED)) || IsKingInCheck(Player(YELLOW));
    }
    return IsKingInCheck(Player(BLUE)) || IsKingInCheck(Player(GREEN));
  }

  GameResult Board::CheckWasLastMoveKingCapture() const
  {
    // King captured last move
    if (!moves_.empty())
    {
      const auto &last_move = moves_.back();
      const auto capture = last_move.GetCapturePiece();
      if (capture.Present() && capture.GetPieceType() == KING)
      {
        return capture.GetTeam() == RED_YELLOW ? WIN_BG : WIN_RY;
      }
    }
    return IN_PROGRESS;
  }

  void Board::SetPiece(
      const BoardLocation &location,
      const Piece &piece)
  {
    location_to_piece_[location.GetRow()][location.GetCol()] = piece;
    // Add to piece_list_
    piece_list_[piece.GetColor()].emplace_back(location, piece);
    // UpdatePieceHash(piece, location);
    // Update king location
    if (piece.GetPieceType() == KING)
    {
      king_locations_[piece.GetColor()] = location;
    }
  }

  void Board::RemovePiece(const BoardLocation &location)
  {
    const auto piece = GetPiece(location);
    assert(piece.Present());
    // UpdatePieceHash(piece, location);
    location_to_piece_[location.GetRow()][location.GetCol()] = Piece();
    auto &placed_pieces = piece_list_[piece.GetColor()];
    for (auto it = placed_pieces.begin(); it != placed_pieces.end();)
    {
      const auto &placed_piece = *it;
      if (placed_piece.GetLocation() == location)
      {
        placed_pieces.erase(it);
        break;
      }
      ++it;
    }
    // Update king location
    if (piece.GetPieceType() == KING)
    {
      king_locations_[piece.GetColor()] = BoardLocation::kNoLocation;
    }
  }

  // void Board::InitializeHash()
  // {
  //   for (int color = 0; color < 4; color++)
  //   {
  //     for (const auto &placed_piece : piece_list_[color])
  //     {
  //       UpdatePieceHash(placed_piece.GetPiece(), placed_piece.GetLocation());
  //     }
  //   }
  //   UpdateTurnHash(static_cast<int>(turn_.GetColor()));
  // }

  void Board::MakeMove(const Move &move)
  {
    // Cases:
    // 1. Move
    // 2. Capture
    // 3. En-passant
    // 4. Promotion
    // 5. Castling (rights, rook move)

    const auto piece = GetPiece(move.From());

    // TODO: refactor this -- it should account for the case where we both capture
    // and en-passant capture.
    const auto capture = move.GetCapturePiece();
    if (capture.Present())
    {
      int value = piece_evaluations_[capture.GetPieceType()];
      if (capture.GetTeam() == RED_YELLOW)
      {
        piece_evaluation_ -= value;
      }
      else
      {
        piece_evaluation_ += value;
      }
      player_piece_evaluations_[capture.GetColor()] -= value;
    }

    // Capture
    const auto standard_capture = GetPiece(move.To());
    if (standard_capture.Present())
    {
      RemovePiece(move.To());
    }

    if (piece.Missing())
    {
      std::ostringstream err_msg;
      err_msg << "Error: piece missing for move: " << std::endl
              << "from: " << move.From().PrettyStr()
              << " to: " << move.To().PrettyStr() << std::endl
              << " turn: " << turn_ << std::endl;
      throw std::runtime_error(err_msg.str());
    }

    RemovePiece(move.From());
    const auto promotion_piece_type = move.GetPromotionPieceType();
    if (promotion_piece_type != NO_PIECE)
    { // Promotion
      SetPiece(
          move.To(),
          Piece(turn_.GetColor(), promotion_piece_type));
    }
    else
    { // Move
      SetPiece(move.To(), piece);
    }

    // En-passant
    const auto enpassant_location = move.GetEnpassantLocation();
    if (enpassant_location.Present())
    {
      RemovePiece(enpassant_location);
    }
    else
    {
      // Castling
      const auto rook_move = move.GetRookMove();
      if (rook_move.Present())
      {
        const auto rook = GetPiece(rook_move.From());
        assert(rook.Present());
        RemovePiece(rook_move.From());
        SetPiece(rook_move.To(), rook);
      }

      // Castling: rights update
      const auto castling_rights = move.GetCastlingRights();
      if (castling_rights.Present())
      {
        castling_rights_[turn_.GetColor()] = castling_rights;
      }
    }

    int t = static_cast<int>(turn_.GetColor());
    // UpdateTurnHash(t);
    // UpdateTurnHash((t + 1) % 4);

    turn_ = GetNextPlayer(turn_);
    moves_.push_back(move);

    if (moves_.size() > max_moves_storage)
    {
      // Remove the oldest move
      moves_.erase(moves_.begin());
    }
  }

  void Board::UndoMove()
  {
    // Cases:
    // 1. Move
    // 2. Capture
    // 3. En-passant
    // 4. Promotion
    // 5. Castling (rights, rook move)

    assert(!moves_.empty());
    const Move &move = moves_.back();
    Player turn_before = GetPreviousPlayer(turn_);

    const BoardLocation &to = move.To();
    const BoardLocation &from = move.From();

    // Move the piece back.
    const auto piece = GetPiece(to);
    if (piece.Missing())
    {
      std::cout << "piece missing in UndoMove" << std::endl;
      std::cout << *this << std::endl;
      abort();
    }

    RemovePiece(to);
    const auto promotion_piece_type = move.GetPromotionPieceType();
    if (promotion_piece_type != NO_PIECE)
    {
      // Handle promotions
      SetPiece(from, Piece(turn_before.GetColor(), PAWN));
    }
    else
    {
      SetPiece(from, piece);
    }

    // TODO: refactor this -- it should account for the case where we both capture
    // and en-passant capture.
    const auto capture = move.GetCapturePiece();
    if (capture.Present())
    {
      int value = piece_evaluations_[capture.GetPieceType()];
      if (capture.GetTeam() == RED_YELLOW)
      {
        piece_evaluation_ += value;
      }
      else
      {
        piece_evaluation_ -= value;
      }
      player_piece_evaluations_[capture.GetColor()] += value;
    }

    // Place back captured pieces
    const auto standard_capture = move.GetStandardCapture();
    if (standard_capture.Present())
    {
      SetPiece(to, standard_capture);
    }

    // Place back en-passant pawns
    const auto enpassant_location = move.GetEnpassantLocation();
    if (enpassant_location.Present())
    {
      SetPiece(enpassant_location,
               move.GetEnpassantCapture());
    }
    else
    {
      // Castling: rook move
      const auto rook_move = move.GetRookMove();
      if (rook_move.Present())
      {
        RemovePiece(rook_move.To());
        SetPiece(rook_move.From(), Piece(turn_before.GetColor(), ROOK));
      }

      // Castling: rights update
      const auto initial_castling_rights = move.GetInitialCastlingRights();
      if (initial_castling_rights.Present())
      {
        castling_rights_[turn_before.GetColor()] = initial_castling_rights;
      }
    }

    turn_ = turn_before;
    moves_.pop_back();
    int t = static_cast<int>(turn_.GetColor());
    // UpdateTurnHash(t);
    // UpdateTurnHash((t + 1) % 4);
  }

  BoardLocation Board::GetKingLocation(PlayerColor color) const
  {
    return king_locations_[color];
  }

  Team Board::TeamToPlay() const
  {
    return GetTeam(GetTurn().GetColor());
  }

  Board::Board(
      Player turn,
      std::unordered_map<BoardLocation, Piece> location_to_piece,
      std::optional<std::unordered_map<Player, CastlingRights>> castling_rights,
      std::optional<EnpassantInitialization> enp)
      : turn_(std::move(turn))
  {

    // In centipawns
    piece_evaluations_[PAWN] = 50;
    piece_evaluations_[KNIGHT] = 300;
    piece_evaluations_[BISHOP] = 400;
    piece_evaluations_[ROOK] = 500;
    piece_evaluations_[QUEEN] = 1000;
    piece_evaluations_[KING] = 10000;

    for (int color = 0; color < 4; color++)
    {
      castling_rights_[color] = CastlingRights(false, false);
      if (castling_rights.has_value())
      {
        auto &cr = castling_rights.value();
        Player pl(static_cast<PlayerColor>(color));
        auto it = cr.find(pl);
        if (it != cr.end())
        {
          castling_rights_[color] = it->second;
        }
      }
    }
    if (enp.has_value())
    {
      enp_ = std::move(enp.value());
    }
    // move_buffer_.reserve(1000);

    for (int i = 0; i < 14; ++i)
    {
      for (int j = 0; j < 14; ++j)
      {
        locations_[i][j] = BoardLocation(i, j);
        location_to_piece_[i][j] = Piece();
      }
    }

    for (int i = 0; i < 4; i++)
    {
      piece_list_.push_back(std::vector<PlacedPiece>());
      piece_list_[i].reserve(16);
      king_locations_[i] = BoardLocation::kNoLocation;
    }

    for (const auto &it : location_to_piece)
    {
      const auto &location = it.first;
      const auto &piece = it.second;
      PlayerColor color = piece.GetColor();
      location_to_piece_[location.GetRow()][location.GetCol()] = piece;
      piece_list_[piece.GetColor()].push_back(PlacedPiece(
          locations_[location.GetRow()][location.GetCol()],
          piece));
      PieceType piece_type = piece.GetPieceType();
      if (piece.GetTeam() == RED_YELLOW)
      {
        piece_evaluation_ += piece_evaluations_[static_cast<int>(piece_type)];
      }
      else
      {
        piece_evaluation_ -= piece_evaluations_[static_cast<int>(piece_type)];
      }
      player_piece_evaluations_[piece.GetColor()] += piece_evaluations_[piece_type];
      if (piece.GetPieceType() == KING)
      {
        king_locations_[color] = location;
      }
    }

    struct
    {
      bool operator()(const PlacedPiece &a, const PlacedPiece &b)
      {
        // this doesn't need to be fast.
        int piece_move_order_scores[6];
        piece_move_order_scores[PAWN] = 1;
        piece_move_order_scores[KNIGHT] = 2;
        piece_move_order_scores[BISHOP] = 3;
        piece_move_order_scores[ROOK] = 4;
        piece_move_order_scores[QUEEN] = 5;
        piece_move_order_scores[KING] = 0;

        int order_a = piece_move_order_scores[a.GetPiece().GetPieceType()];
        int order_b = piece_move_order_scores[b.GetPiece().GetPieceType()];
        return order_a < order_b;
      }
    } customLess;

    for (auto &placed_pieces : piece_list_)
    {
      std::sort(placed_pieces.begin(), placed_pieces.end(), customLess);
    }

    // Initialize hashes for each piece at each location, and each turn
    std::srand(958829);
    for (int color = 0; color < 4; color++)
    {
      turn_hashes_[color] = rand64();
    }
    // for (int color = 0; color < 4; color++)
    // {
    //   for (int piece_type = 0; piece_type < 6; piece_type++)
    //   {
    //     for (int row = 0; row < 14; row++)
    //     {
    //       for (int col = 0; col < 14; col++)
    //       {
    //         piece_hashes_[color][piece_type][row][col] = rand64();
    //       }
    //     }
    //   }
    // }

    // InitializeHash();
  }

  Board::Board(const Board &other)
      : turn_(other.turn_),
        enp_(other.enp_),
        moves_(other.moves_),
        move_buffer_(other.move_buffer_),
        piece_evaluation_(other.piece_evaluation_),
        hash_key_(other.hash_key_),
        move_buffer_size_(other.move_buffer_size_)
  {
    // Perform copying
    std::copy(&other.location_to_piece_[0][0], &other.location_to_piece_[0][0] + 14 * 14, &location_to_piece_[0][0]);
    piece_list_ = other.piece_list_;
    std::copy(&other.locations_[0][0], &other.locations_[0][0] + 14 * 14, &locations_[0][0]);
    std::copy(std::begin(other.castling_rights_), std::end(other.castling_rights_), std::begin(castling_rights_));
    std::copy(std::begin(other.piece_evaluations_), std::end(other.piece_evaluations_), std::begin(piece_evaluations_));
    std::copy(std::begin(other.player_piece_evaluations_), std::end(other.player_piece_evaluations_), std::begin(player_piece_evaluations_));
    std::copy(std::begin(other.turn_hashes_), std::end(other.turn_hashes_), std::begin(turn_hashes_));
    std::copy(std::begin(other.king_locations_), std::end(other.king_locations_), std::begin(king_locations_));
  }

  inline Team GetTeam(PlayerColor color)
  {
    return (color == RED || color == YELLOW) ? RED_YELLOW : BLUE_GREEN;
  }

  Player GetNextPlayer(const Player &player)
  {
    switch (player.GetColor())
    {
    case RED:
      return kBluePlayer;
    case BLUE:
      return kYellowPlayer;
    case YELLOW:
      return kGreenPlayer;
    case GREEN:
    default:
      return kRedPlayer;
    }
  }

  Player GetPartner(const Player &player)
  {
    switch (player.GetColor())
    {
    case RED:
      return kYellowPlayer;
    case BLUE:
      return kGreenPlayer;
    case YELLOW:
      return kRedPlayer;
    case GREEN:
    default:
      return kBluePlayer;
    }
  }

  Player GetPreviousPlayer(const Player &player)
  {
    switch (player.GetColor())
    {
    case RED:
      return kGreenPlayer;
    case BLUE:
      return kRedPlayer;
    case YELLOW:
      return kBluePlayer;
    case GREEN:
    default:
      return kYellowPlayer;
    }
  }

  std::shared_ptr<Board> Board::CreateStandardSetup()
  {
    std::unordered_map<BoardLocation, Piece> location_to_piece;
    std::unordered_map<Player, CastlingRights> castling_rights;

    std::vector<PieceType> piece_types = {
        ROOK,
        KNIGHT,
        BISHOP,
        QUEEN,
        KING,
        BISHOP,
        KNIGHT,
        ROOK,
    };
    std::vector<PlayerColor> player_colors = {RED, BLUE, YELLOW, GREEN};

    for (const PlayerColor &color : player_colors)
    {
      Player player(color);
      castling_rights[player] = CastlingRights(true, true);

      BoardLocation piece_location;
      int delta_row = 0;
      int delta_col = 0;
      int pawn_offset_row = 0;
      int pawn_offset_col = 0;

      switch (color)
      {
      case RED:
        piece_location = BoardLocation(13, 3);
        delta_col = 1;
        pawn_offset_row = -1;
        break;
      case BLUE:
        piece_location = BoardLocation(3, 0);
        delta_row = 1;
        pawn_offset_col = 1;
        break;
      case YELLOW:
        piece_location = BoardLocation(0, 10);
        delta_col = -1;
        pawn_offset_row = 1;
        break;
      case GREEN:
        piece_location = BoardLocation(10, 13);
        delta_row = -1;
        pawn_offset_col = -1;
        break;
      default:
        assert(false);
        break;
      }

      for (const PieceType piece_type : piece_types)
      {
        BoardLocation pawn_location = piece_location.Relative(
            pawn_offset_row, pawn_offset_col);
        location_to_piece[piece_location] = Piece(player.GetColor(), piece_type);
        location_to_piece[pawn_location] = Piece(player.GetColor(), PAWN);
        piece_location = piece_location.Relative(delta_row, delta_col);
      }
    }

    return std::make_shared<Board>(
        Player(RED), std::move(location_to_piece), std::move(castling_rights));
  }

  int Move::ManhattanDistance() const
  {
    return std::abs(from_.GetRow() - to_.GetRow()) + std::abs(from_.GetCol() - to_.GetCol());
  }

  namespace
  {

    std::string ToStr(PlayerColor color)
    {
      switch (color)
      {
      case RED:
        return "RED";
      case BLUE:
        return "BLUE";
      case YELLOW:
        return "YELLOW";
      case GREEN:
        return "GREEN";
      default:
        return "UNINITIALIZED_PLAYER";
      }
    }

    std::string ToStr(PieceType piece_type)
    {
      switch (piece_type)
      {
      case PAWN:
        return "P";
      case ROOK:
        return "R";
      case KNIGHT:
        return "N";
      case BISHOP:
        return "B";
      case KING:
        return "K";
      case QUEEN:
        return "Q";
      default:
        return "U";
      }
    }

  } // namespace

  std::ostream &operator<<(
      std::ostream &os, const Piece &piece)
  {
    os << ToStr(piece.GetColor()) << "(" << ToStr(piece.GetPieceType()) << ")";
    return os;
  }

  std::ostream &operator<<(
      std::ostream &os, const PlacedPiece &placed_piece)
  {
    os << placed_piece.GetPiece() << "@" << placed_piece.GetLocation();
    return os;
  }

  std::ostream &operator<<(
      std::ostream &os, const Player &player)
  {
    os << "Player(" << ToStr(player.GetColor()) << ")";
    return os;
  }

  std::ostream &operator<<(
      std::ostream &os, const BoardLocation &location)
  {
    os << "Loc(" << (int)location.GetRow() << ", " << (int)location.GetCol() << ")";
    return os;
  }

  std::ostream &operator<<(std::ostream &os, const Move &move)
  {
    os << "Move(" << move.From().PrettyStr() << " -> " << move.To().PrettyStr() << ")";
    return os;
  }

  std::ostream &operator<<(std::ostream &os, const Board &board)
  {
    for (int i = 0; i < 14; i++)
    {
      os << (13 - i < 9 ? " " : "") << 13 - i + 1 << ":";
      for (int j = 0; j < 14; j++)
      {
        const auto piece = board.location_to_piece_[i][j];
        if (board.IsLegalLocation(BoardLocation(i, j)))
        {
          if (piece.Missing())
          {
            os << " . ";
          }
          else
          {
            // os << ' ' << ToStr(piece.GetPieceType()) << ' ';
            os << static_cast<int>(piece.GetColor()) << ToStr(piece.GetPieceType()) << ' ';
          }
        }
        else
        {
          os << "   ";
        }
      }
      os << std::endl;
    }

    // Column labels at the bottom, from 'a' to 'n'
    os << "   "; // Align with the row labels
    for (int j = 0; j < 14; j++)
    {
      os << " " << char('a' + j) << " ";
    }
    os << std::endl;

    os << "Turn: " << board.turn_ << std::endl;
    return os;
  }

  const CastlingRights &Board::GetCastlingRights(const Player &player)
  {
    return castling_rights_[player.GetColor()];
  }

  std::optional<CastlingType> Board::GetRookLocationType(
      const Player &player, const BoardLocation &location) const
  {
    switch (player.GetColor())
    {
    case RED:
      if (location == kRedInitialRookLocationKingside)
      {
        return KINGSIDE;
      }
      else if (location == kRedInitialRookLocationQueenside)
      {
        return QUEENSIDE;
      }
      break;
    case BLUE:
      if (location == kBlueInitialRookLocationKingside)
      {
        return KINGSIDE;
      }
      else if (location == kBlueInitialRookLocationQueenside)
      {
        return QUEENSIDE;
      }
      break;
    case YELLOW:
      if (location == kYellowInitialRookLocationKingside)
      {
        return KINGSIDE;
      }
      else if (location == kYellowInitialRookLocationQueenside)
      {
        return QUEENSIDE;
      }
      break;
    case GREEN:
      if (location == kGreenInitialRookLocationKingside)
      {
        return KINGSIDE;
      }
      else if (location == kGreenInitialRookLocationQueenside)
      {
        return QUEENSIDE;
      }
      break;
    default:
      assert(false);
      break;
    }
    return std::nullopt;
  }

  Team OtherTeam(Team team)
  {
    return team == RED_YELLOW ? BLUE_GREEN : RED_YELLOW;
  }

  std::string BoardLocation::PrettyStr() const
  {
    std::string s;
    s += ('a' + GetCol());
    s += std::to_string(14 - GetRow());
    return s;
  }

  std::string Move::PrettyStr() const
  {
    std::string s = from_.PrettyStr() + "-" + to_.PrettyStr();
    if (promotion_piece_type_ != NO_PIECE)
    {
      s += "=" + ToStr(promotion_piece_type_);
    }
    return s;
  }

  bool Board::DeliversCheck(const Move &move)
  {
    int color = GetTurn().GetColor();
    Piece piece = GetPiece(move.From());

    bool checks = false;

    for (int add = 1; add < 4; add += 2)
    {
      int other = (color + add) % 4;
      auto king_loc = GetKingLocation(static_cast<PlayerColor>(other));
      if (king_loc.Present())
      {
        if (king_loc == move.To())
        {
          checks = true;
          break;
        }
        switch (piece.GetPieceType())
        {
        case PAWN:
          checks = PawnAttacks(move.To(), piece.GetColor(), king_loc);
          break;
        case KNIGHT:
          checks = KnightAttacks(move.To(), king_loc);
          break;
        case BISHOP:
          checks = BishopAttacks(move.To(), king_loc);
          break;
        case ROOK:
          checks = RookAttacks(move.To(), king_loc);
          break;
        case QUEEN:
          checks = QueenAttacks(move.To(), king_loc);
          break;
        default:
          break;
        }
        if (checks)
        {
          break;
        }
      }
    }

    return checks;
  }

  void Board::MakeNullMove()
  {
    int t = static_cast<int>(turn_.GetColor());
    // UpdateTurnHash(t);
    // UpdateTurnHash((t + 1) % 4);

    turn_ = GetNextPlayer(turn_);
  }

  void Board::UndoNullMove()
  {
    turn_ = GetPreviousPlayer(turn_);

    int t = static_cast<int>(turn_.GetColor());
    // UpdateTurnHash(t);
    // UpdateTurnHash((t + 1) % 4);
  }

  bool Move::DeliversCheck(Board &board)
  {
    if (delivers_check_ < 0)
    {
      delivers_check_ = board.DeliversCheck(*this);
    }
    return delivers_check_;
  }

} // namespace chess
