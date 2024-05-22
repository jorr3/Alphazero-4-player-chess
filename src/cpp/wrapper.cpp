#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <torch/extension.h>
#include <sstream>
#include "move.h"
#include "board.h"
#include "node.h"
#include "../../engine/board.h"
#include "../../engine/player.h"

namespace py = pybind11;

PYBIND11_MODULE(alphazero_cpp, m)
{
    py::register_exception_translator([](std::exception_ptr p)
                                      {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception &e) {
            // Only use the exception's what() method to get the message
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (...) {
            // This catches any other exceptions not derived from std::exception
            PyErr_SetString(PyExc_Exception, "An unknown exception occurred.");
        } });

    py::module_::import("torch");

    // Enums
    py::enum_<chess::PieceType>(m, "PieceType")
        .value("PAWN", chess::PieceType::PAWN)
        .value("KNIGHT", chess::PieceType::KNIGHT)
        .value("BISHOP", chess::PieceType::BISHOP)
        .value("ROOK", chess::PieceType::ROOK)
        .value("QUEEN", chess::PieceType::QUEEN)
        .value("KING", chess::PieceType::KING)
        .value("NO_PIECE", chess::PieceType::NO_PIECE)
        .export_values();

    m.def("piece_value", [](const chess::PieceType &type)
          { return static_cast<int>(type); });

    py::enum_<chess::PlayerColor>(m, "PlayerColor")
        .value("UNINITIALIZED_PLAYER", chess::PlayerColor::UNINITIALIZED_PLAYER)
        .value("RED", chess::PlayerColor::RED)
        .value("BLUE", chess::PlayerColor::BLUE)
        .value("YELLOW", chess::PlayerColor::YELLOW)
        .value("GREEN", chess::PlayerColor::GREEN)
        .export_values();

    m.def("color_value", [](const chess::PlayerColor &color)
          { return static_cast<int>(color); });

    py::enum_<chess::Team>(m, "Team")
        .value("RED_YELLOW", chess::Team::RED_YELLOW)
        .value("BLUE_GREEN", chess::Team::BLUE_GREEN)
        .export_values();

    py::enum_<chess::GameResult>(m, "GameResult")
        .value("IN_PROGRESS", chess::GameResult::IN_PROGRESS)
        .value("WIN_RY", chess::GameResult::WIN_RY)
        .value("WIN_BG", chess::GameResult::WIN_BG)
        .value("STALEMATE", chess::GameResult::STALEMATE)
        .export_values();

    // Structs
    py::class_<chess::EvaluationOptions>(m, "EvaluationOptions")
        .def(py::init<>())
        .def_readwrite("timelimit", &chess::EvaluationOptions::timelimit)
        .def_readwrite("search_moves", &chess::EvaluationOptions::search_moves)
        .def_readwrite("ponder", &chess::EvaluationOptions::ponder)
        .def_readwrite("red_time", &chess::EvaluationOptions::red_time)
        .def_readwrite("blue_time", &chess::EvaluationOptions::blue_time)
        .def_readwrite("yellow_time", &chess::EvaluationOptions::yellow_time)
        .def_readwrite("green_time", &chess::EvaluationOptions::green_time)
        .def_readwrite("red_inc", &chess::EvaluationOptions::red_inc)
        .def_readwrite("blue_inc", &chess::EvaluationOptions::blue_inc)
        .def_readwrite("yellow_inc", &chess::EvaluationOptions::yellow_inc)
        .def_readwrite("green_inc", &chess::EvaluationOptions::green_inc)
        .def_readwrite("moves_to_go", &chess::EvaluationOptions::moves_to_go)
        .def_readwrite("depth", &chess::EvaluationOptions::depth)
        .def_readwrite("nodes", &chess::EvaluationOptions::nodes)
        .def_readwrite("mate", &chess::EvaluationOptions::mate)
        .def_readwrite("infinite", &chess::EvaluationOptions::infinite);

    py::class_<chess::SimpleBoardState>(m, "SimpleBoardState")
        .def(py::init<>())
        .def_readwrite("turn", &chess::SimpleBoardState::turn)
        .def_readwrite("pieces", &chess::SimpleBoardState::pieces)
        .def_readwrite("castlingRights", &chess::SimpleBoardState::castlingRights)
        .def_readwrite("enpassantInitialization", &chess::SimpleBoardState::enpassantInitialization)
        .def_readwrite("attackedSquares", &chess::SimpleBoardState::attackedSquares);

    // Classes
    py::class_<chess::Player>(m, "Player")
        .def(py::init<>())
        .def(py::init<chess::PlayerColor>())
        .def("GetColor", &chess::Player::GetColor)
        .def("GetTeam", &chess::Player::GetTeam)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<chess::Piece>(m, "Piece")
        .def(py::init<>())
        .def(py::init<bool, chess::PlayerColor, chess::PieceType>())
        .def(py::init<chess::PlayerColor, chess::PieceType>())
        .def(py::init<chess::Player, chess::PieceType>())
        .def("GetColor", &chess::Piece::GetColor)
        .def("GetPieceType", &chess::Piece::GetPieceType)
        .def("GetPlayer", &chess::Piece::GetPlayer)
        .def("PieceTypeToStr", &chess::Piece::PieceTypeToStr, py::arg("type"))
        .def("ColorToStr", [](const chess::Piece &self, chess::PlayerColor color)
             { return self.ColorToStr(color); })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__str__", [](const chess::Piece &self)
             { return self.PrettyStr(); });

    py::class_<chess::BoardLocation>(m, "BoardLocation")
        .def(py::init<>())
        .def(py::init<int8_t, int8_t>())
        .def("GetRow", &chess::BoardLocation::GetRow)
        .def("GetCol", &chess::BoardLocation::GetCol)
        .def("__eq__", [](const chess::BoardLocation &self, const chess::BoardLocation &other)
             { return self == other; })
        .def("__hash__", [](const chess::BoardLocation &self)
             { return std::hash<int>()(self.GetRow()) ^ std::hash<int>()(self.GetCol()); })
        .def("__str__", [](const chess::BoardLocation &self)
             { return self.PrettyStr(); });

    py::class_<chess::CastlingRights>(m, "CastlingRights")
        .def(py::init<>())
        .def(py::init<bool, bool>())
        .def("Kingside", &chess::CastlingRights::Kingside)
        .def("Queenside", &chess::CastlingRights::Queenside)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<chess::PlacedPiece>(m, "PlacedPiece")
        .def(py::init<>())
        .def(py::init<const chess::BoardLocation &, const chess::Piece &>())
        .def("GetLocation", &chess::PlacedPiece::GetLocation)
        .def("GetPiece", &chess::PlacedPiece::GetPiece)
        .def("__str__", [](const chess::PlacedPiece &self)
             { return self.PrettyStr(); });

    py::class_<chess::EnpassantInitialization>(m, "EnpassantInitialization")
        .def(py::init<>())
        .def_readwrite("enp_moves", &chess::EnpassantInitialization::enp_moves);

    py::class_<chess::IAlphaBetaPlayer>(m, "IAlphaBetaPlayer");

    py::class_<chess::AlphaBetaPlayer, chess::IAlphaBetaPlayer>(m, "AlphaBetaPlayer")
        .def(py::init<>())
        .def(py::init<std::optional<chess::PlayerOptions>>(), py::arg("options") = std::nullopt);

    py::class_<fpchess::Move, std::shared_ptr<fpchess::Move>>(m, "Move")
        .def(py::init<const chess::Move &>(), py::arg("c_move"))
        .def(py::init<int, chess::BoardLocation>(), py::arg("action_plane"), py::arg("from"))
        .def(py::init<int>(), py::arg("flat_index"))

        // Standard move constructor
        .def(py::init<chess::BoardLocation, chess::BoardLocation, chess::Piece, chess::CastlingRights, chess::CastlingRights>(),
             py::arg("from"), py::arg("to"), py::arg("standard_capture") = chess::Piece::kNoPiece,
             py::arg("initial_castling_rights") = chess::CastlingRights::kMissingRights,
             py::arg("castling_rights") = chess::CastlingRights::kMissingRights)

        // Pawn move constructor
        .def(py::init<chess::BoardLocation, chess::BoardLocation, chess::Piece, chess::BoardLocation, chess::Piece, chess::PieceType>(),
             py::arg("from"), py::arg("to"), py::arg("standard_capture"), py::arg("en_passant_location"),
             py::arg("en_passant_capture"), py::arg("promotion_piece_type") = chess::PieceType::NO_PIECE)

        // Castling constructor
        .def(py::init<chess::BoardLocation, chess::BoardLocation, chess::SimpleMove, chess::CastlingRights, chess::CastlingRights>(),
             py::arg("from"), py::arg("to"), py::arg("rook_move"), py::arg("initial_castling_rights"), py::arg("castling_rights"))

        .def_readonly_static("board_size", &fpchess::Move::board_size)
        .def_readonly_static("num_queen_moves_per_direction", &fpchess::Move::num_queen_moves_per_direction)
        .def_readonly_static("num_queen_moves", &fpchess::Move::num_queen_moves)
        .def_readonly_static("num_knight_moves", &fpchess::Move::num_knight_moves)

        .def("From", &fpchess::Move::From)
        .def("To", &fpchess::Move::To)
        .def("GetIndex", &fpchess::Move::GetIndex)
        .def("GetFlatIndex", &fpchess::Move::GetFlatIndex)
        .def("__str__",
             [](const fpchess::Move &move)
             {
                 std::ostringstream os;
                 os << move;
                 return os.str();
             });

    py::class_<fpchess::Board, std::shared_ptr<fpchess::Board>>(m, "Board")
        .def(py::init<const fpchess::Board &>())
        .def(py::init<chess::Player,
                      std::unordered_map<chess::BoardLocation, chess::Piece>,
                      std::optional<std::unordered_map<chess::Player, chess::CastlingRights>>,
                      std::optional<chess::EnpassantInitialization>>(),
             py::arg("turn"),
             py::arg("location_to_piece"),
             py::arg("castling_rights") = std::nullopt,
             py::arg("enp") = std::nullopt)
        //.def(py::init<chess::Player, std::unordered_map<chess::BoardLocation, chess::Piece>, std::unordered_map<chess::Player, chess::CastlingRights>, chess::EnpassantInitialization>())

        .def_readonly_static("board_size", &fpchess::Board::board_size)
        .def_readonly_static("num_state_channels", &fpchess::Board::num_state_channels)
        .def_readonly_static("state_space_size", &fpchess::Board::state_space_size)
        .def_readonly_static("num_action_channels", &fpchess::Board::num_action_channels)
        .def_readonly_static("action_space_size", &fpchess::Board::actionSpaceSize)
        .def_readonly_static("action_space_dims", &fpchess::Board::action_space_dims)
        .def_readonly_static("state_space_dims", &fpchess::Board::state_space_dims)
        .def_readonly_static("start_fen", &fpchess::Board::start_fen)

        .def("GetTurn", &fpchess::Board::GetTurn)
        .def("SetTurn", &fpchess::Board::SetTurn)
        .def("GetTerminated", &fpchess::Board::GetTerminated)
        .def("GetOpponentValue", &fpchess::Board::GetOpponentValue)
        .def("GetPieceAt", &fpchess::Board::GetLocationToPiece, py::arg("x"), py::arg("y"))
        .def("GetBoardLocation", &fpchess::Board::GetBoardLocation, py::arg("x"), py::arg("y"))
        .def("GetPieces", &fpchess::Board::GetPieces)
        .def("GetRoot", &fpchess::Board::GetRoot)
        .def("SetRoot", &fpchess::Board::SetRoot)
        .def("GetNode", &fpchess::Board::GetNode)
        .def("SetNode", &fpchess::Board::SetNode)
        .def("GetMemory", &fpchess::Board::GetMemory)
        .def("AppendToMemory", &fpchess::Board::AppendToMemory)
        .def(
            "GetGameResult", [](fpchess::Board &self, std::optional<chess::Player> opt_player)
            { return self.GetGameResult(opt_player); },
            py::arg_v("opt_player", std::nullopt, "Optional player turn"))
        .def("GetSimpleState", &fpchess::Board::GetSimpleState)
        .def("IsMoveLegal", &fpchess::Board::IsMoveLegal)
        .def("GetLegalMoves", &fpchess::Board::GetLegalMoves)
        .def("Eval", &fpchess::Board::Eval)
        .def("GetAttackedSquares", &fpchess::Board::GetAttackedSquares)
        .def("IsAttackedByPlayer", &fpchess::Board::IsAttackedByPlayer)
        .def("TakeAction", &fpchess::Board::TakeAction)
        .def("ParseActionspace", &fpchess::Board::ParseActionspace)

        .def_static("GetOpponent", py::overload_cast<const chess::PlayerColor &>(&fpchess::Board::GetOpponent))
        .def_static("GetOpponent", py::overload_cast<const chess::Player &>(&fpchess::Board::GetOpponent))
        .def_static("ChangePerspective", &fpchess::Board::ChangePerspective)
        .def_static("GetEncodedStates", &fpchess::Board::GetEncodedStates)
        .def_static("GetLegalMovesMask", &fpchess::Board::GetLegalMovesMask)
        .def("__str__", [](const fpchess::Board &board)
             {
                std::ostringstream os;
                os << board;
                return os.str(); });

    py::class_<fpchess::Node, std::shared_ptr<fpchess::Node>>(m, "Node")
        .def(py::init<const std::map<std::string, double> &,
                      std::shared_ptr<fpchess::Board>,
                      chess::PlayerColor,
                      std::shared_ptr<fpchess::Node>,
                      std::shared_ptr<fpchess::Move>,
                      double,
                      int>(),
             py::arg("args"), py::arg("state"), py::arg("turn"),
             py::arg("parent") = nullptr, py::arg("action_taken") = nullptr,
             py::arg("prior") = 0.0, py::arg("visit_count") = 0)
        .def("GetMoveMade", &fpchess::Node::GetMoveMade)
        .def("GetState", &fpchess::Node::GetState)
        .def("GetSimpleState", &fpchess::Node::GetSimpleState)
        .def("GetTurn", &fpchess::Node::GetTurn)
        .def("GetChildren", &fpchess::Node::GetChildren)
        .def("GetVisitCount", &fpchess::Node::GetVisitCount)
        .def("SetVisitCount", &fpchess::Node::SetVisitCount)
        .def("IsFullyExpanded", &fpchess::Node::IsFullyExpanded)
        .def("SelectChild", &fpchess::Node::SelectChild)
        .def("Expand", &fpchess::Node::Expand)
        .def("Backpropagate", &fpchess::Node::Backpropagate);
}