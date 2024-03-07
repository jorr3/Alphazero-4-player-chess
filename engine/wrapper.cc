#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include "board.h"
#include "player.h"

namespace py = pybind11;

template <typename Func>
auto exception_wrapper(Func &&func)
{
    try
    {
        return func();
    }
    catch (const std::runtime_error &e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        throw py::error_already_set();
    }
    catch (const std::invalid_argument &e)
    {
        PyErr_SetString(PyExc_ValueError, e.what());
        throw py::error_already_set();
    }
    catch (...)
    {
        PyErr_SetString(PyExc_Exception, "An unknown error occurred");
        throw py::error_already_set();
    }
}

std::string board_to_string(const chess::Board &board)
{
    std::ostringstream os;
    os << board;
    return os.str();
}

PYBIND11_MODULE(chessenv, m)
{

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
        .def("__str__", [](const chess::PlacedPiece &self)
             { return self.PrettyStr(); });

    py::class_<chess::CastlingRights>(m, "CastlingRights")
        .def(py::init<>())
        .def(py::init<bool, bool>())
        .def("Kingside", &chess::CastlingRights::Kingside)
        .def("Queenside", &chess::CastlingRights::Queenside)
        .def(py::self == py::self)
        .def(py::self != py::self);

    py::class_<chess::Move>(m, "Move")
        // Default constructor
        .def(py::init<>())

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

        .def("From", &chess::Move::From)
        .def("To", &chess::Move::To);

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

    py::class_<chess::Board>(m, "Board")
        .def(py::init<chess::Player, std::unordered_map<chess::BoardLocation, chess::Piece>>())
        .def(py::init<chess::Player, std::unordered_map<chess::BoardLocation, chess::Piece>, std::unordered_map<chess::Player, chess::CastlingRights>, chess::EnpassantInitialization>())
        .def(py::init<const chess::Board &>())
        .def("GetTurn", &chess::Board::GetTurn)
        .def("SetTurn", &chess::Board::SetTurn)
        .def("GetState", &chess::Board::GetState)
        .def("GetPieceAt", &chess::Board::GetLocationToPiece, py::arg("x"), py::arg("y"))
        .def("GetBoardLocation", &chess::Board::GetBoardLocation, py::arg("x"), py::arg("y"))
        .def("GetPieces", &chess::Board::GetPieces)
        .def("IsMoveLegal", &chess::Board::IsMoveLegal, py::arg("move"))
        .def("GetLegalMoves", &chess::Board::GetLegalMoves)
        .def(
            "GetGameResult", [](chess::Board &self, std::optional<chess::Player> opt_player)
            { return self.GetGameResult(opt_player); },
            py::arg_v("opt_player", std::nullopt, "Optional player turn"))
        .def("Eval", &chess::Board::Eval, py::arg("player"), py::arg("options"))
        .def("MakeMove", &chess::Board::MakeMove, py::arg("move"))
        .def("__str__", &board_to_string);

    py::class_<chess::IAlphaBetaPlayer>(m, "IAlphaBetaPlayer");

    py::class_<chess::AlphaBetaPlayer, chess::IAlphaBetaPlayer>(m, "AlphaBetaPlayer")
        .def(py::init<>())
        .def(py::init<std::optional<chess::PlayerOptions>>(), py::arg("options") = std::nullopt);
}
