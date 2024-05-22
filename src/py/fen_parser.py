from typing import List, Optional, Tuple
from alphazero_cpp import *

def split_str_on_whitespace(x: str) -> List[str]:
    return x.split()

def split_str(s: str, delimiter: str) -> List[str]:
    return s.split(delimiter)

def parse_int(input: str) -> Optional[int]:
    try:
        return int(input)
    except ValueError:
        return None

def parse_castling_availability(fen_substr: str) -> Optional[List[bool]]:
    parts = split_str(fen_substr, ",")
    if len(parts) != 4:
        return None
    availability = [part == "1" for part in parts]
    if len(availability) != 4:
        return None
    return availability

def parse_enp_location(enp: str, board_size: int) -> Optional[BoardLocation]:
    pos = enp.find(':')
    if pos == -1:
        return None
    to = enp[pos+1:]
    if to.endswith("'"):
        to = to[:-1]
    if len(to) < 2 or len(to) > 3:
        return None
    col = ord(to[0]) - ord('a')
    if col < 0 or col > 13:
        return None
    row = int(to[1:])
    if row < 0 or row > 13:
        return None
    row = board_size - row
    return BoardLocation(row, col)

def parse_location(move_str: str, start: int, board_size: int) -> Optional[Tuple[int, BoardLocation]]:
    if start < len(move_str) and (move_str[start] == '-' or move_str[start] == 'x'):
        start += 1
    if len(move_str) < start + 2:
        return None

    c = move_str[start]
    if c in ['K', 'Q', 'N', 'B', 'R']:
        start += 1

    col = ord(move_str[start]) - ord('a')
    if col < 0 or col >= board_size:
        return None
    start += 1

    row = int(move_str[start:])
    row = board_size - row
    return start, BoardLocation(row, col)

def parse_promotion(move_str: str, start: int) -> Optional[Tuple[int, str]]:
    if start >= len(move_str):
        return start, 'NO_PIECE'
    if move_str[start] == '=':
        start += 1
    if start >= len(move_str):
        return None
    c = move_str[start]
    if c in ['N', 'n']:
        return start + 1, 'KNIGHT'
    elif c in ['B', 'b']:
        return start + 1, 'BISHOP'
    elif c in ['R', 'r']:
        return start + 1, 'ROOK'
    elif c in ['Q', 'q']:
        return start + 1, 'QUEEN'
    else:
        return None

def parse_move(board: 'Board', move_str: str, board_size: str) -> Optional['Move']:
    from_location = parse_location(move_str, 0, board_size)
    if not from_location:
        return None
    to_location = parse_location(move_str, from_location[0], board_size)
    if not to_location:
        return None
    promotion = parse_promotion(move_str, to_location[0])
    if not promotion:
        return None

    from_loc = from_location[1]
    to_loc = to_location[1]
    promotion_piece_type = promotion[1]

    moves = board.get_pseudo_legal_moves2(300)
    for move in moves:
        if move.from_location() == from_loc and move.to_location() == to_loc and move.get_promotion_piece_type() == promotion_piece_type:
            return move
    return None

def parse_board_args_from_fen(fen: str, board_size: int) -> Optional['Board']:
    parts = split_str(fen, "-")
    if len(parts) < 7 or len(parts) > 8:
        return None

    player_str = parts[0]
    castling_availability_kingside = parts[2]
    castling_availability_queenside = parts[3]
    piece_placement = parts[-1]

    enpassant = parts[6] if len(parts) == 8 else ""

    # Parse player
    if len(player_str) != 1:
        return None
    pchar = player_str[0]
    if pchar == 'R':
        player = Player(RED)
    elif pchar == 'B':
        player = Player(BLUE)
    elif pchar == 'Y':
        player = Player(YELLOW)
    elif pchar == 'G':
        player = Player(GREEN)
    else:
        return None

    # Parse castling availability
    kingside = parse_castling_availability(castling_availability_kingside)
    if not kingside:
        return None
    queenside = parse_castling_availability(castling_availability_queenside)
    if not queenside:
        return None

    castling_rights = {}
    for player_color in range(4):
        pl = Player(PlayerColor(player_color))
        castling_rights[pl.GetColor()] = CastlingRights(kingside[player_color], queenside[player_color])

    # Parse enpassant
    enp = EnpassantInitialization()
    if enpassant:
        parts = split_str(enpassant[1:-1], ",")
        if len(parts) != 4:
            return None
        for i in range(4):
            enp_location = parse_enp_location(parts[i], board_size)
            if enp_location:
                to = enp_location
                from_row, from_col = to.get_row(), to.get_col()
                if i == RED:
                    from_row += 2
                elif i == BLUE:
                    from_col -= 2
                elif i == YELLOW:
                    from_row -= 2
                elif i == GREEN:
                    from_col += 2
                enp.enp_moves[i] = Move(BoardLocation(from_row, from_col), to)

    # Parse piece placement
    rows = split_str(piece_placement, "/")
    if len(rows) != board_size:
        return None
    location_to_piece = {}
    for row in range(len(rows)):
        cols = split_str(rows[row], ",")
        col = 0
        for col_str in cols:
            if not col_str:
                return None
            ch = col_str[0]
            if ch in ['r', 'b', 'y', 'g']:
                if len(col_str) != 2:
                    return None
                location = BoardLocation(row, col)
                player_color = {'r': RED, 'b': BLUE, 'y': YELLOW, 'g': GREEN}[ch]
                piece_type = {'P': PAWN, 'R': ROOK, 'N': KNIGHT, 'B': BISHOP, 'K': KING, 'Q': QUEEN}[col_str[1]]
                piece = Piece(Player(PlayerColor(player_color)), piece_type)
                location_to_piece[location] = piece
                col += 1
            elif ch == 'x':
                col += 1
            else:
                num_empty = parse_int(col_str)
                if not num_empty or num_empty <= 0:
                    return None
                col += num_empty

    # board = Board(player, location_to_piece, castling_rights, enp)
    # TODO: fix this castling_rights and enp inputs!!
    return player, location_to_piece

