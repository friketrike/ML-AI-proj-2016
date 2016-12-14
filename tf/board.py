import copy
import position as p

BOARD_SIZE = 8
BLACK = 1
WHITE = -1


def opposite(turn):
    return turn * -1


class Board:
    directions = {'up': p.Pos.up, 'left': p.Pos.left, 'up_left': p.Pos.up_left, 'up_right': p.Pos.up_right,
                  'down_left': p.Pos.down_left, 'down_right': p.Pos.down_right, 'right': p.Pos.right, 'down': p.Pos.down}

    def __init__(self, prev_board=None, single_list_board=None):
        if prev_board:
            self.squares = copy.deepcopy(prev_board.squares)
        elif single_list_board:
            assert single_list_board.__len__() == BOARD_SIZE * BOARD_SIZE
            self.squares = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    self.squares[row][col] = single_list_board[(row*BOARD_SIZE)+col]
        else:
            self.squares = [[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
            s = int(BOARD_SIZE/2)
            self.squares[s][s] = BLACK
            self.squares[s-1][s-1] = BLACK
            self.squares[s-1][s] = WHITE
            self.squares[s][s-1] = WHITE

    def to_string(self):
        the_str = "(\n"
        char_map = {BLACK: 'B', 0: '0', WHITE: 'W', 2: '*'}
        for idx_v, row in enumerate(self.squares):
            for idx_h, token in enumerate(row):
                if idx_h == 0:
                    the_str += "    ("
                the_str += char_map[token]
                if idx_h == 7:
                    the_str += ")" + '\n'
                else:
                    the_str += ", "
        the_str += ")" + '\n'
        return the_str

    def do_move(self, pos, turn):
        flips = self.get_flips(pos, turn)
        self.do_flips(flips)
        self.place_token(pos, turn)

    def do_flips(self, flips):
        for flip in flips:
            self.place_token(flip, opposite(self.get_token(flip)))

    def place_token(self, pos, token):
        self.squares[pos.v][pos.h] = token

    def get_token(self, pos):
        if not pos.is_valid():
            raise ValueError
        return self.squares[pos.v][pos.h]

    def get_valid_moves(self, turn):
        emptys = self.get_empty_squares()
        valid_moves = []
        for pos in emptys:
            for k in self.directions:
                new_pos = copy.deepcopy(pos)
                direc = self.directions[k]
                if direc(new_pos) and self.is_dir_valid(new_pos, direc, turn) \
                        and (valid_moves.count(pos) == 0):
                    valid_moves.append(pos)
        return valid_moves

    def get_flips(self, pos, turn):
        flips = []
        for k in self.directions:
            direc = self.directions[k]
            new_pos = copy.deepcopy(pos)
            direc(new_pos)
            while self.is_dir_valid(new_pos, direc, turn):
                flips.append(new_pos)
                new_pos = copy.deepcopy(new_pos)
                direc(new_pos)
        return flips

    def is_dir_valid(self, cur_pos, direc, turn):
        new_pos = copy.deepcopy(cur_pos)
        while new_pos.is_valid() and self.get_token(new_pos) == opposite(turn):
            if not direc(new_pos):
                break
        return new_pos.is_valid() and new_pos != cur_pos and self.get_token(new_pos) == turn

    def get_empty_squares(self):
        empty_positions = []
        for v, row in enumerate(self.squares):
            for h, square in enumerate(row):
                if self.squares[v][h] == 0:
                    pos = p.Pos(h, v)
                    empty_positions.append(pos)
        return empty_positions

    def get_score(self):
        black_count = 0
        white_count = 0
        for row in self.squares:
            for h in row:
                if h == BLACK:
                    black_count += 1
                elif h == WHITE:
                    white_count += 1
        return{'Black': black_count, 'White': white_count}