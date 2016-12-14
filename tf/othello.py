
# bunch of imports
import random
import board as b

BLACK = b.BLACK
WHITE = b.WHITE

class game():
    def __init__(self, board=None, turn=None, pass_moves=None):
        self.board = board if board else b.Board()
        self.turn = turn if turn else BLACK
        self.pass_moves = pass_moves if pass_moves else 0

    def reset(self):
        self.board = b.Board()
        self.turn = BLACK
        self.pass_moves = 0

    @classmethod
    def started(cls, board, turn, pass_moves=None):
        p_m = pass_moves if pass_moves else 0
        return cls(board, turn, p_m)

    def play_random_turn(self, turn):
        if self.pass_moves >= 2:
            score = self.board.get_score()
            print('Game Over')
            print('Score: Black - ', score['Black'], ' White - ', score['White'])
            return False
        moves = self.board.get_valid_moves(turn)
        if not moves:
            print('no moves')
            self.pass_moves += 1
            return True
        self.pass_moves = 0
        move = random.choice(moves)
        self.board.do_move(move, turn)
        print(self.board.to_string())
        return True

    def play_move(self, move, turn):
        self.board.do_move(move, turn)

    def show_available_moves(self, turn):
        moves = self.board.get_valid_moves(turn)
        board_copy = b.Board(self.board)
        for move in moves:
            board_copy.place_token(move, 2)
        print(board_copy.to_string())

    def turn_to_string(turn):
        return 'Black' if (turn == b.BLACK) else 'White'
