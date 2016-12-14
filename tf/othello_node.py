import board as b
#import position as p
#import othello as o

class Node:

    def __init__(self, board, turn, parent=None, move=None):
        self.board = board
        self.turn = turn
        self.children = []
        if not parent:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self.parent = parent
        self.move = move

    @classmethod
    def child_node(cls, parent, move):
        turn = b.opposite(parent.turn)
        board = b.Board(parent.board)
        board.do_move(move, parent.turn)
        return cls(board, turn, parent, move)

    def insert_child(self, node):
        self.children.append(node)

