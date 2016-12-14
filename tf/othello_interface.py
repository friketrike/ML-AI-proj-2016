import othello as o
import position as p
import board as b
import othello_node as n
import tensorflow as tf
import othelloFfNetV1 as otnet
import time

PLY = 1

session = tf.Session()
game = o.game()
on = otnet.othello_net(session)


def batch(turn):
    board_now = game.board
    possible_moves = board_now.get_valid_moves(turn)
    boards = []
    for m in possible_moves:
        new_board = b.Board(board_now)
        new_board.do_move(m, turn)
        boards.append(new_board.squares)
    return boards, possible_moves

def play_net(train=False):
    done = False
    value_series = []
    tic = time.time()
    while not done:
        #net plays black TODO allow for playing white
        boards, moves = batch(b.BLACK)
        if boards:
            idx, v = on.evaluate(boards, session)
            game.play_move(moves[idx], b.BLACK)
            value_series.append(v)
        else:
            game.pass_moves += 1
        done = not game.play_random_turn(b.WHITE)
    outcome = game.board.get_score()
    if train:
        # change outcome so that later on we can have the net play either white or black
        color_blind_outcome = {'net':outcome['Black'], 'opponent':outcome['White']}
        on.train(color_blind_outcome, session)
    game.reset()
    on.reset_for_game()
    print('The round took:', time.time()-tic, ' seconds.')
    return outcome


def save_checkpoint(path="./otnet.ckpt"):
    saver = tf.train.Saver()
    save_path = saver.save(session, path)
    print('Saved checkpoint')

def apply_action(values, root):
    idx = values.index(min(values))
    move = root.children[idx].move
    game.play_move(move, root.turn)


def create_tree():
    root = n.Node(game.board, game.turn)
    create_sub_tree(root)
    return root


def create_sub_tree(root):
    if root.depth >= PLY:
        return
    moves = root.board.get_valid_moves(game.turn)
    for move in moves:
        new_node = n.Node.child_node(root, move)
        root.insert_child(new_node)
        create_sub_tree(new_node)