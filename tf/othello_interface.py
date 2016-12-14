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
score_series = []
wins = 0
losses = 0
ties = 0

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
    tic = time.time()
    global wins
    global losses
    global ties
    global score_series
    print(game.board.to_string())
    while not done:
        #net plays black TODO allow for playing white
        boards, moves = batch(b.BLACK)
        if boards:
            print('Black plays:')
            idx, v = on.evaluate(boards, session)
            game.play_move(moves[idx], b.BLACK)
            print(game.board.to_string())
        else:
            game.pass_moves += 1
        print('White\'s turn:')
        done = not game.play_random_turn(b.WHITE)
    outcome = game.board.get_score()
    score_series.append(outcome)
    if score_series[-1]['Black'] > score_series[-1]['White']:
        wins += 1
    elif score_series[-1]['Black'] < score_series[-1]['White']:
        losses += 1
    else:
        ties += 1
    if train:
        # change outcome so that later on we can have the net play either white or black
        color_blind_outcome = {'net':outcome['Black'], 'opponent':outcome['White']}
        total_loss = on.train(color_blind_outcome, session)
        save_checkpoint()
    game.reset()
    on.reset_for_game()
    print('The round took:', time.time()-tic, ' seconds.')
    print('Up to now: ', wins, ' wins, ', losses, ' losses and ', ties, ' ties.')
    return outcome, total_loss


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
