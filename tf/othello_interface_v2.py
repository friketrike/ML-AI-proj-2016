# COMP 6321 Machine Learning, Fall 2016
# Federico O'Reilly Regueiro - 40012304
# Final project - othello with neural nets

import othello as o
import position as p
import board as b
import tensorflow as tf
import othelloNetV2 as otnet
import time
import random

session = tf.Session()
game = o.game()
on = otnet.othello_net(session)
score_series = []


def batch(color):
    board_now = game.board
    possible_moves = board_now.get_valid_moves(color)
    boards = []
    for m in possible_moves:
        new_board = b.Board(board_now)
        new_board.do_move(m, color)
        new_board.relativize(color)
        boards.append(new_board.squares)
    return boards, possible_moves


def play_net(train=False, verbose=False):
    color = random.choice((b.BLACK, b.WHITE))
    done = False
    tic = time.time()
    global score_series
    if verbose:
        print(game.board.to_string())
    if color == b.WHITE:
        done = not game.play_random_turn(b.opposite(color), verbose)
    while not done:
        boards, moves = batch(color)
        if boards:
            if verbose:
                print(game.turn_to_string(color), ' plays:')
            idx, v = on.evaluate(boards, session, train)
            game.play_move(moves[idx], color)
            if verbose:
                print(game.board.to_string())
        else:
            game.pass_moves += 1
        if verbose:
            print(game.turn_to_string(b.opposite(color)), '\'s turn:')
        done = not game.play_random_turn(b.opposite(color), verbose)
    outcome = game.board.get_score()
    color_blind_outcome = {'net': outcome['Black'], 'opponent': outcome['White'], 'error':0}
    if train:
        score_series.append(color_blind_outcome)
        error = on.learn_from_outcome(color_blind_outcome['net'] - color_blind_outcome['opponent'], session)
        color_blind_outcome['error'] = error
    game.reset()
    on.reset_for_game()
    if verbose:
        print('The round took:', time.time()-tic, ' seconds.')
        # print('Up to now: ', wins, ' wins, ', losses, ' losses and ', ties, ' ties.')
    return color_blind_outcome


def save_checkpoint(path="./otnet_v2.ckpt"):
    saver = tf.train.Saver()
    saver.save(session, path)
    print('Saved checkpoint')


def restore_checkpoint(path="./otnet_v2.ckpt"):
    saver = tf.train.Saver()
    saver.restore(session, path)
    print("Model restored.")
