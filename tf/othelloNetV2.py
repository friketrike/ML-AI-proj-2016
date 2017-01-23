import tensorflow as tf
import numpy as np

class othello_net():
    @staticmethod
    def weight_variables(shape, name=None):
      initial = tf.random_normal(shape, stddev=0.1, name=name)
      return tf.Variable(initial)

    @staticmethod
    def bias_variables(shape, name=None):
      initial = tf.constant(1.0, shape=shape, name=name)
      return tf.Variable(initial)
    
    def __init__(self, session, parent=None):
        if parent:
            pass # TODO for GAs some form of deepcopy + modif
        else:
            self.initialize_conv1_weights()
            self.initialize_fc_weights()
            self.initialize_board_placeholders()
            self.initialize_training_placeholders()
            self.initialize_train_vars()
            self.initialize_convs()
            self.initialize_ff()
            self.accum_grads = []
            self.discount_factor = 0.999
            self.lambdaa = 0.1
            self.opt = tf.train.AdamOptimizer(1.0)
            self.vars_list = [self.conv1_weights, self.conv1_bias, 
                              self.conv2_weights, self.conv2_bias,
                              self.conv_diag_weights, self.conv_diag_bias,
                              self.fc1_weights, self.fc1_bias, self.fc2_weights,
                              self.fc2_bias, self.out_weights, self.out_bias]
            self.grad_var_list = self.opt.compute_gradients(self.h_out, self.vars_list)
            self.initialize_accum_grad_vars()
            self.append_lambda_op()
            self.values_history = []
            session.run(tf.initialize_all_variables())
 
    # TODO if we ever get to do the evolutionary variant, tic toc... time is pressing
    @classmethod
    def spawn(cls, parent):
        return cls(parent)

    # When we want to start a new match we need to clear some variables
    def reset_for_game(self):
        _ = [self.lambda_resets[idx] for idx in range(self.gaccum_list.__len__())]
        self.values_history = []

    # give an appraisal given a board configuration, train if told to do so
    def evaluate(self, boards, session, train=False, verbose=False):
        boards = np.asarray(boards)
        if boards.ndim == 2:
            boards = np.expand_dims(boards, axis=0)
        batch_size = boards.shape[0]
        diag = self.get_diagonal(boards)
        half_sym, chopped = self.turn_boards(boards)
        if train:
            v, grad_var_list = session.run([self.h_out, self.grad_var_list],
                                           feed_dict={self.boards_half_sym: half_sym,
                                                    self.boards_chopped: chopped,
                                                    self.boards_diag: diag, 
                                                    self.keep_prob: 1.0, # don't play with this while it doesn't work
                                                    self.batch_size: batch_size}) # self.keep_prob: 1,
            idx = np.argmax(v, 0)
            self.values_history.append(v[idx])
            _ = [self.lambda_updates[idx] for idx in range(self.gaccum_list.__len__())]
        else:
            v = session.run(self.h_out, feed_dict={self.boards_half_sym: half_sym,
                                                   self.boards_chopped: chopped,
                                                   self.boards_diag: diag, 
                                                   self.keep_prob: 1,
                                                   self.batch_size: batch_size}) #self.keep_prob: 1,
            idx = np.argmax(v, 0)
        if verbose:
            print('max value v at index idx is: ', v[idx][0][0], idx[0])
        return idx[0], v[idx][0][0] # TODO this looks ugly fix upstream

    # NOTE if training, this should be called at the end of a match
    def learn_from_outcome(self, tally, session, verbose=False):
        # TODO verify the sign of this error!
        error = (self.values_history[-1] - np.tanh(tally/16))
        if verbose:
            print('Loss at end-game: ', error) # want to know this for debugging purposes
        for idx, vars in enumerate(self.vars_list):
            session.run(self.gradient_applications[idx], feed_dict={self.error: error})
        return error[0][0]

    # TODO fix this, for some reason it's broken on the tf side
    def set_epochs(self, epochs):
        self.epochs.assign(epochs)

    def initialize_conv1_weights(self):
        # create 8 by 1 filter - row/col
        self.conv1_weights = self.weight_variables([1, 8, 4, 10])
        self.conv1_bias = self.bias_variables([1, 1, 1, 10])
        
        # Chop 5x5 part of the board and slide a 3x3 window over it
        self.conv2_weights = self.weight_variables([3, 3, 8, 10])
        self.conv2_bias = self.bias_variables([1, 1, 1, 10])
        
        # a bit of a mis-use of conv-nets but pass the diagonals into a single input
        # the useful bits of input channels and features are the reason for this
        self.conv_diag_weights = self.weight_variables([1, 1, 4, 4])
        self.conv_diag_bias = self.bias_variables([1, 1, 1, 4])

    # Try with 64/32 1st layer, second layer neurons
    def initialize_fc_weights(self):
        self.fc1_weights = self.weight_variables([174, 64])
        self.fc1_bias = self.bias_variables([1,64])

        self.fc2_weights = self.weight_variables([64, 32])
        self.fc2_bias = self.bias_variables([1, 32])

        self.out_weights = self.weight_variables([32,1])
        self.out_bias = self.bias_variables([1,1])

    def initialize_accum_grad_vars(self):
        self.gaccum_list = []
        for gv in self.grad_var_list:
            g = tf.Variable(tf.zeros(gv[0].get_shape()))
            self.gaccum_list.append(g)

    def append_lambda_op(self):
        self.lambda_updates = []
        self.lambda_resets = []
        self.gradient_applications = []
        for idx, gv in enumerate(self.grad_var_list):
            self.lambda_updates.append(
                tf.add(self.gaccum_list[idx].__mul__(self.lambdaa),
                       gv[0]))
            self.lambda_resets.append(self.gaccum_list[idx].__mul__(0))
            self.gradient_applications.append(
                self.opt.apply_gradients(
                    [(self.gaccum_list[idx].__mul__(self.error), gv[1])]))

    def initialize_train_vars(self):
        self.epochs = tf.Variable(0)

    def initialize_board_placeholders(self):
        self.boards_half_sym = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
        self.boards_chopped = tf.placeholder(tf.float32, shape=[None, 5, 5, 8])
        self.boards_diag = tf.placeholder(tf.float32, shape=[None, 8, 1, 4])

    def initialize_training_placeholders(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.error = tf.placeholder(tf.float32)

    @staticmethod
    def turn_boards(boards):
        boards = np.asarray(boards)
        sym1 = boards[:,::-1,::]
        sym2 = boards[:,:,::-1]
        sym3 = sym2[:,::-1, ::]
        sym4 = np.transpose(boards, (0, 2, 1))
        sym5 = np.transpose(sym1, (0, 2, 1))
        sym6 = np.transpose(sym2, (0, 2, 1))
        sym7 = np.transpose(sym3, (0, 2, 1))
        full_sym = np.stack([boards, sym1, sym2, sym3, sym4, sym5, sym6, sym7], axis=3)
        half_sym = full_sym[:, :, :, 0:4]
        chopped = full_sym[:, 0:5, 0:5, :]
        return half_sym, chopped

    @staticmethod
    def get_diagonal(boards):
        boards = np.asarray(boards)
        diags1 = np.expand_dims(boards.diagonal(0, 1, 2), axis=2)
        diags2 = np.expand_dims(np.fliplr(boards).diagonal(0,2,1), axis=2)
        # Yes, lr does the trick, flipud actually reverses batches (dimension 0)
        diags3 = np.fliplr(diags1)
        diags4 = np.fliplr(diags2)
        return np.stack([diags1, diags2, diags3, diags4], axis=3)

    def initialize_convs(self):
        self.h_conv1 = tf.nn.tanh(tf.nn.conv2d(self.boards_half_sym, 
            self.conv1_weights, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias)

        self.h_conv2 = tf.nn.tanh(tf.nn.conv2d(self.boards_chopped,
            self.conv2_weights, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_bias)

        self.h_conv_diag = tf.nn.tanh(tf.nn.conv2d(self.boards_diag, self.conv_diag_weights, 
            strides = [1, 8, 1, 1], padding='VALID') + self.conv_diag_bias)

    def initialize_ff(self):
        conv1_flat = tf.reshape(self.h_conv1, [-1, 1*8*10])
        conv2_flat = tf.reshape(self.h_conv2, [-1, 3*3*10])
        conv_diag_flat = tf.reshape(self.h_conv_diag, [-1, 1*1*4])
        conv_out = tf.concat([conv1_flat, conv2_flat, conv_diag_flat], 1)
        self.h_fc1 = tf.nn.tanh(tf.matmul(conv_out, self.fc1_weights)+self.fc1_bias)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.h_fc2 = tf.nn.tanh(tf.matmul(self.h_fc1_drop, self.fc2_weights) + self.fc2_bias)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)
        self.h_out = tf.nn.tanh(tf.matmul(self.h_fc2_drop, self.out_weights) + self.out_bias)

