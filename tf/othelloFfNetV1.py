import tensorflow as tf
import numpy as np

class othello_net():
    @staticmethod
    def weight_variables(shape):
      initial = tf.truncated_normal(shape)
      return tf.Variable(initial)

    @staticmethod
    def bias_variables(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    
    def __init__(self, session, parent=None):
        if parent:
            pass #some form of deepcopy + modif
        else:
            self.initialize_conv1_weights()
            self.initialize_fc_weights()
            self.initialize_board_placeholders()
            self.initialize_training_placeholders()
            self.initialize_turn_boards()
            self.initialize_train_vars()
            self.initialize_convs()
            self.initialize_ff()
            self.accum_grads = []
            self.discount_factor = 0.9
            self.lambdaa = 0.7
            self.loss = 0.5*tf.reduce_mean(tf.square(tf.reduce_max(self.h_out) - self.target_val))
            self.opt = tf.train.GradientDescentOptimizer(1e-2)
            self.vars_list = [self.conv1_weights, self.conv1_bias, 
                self.conv2_weights, self.conv2_bias, self.conv_diag_weights, self.conv_diag_bias,
                self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias, self.out_weights, self.out_bias]
            self.grad_var_list = self.opt.compute_gradients(self.loss, self.vars_list)
            self.accum_grad = [(self.update_lambda_grads(g_v[0], index), g_v[1]) for index, g_v in enumerate(self.grad_var_list)]
            self.apply_lambda_gradients = self.opt.apply_gradients(self.accum_grad)
            self.boards_history = []
            session.run(tf.initialize_all_variables())
 
    # TODO if we ever get to do the evolutionary variant, tic toc...
    @classmethod
    def spawn(cls, parent):
        return cls(parent)


    def reset_for_game(self):
        self.accum_grads = []
        self.boards_history = []

    def evaluate(self, boards, session):
        boards = np.asarray(boards)
        if boards.ndim == 2:
            boards = np.expand_dims(boards, axis=0)
        batch_size = boards.shape[0]
        diag = self.get_diagonal(boards)
        keep_prob = 1
        v = session.run(self.h_out, feed_dict={self.boards:boards, 
            self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size})
        idx = np.argmax(v, 0)
        self.boards_history.append(boards[idx])
        print('max value v at index idx is: ', v[idx][0][0], idx[0])
        return idx[0], v[idx][0][0] # TODO this looks ugly fix upstream

    def initialize_conv1_weights(self):
        # create 8 by 1 filter - row/col
        self.conv1_weights = self.weight_variables([1, 8, 4, 10])
        self.conv1_bias = self.bias_variables([1, 1, 1, 10])
        
        # Chop 6x5 part of the board and slide a 3x3 window over it
        self.conv2_weights = self.weight_variables([3, 3, 8, 10])
        self.conv2_bias = self.bias_variables([1, 1, 1, 10])
        
        # a bit of a mis-use of convenets but pass the diagonals into a single input
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

    def initialize_train_vars(self):
        self.epochs = tf.Variable(0)

    def initialize_board_placeholders(self):
        self.boards = tf.placeholder(tf.float32, shape=[None, 8, 8])
        self.boards_diag = tf.placeholder(tf.float32, shape=[None, 8, 1, 4])

    def initialize_training_placeholders(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.target_val = tf.placeholder(tf.float32)

    def initialize_turn_boards(self):
        self.sym1 = tf.reverse(self.boards, [False, True, False])
        self.sym2 = tf.reverse(self.boards, [False, False, True])
        self.sym3 = tf.reverse(self.sym2, [False, True, False])
        self.sym4 = tf.transpose(self.boards, perm=[0, 2, 1])
        self.sym5 = tf.transpose(self.sym1, perm=[0, 2, 1])
        self.sym6 = tf.transpose(self.sym2, perm=[0, 2, 1])
        self.sym7 = tf.transpose(self.sym3, perm=[0, 2, 1])
        self.board_sym_tensor = tf.transpose([self.boards, self.sym1, self.sym2, self.sym3, self.sym4, 
            self.sym5, self.sym6, self.sym7], [1,2,3,0])    
        self.boards_half_sym = tf.slice(self.board_sym_tensor, [0, 0, 0, 0], [-1, -1, -1, 4])
        self.boards_chopped = tf.slice(self.board_sym_tensor, [0, 0, 0, 0], [-1, 5, 5, -1])

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
        conv_out = tf.concat(1, [conv1_flat, conv2_flat, conv_diag_flat])
        self.h_fc1 = tf.nn.tanh(tf.matmul(conv_out, self.fc1_weights)+self.fc1_bias)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.h_fc2 = tf.nn.tanh(tf.matmul(self.h_fc1_drop, self.fc2_weights) + self.fc2_bias)
        self.h_fc2_drop = tf.nn.dropout(self.h_fc2, self.keep_prob)
        self.h_out = tf.nn.tanh(tf.matmul(self.h_fc2_drop, self.out_weights) + self.out_bias)
    
    # TODO need to find a store gradient corresponding to a weight in order for TD(lambda)
    # Do TD(0) for now, return grad instead of self.lambdaa * self.accum_grads[?] + grad
    # otherwise, accumulate lambda gradients from turn 1
    def update_lambda_grads(self, grad, idx):
        if not self.accum_grads or idx == self.accum_grads.__len__():
            self.accum_grads.append(grad)
            return grad
        else:
            self.accum_grads[idx].__mul__(self.lambdaa)
            self.accum_grads[idx] = tf.add(self.accum_grads[idx], grad)
            #self.accum_grads[idx] *= self.lambdaa
            #self.accum_grads[idx] += grad
            #print(self.accum_grads[idx])
            return self.accum_grads[idx]
            

    def train(self, outcome, session):
        #if self.apply_lambda_gradients is None:
        #    self.accum_grad = [(self.update_lambda_grads(g_v[0], index), g_v[1]) for index, g_v in enumerate(self.grad_var)]
        #    self.apply_lambda_gradients = self.opt.apply_gradients(self.accum_grad)
        gamma = self.discount_factor
        # add one to both scores for smoothing (avoid divide by zeros)
        squashed_outcome = np.tanh((outcome['net'] + 1)/(outcome['opponent'] + 1))
        num_moves = self.boards_history.__len__()
        total_loss = []
        all_gammas = 0
        for m in range(num_moves):
            all_gammas += gamma**m
        discounted_score = squashed_outcome/all_gammas
        target_series = []
        target_t = 0
        temp = discounted_score
        print('Epoch: ', self.epochs.eval(session=session), 'Training on moves, might take a moment')
        for m in range(num_moves):
            #print(m, end=', ')
            target_t += temp
            temp *= gamma
            batch_size = 1
            keep_prob = 0.7
            diag = self.get_diagonal(self.boards_history[m])
            # grad_var = self.grad_var
            self.accum_grad = [(self.update_lambda_grads(g_v[0], index), g_v[1]) for index, g_v in enumerate(self.grad_var_list)]
            # TODO print loss
            total_loss.append(self.loss.eval(session=session, feed_dict={self.boards:self.boards_history[m], 
                self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size, self.target_val:target_t}))
            session.run(self.apply_lambda_gradients, feed_dict={self.boards:self.boards_history[m], 
                self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size, self.target_val:target_t})
            #session.run(self.opt.apply_gradients(grad_var), feed_dict={self.boards:self.boards_history[m], 
            #    self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size, self.target_val:target_t})
        print('Total loss accross turns: ', np.sum(total_loss))    
        self.epochs += 1
        return(np.sum(total_loss))

