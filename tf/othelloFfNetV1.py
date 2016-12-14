import tensorflow as tf

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
            self.initialize_convs()
            self.initialize_ff()
            self.accum_grads = []
            self.discount_factor = 0.9
            self.lambdaa = 0.6
            self.loss = tf.reduce_mean(tf.square(tf.reduce_max(self.h_out) - self.target_val))
            self.opt = tf.train.GradientDescentOptimizer(2e-3)
            self.grad_var = self.opt.compute_gradients(self.loss, [self.conv1_weights, self.conv1_bias, 
                self.conv2_weights, self.conv2_bias, self.conv_diag_weights, self.conv_diag_bias,
                self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias, self.out_weights, self.out_bias])
            self.accum_grad = [(self.update_lambda_grads(g_v[0], index), g_v[1]) for index, g_v in enumerate(self.grad_var)]
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

    #keep the possibility of having a batch of boards along the 0th dimension
    #even if for now we might only evaluate a single board
    def evaluate(self, boards, session):
        if (session.run(tf.rank(boards)) == 2):
            boards = [boards]
        batch_size = session.run(tf.shape(boards))[0]
        board_sym_tensor = self.turn_boards(boards)
        half_sym = session.run(tf.slice(board_sym_tensor, [0, 0, 0, 0], [-1, -1, -1, 4]))
        chopped = session.run(tf.slice(board_sym_tensor, [0, 0, 0, 0], [-1, 5, 5, -1]) )
        diag = session.run(self.get_diagonal(boards, batch_size))
        keep_prob = 1
        v = session.run(self.h_out, feed_dict={self.boards_half_sym:half_sym, self.boards_chopped:chopped, 
            self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size})
        idx = session.run(tf.arg_max(v, 0))
        #self.value_series.append(v[idx]) - can't do it this way as the optimizer can't compute gradients
        self.boards_history.append([boards[idx]])
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
        pass

    def initialize_board_placeholders(self):
        self.boards_half_sym = tf.placeholder(tf.float32, shape=[None, 8, 8, 4])
        self.boards_chopped = tf.placeholder(tf.float32, shape=[None, 5, 5, 8])
        self.boards_diag = tf.placeholder(tf.float32, shape=[None, 8, 1, 4])

    def initialize_training_placeholders(self):
        self.batch_size = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        self.target_val = tf.placeholder(tf.float32)

    @staticmethod
    def turn_boards(boards):
        sym1 = tf.reverse(boards, [False, True, False])
        sym2 = tf.reverse(boards, [False, False, True])
        sym3 = tf.reverse(sym2, [False, True, False])
        sym4 = tf.transpose(boards, perm=[0, 2, 1])
        sym5 = tf.transpose(sym1, perm=[0, 2, 1])
        sym6 = tf.transpose(sym2, perm=[0, 2, 1])
        sym7 = tf.transpose(sym3, perm=[0, 2, 1])
        return tf.transpose([boards, sym1, sym2, sym3, sym4, sym5, sym6, sym7], [1,2,3,0])
    
    @staticmethod
    def get_diagonal(boards, batch_size):
        diag_batch = None
        for idx in range(batch_size):
            diag1 = tf.diag_part(boards[idx])
            diag2 = tf.diag_part(tf.reverse(boards[idx], [False, True]))
            diag3 = tf.reverse(diag1, [True])
            diag4 = tf.reverse(diag2, [True])
            if diag_batch is not None:
                diag_batch = tf.concat(0, [diag_batch,tf.transpose([[[diag1, diag2, diag3, diag4]]], [0,3,1,2])])
            else:
                diag_batch = tf.transpose([[[diag1, diag2, diag3, diag4]]], [0,3,1,2])
        return diag_batch

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
        print(grad)
        if not self.accum_grads:
            return grad

    def train(self, outcome, session):
        gamma = self.discount_factor
        # add one to both scores for smoothing (avoid divide by zeros)
        squashed_outcome = session.run(tf.nn.tanh((outcome['net'] + 1)/(outcome['opponent'] + 1)))
        num_moves = self.boards_history.__len__()
        all_gammas = 0
        for m in range(num_moves):
            all_gammas += gamma**m
        discounted_score = squashed_outcome/all_gammas
        target_series = []
        target_t = 0
        temp = discounted_score
        # implement TD(0) for starters, lambda would be a nice step further
        for m in range(num_moves):
            print('Training on move: ', m, ' ...')
            target_t += temp
            temp *= gamma
            #target_series.append(temp1)
            batch_size = 1
            keep_prob = 0.7
            board_sym_tensor = self.turn_boards(self.boards_history[m])
            half_sym = session.run(tf.slice(board_sym_tensor, [0, 0, 0, 0], [-1, -1, -1, 4]))
            chopped = session.run(tf.slice(board_sym_tensor, [0, 0, 0, 0], [-1, 5, 5, -1]) )
            diag = session.run(self.get_diagonal(self.boards_history[m], batch_size))
            session.run(self.apply_lambda_gradients, feed_dict={self.boards_half_sym:half_sym, self.boards_chopped:chopped, 
                self.boards_diag:diag, self.keep_prob:keep_prob, self.batch_size:batch_size, self.target_val:target_t})
        


