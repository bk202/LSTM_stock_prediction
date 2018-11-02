import tensorflow as tf
from base_model import BaseModel

class LSTMmodel(BaseModel):
    def __init__(self, data_loader, config):
        super(LSTMmodel, self).__init__(config=config)

        self.data_loader = data_loader
        self.train_inputs, self.train_labels = [], []
        self.is_training = None
        self.loss, self.optimizer, self.train_step = None, None, None
        self.learning_rate = tf.placeholder(shape=None,dtype=tf.float32)
        self.min_learning_rate = tf.placeholder(shape=None,dtype=tf.float32)

        # Dimensionality of the data. Since we're only looking back at historical price(y-axis),
        # hence the dimensionality of our data should be 1-D
        self.dimensionality = self.config.dimensionality
        self.time_steps = self.config.time_steps # Number of steps to look back in the past
        self.num_nodes = self.config.num_nodes
        self.layers = len(self.num_nodes)
        self.dropout = self.config.dropout
        self.decay_learning_rate = self.config.decay_learning_rate

        self.build_model()
        self.init_saver()

        return

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)
        return

    def build_model(self):
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        with tf.variable_scope('inputs', reuse=tf.AUTO_REUSE):
            self.is_training = tf.placeholder(dtype=tf.bool, name='training_flag')

            for i in range(self.time_steps):
                input = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, self.dimensionality],
                                       name='train_input_%d'%i)
                label = tf.placeholder(dtype=tf.float32, shape=[self.config.batch_size, 1],
                                        name='train_output_%d'%i)
                self.train_inputs.append(input)
                self.train_labels.append(label)

                tf.add_to_collection('inputs', input)
                tf.add_to_collection('inputs', label)

            tf.add_to_collection('inputs', self.is_training)

        with tf.variable_scope('network', reuse=tf.AUTO_REUSE):
            self.lstm_cells = []
            self.dropout_lstm_cells = []
            for layer in range(self.layers):
                 cell = tf.contrib.rnn.LSTMCell(
                     num_units=self.num_nodes[layer],
                     state_is_tuple=True,
                     initializer=tf.contrib.layers.xavier_initializer(),
                     name='lstm_cell_%d'%layer,
                 )
                 self.lstm_cells.append(cell)

                 dropout_cell = tf.contrib.rnn.DropoutWrapper(
                     cell=cell,
                     input_keep_prob=1.0,
                     output_keep_prob=1.0 - self.dropout,
                     state_keep_prob=1.0 - self.dropout
                 )
                 self.dropout_lstm_cells.append(dropout_cell)

            # Creates a network by wrapping up the previously defined LSTM cells
            self.multi_cell = tf.contrib.rnn.MultiRNNCell(self.lstm_cells)
            self.dropout_multi_cell = tf.contrib.rnn.MultiRNNCell(self.dropout_lstm_cells)

            # Create cell state and hidden state variables to maintain the state of the LSTM
            self.c_state, self.h_state = [], []
            self.initial_state = []
            for layer in range(self.layers):
                c = tf.get_variable(name='c_%d'%layer, shape=[self.config.batch_size, self.num_nodes[layer]],
                                    trainable=False, initializer=tf.zeros_initializer())
                h = tf.get_variable(name='h_%d'%layer, shape=[self.config.batch_size, self.num_nodes[layer]],
                                    trainable=False, initializer=tf.zeros_initializer())
                self.c_state.append(c)
                self.h_state.append(h)
                self.initial_state.append(tf.contrib.rnn.LSTMStateTuple(c, h))
            self.w = tf.get_variable(dtype=tf.float32, shape=[self.num_nodes[-1], 1],
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     name='w')
            self.b = tf.get_variable(dtype=tf.float32, initializer=tf.random_uniform([1], -0.1, 0.1),
                                     name='b')

            # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
            # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
            self.all_inputs = tf.concat([tf.expand_dims(input, 0) for input in self.train_inputs], axis=0)
            print("All inputs shape: ", self.all_inputs.shape)

            # all_outputs is [time_steps, batch_size, num_nodes]
            self.all_lstm_outputs, self.states = tf.nn.dynamic_rnn(
                self.dropout_multi_cell, self.all_inputs, initial_state=tuple(self.initial_state),
                time_major=True, dtype=tf.float32
            )
            self.all_lstm_outputs = tf.reshape(tensor=self.all_lstm_outputs,
                                               shape=[self.config.batch_size * self.time_steps, self.num_nodes[-1]])

            # Shape = [time_steps, batch_size, 1], where 1 = prediction
            self.all_lstm_outputs = tf.matmul(self.all_lstm_outputs, self.w) + self.b

            # Split the outputs into 50 time steps
            self.split_outputs = tf.split(self.all_lstm_outputs, self.time_steps, axis=0)

        with tf.variable_scope('loss_acc', reuse=tf.AUTO_REUSE):
            self.loss = 0.0

            with tf.control_dependencies(
                [tf.assign(self.c_state[layer], self.states[layer][0]) for layer in range(self.layers)] +
                [tf.assign(self.h_state[layer], self.states[layer][1]) for layer in range(self.layers)]
            ):
                for step in range(self.time_steps):
                    self.loss += tf.reduce_mean(0.5 * (self.split_outputs[step] - self.train_labels[step])**2)

        with tf.variable_scope('train_step', reuse=tf.AUTO_REUSE):
            self.learning_rate = tf.maximum(
                tf.train.exponential_decay(self.learning_rate, self.global_step_tensor, decay_steps=1,
                                           decay_rate=self.decay_learning_rate, staircase=True),
                self.min_learning_rate)

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss)

            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # self.gradients, self.v = zip(*self.optimizer.compute_gradients(self.loss))
            # self.gradients = tf.clip_by_global_norm(self.gradients, 5.0)
            # self.train_step = self.optimizer.apply_gradients(zip(self.gradients, self.v))

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)

if __name__ == '__main__':
    class config:
        batch_size = 5
        dimensionality = 1
        time_steps = 50
        num_nodes = [200, 200, 150]
        learning_rate = 0.0001
        decay_learning_rate = 0.5
        dropout = 0.2

    model = LSTMmodel(data_loader=None, config=config)





























