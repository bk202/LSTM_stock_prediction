import tensorflow as tf
from base_trainer import BaseTrain
from model import LSTMmodel
from data_loader import DataLoader
from utility import AverageMeter
from tqdm import tqdm

class LSTMTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger=None, data_loader=None):
        """
            Constructing the Cifar trainer based on the Base Train..
            Here is the pipeline of constructing
            - Assign sess, model, config, logger, data_loader(if_specified)
            - Initialize all variables
            - Load the latest checkpoint
            - Create the summarizer
            - Get the nodes we will need to run it from the graph
            :param sess:
            :param model:
            :param config:
            :param logger:
            :param data_loader:
        """

        super(LSTMTrainer, self).__init__(sess, model, config, logger, data_loader)
        #self.model.load(sess)

        self.summarizer = logger
        self.inputs_collection = tf.get_collection('inputs')
        self.inputs, self.labels, self.is_training = [], [], None

        # Input and labels are being mixed in collections in the order:
        # input, label, input, label,..., input, label, is_training
        for step in range(0, self.model.time_steps):
            self.inputs.append(self.inputs_collection[step * 2])
            self.labels.append(self.inputs_collection[step * 2 + 1])

        self.is_training = self.inputs_collection[(step + 1) * 2]

        print('self.inputs: ', len(self.inputs))
        print('self.labels: ', len(self.labels))
        print('self.is_training: ', self.is_training)

        self.train_op, self.loss_node = tf.get_collection('train')

        return

    def train(self):
        """
        This is the main loop of training
        Looping on the epochs
        :return:
        """
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(cur_epoch)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.test(cur_epoch)

    def train_epoch(self, epoch=None):

        tt = tqdm(range(self.data_loader.train_iterations), total=self.data_loader.train_iterations,
                  desc="epoch-{}-".format(epoch))

        loss_per_epoch = AverageMeter()

        for cur_it in tt:
            # One Train step on the current batch
            loss = self.train_step()
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)

        self.sess.run(self.model.global_epoch_inc)

        self.model.save(self.sess)

        print("""
        Epoch-{}  loss:{:.4f}
                """.format(epoch, loss_per_epoch.val))

        tt.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss & acc of that minibatch.
        :return: (loss, acc) tuple of some metrics to be used in summaries
        """

        feed_dict = {}
        for step in range(self.model.time_steps):
            input, label = self.data_loader.next_batch()

            feed_dict[self.model.train_inputs[step]] = input.reshape(-1, 1)
            feed_dict[self.model.train_labels[step]] = label.reshape(-1, 1)

        feed_dict.update({self.is_training: False, self.model.learning_rate: self.config.learning_rate,
                          self.model.min_learning_rate: self.config.min_learning_rate})
        _, loss = self.sess.run([self.train_op, self.loss_node], feed_dict=feed_dict)

        return loss

    def test(self, epoch):
        # initialize dataset
        self.data_loader.initialize(self.sess, is_training=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.test_iterations), total=self.data_loader.test_iterations,
                  desc="Val-{}-".format(epoch))

        loss_per_epoch = AverageMeter()

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            feed_dict = {}
            for step in range(self.model.time_steps):
                input, label = self.data_loader.next_batch()

                feed_dict[self.model.train_inputs[step]] = input
                feed_dict[self.model.train_labels[step]] = label

            feed_dict.update({self.is_training: False})

            loss = self.sess.run([self.loss_node], feed_dict=feed_dict)
            # update metrics returned from train_step func
            loss_per_epoch.update(loss)

        # summarize
        # summaries_dict = {'test/loss_per_epoch': loss_per_epoch.val,
        #                   'test/acc_per_epoch': acc_per_epoch.val}
        # self.summarizer.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)

        print("""
            Val-{}  loss:{:.4f}
                    """.format(epoch, loss_per_epoch.val))

        tt.close()


if __name__ == '__main__':
    ticker = "AAL"
    file_name = 'stock_market_data-%s.csv' % ticker

    class config:
        pickle_file = file_name
        batch_size = 5
        dimensionality = 1
        time_steps = 50
        num_nodes = [200, 200, 150]
        learning_rate = 0.0001
        min_learning_rate = 0.000001
        decay_learning_rate = 0.5
        dropout = 0.2
        num_epochs = 1

    data_loader = DataLoader(config)
    model = LSTMmodel(data_loader=None, config=config)
    sess = tf.Session()

    trainer = LSTMTrainer(sess=sess, model=model, config=config, logger=None, data_loader=data_loader)
    trainer.train()



