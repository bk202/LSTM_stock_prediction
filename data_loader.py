import pickle
from pandas_datareader import data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

class DataLoader:
    def __init__(self, config):
        self.config = config

        self.data_pkl = pd.read_csv(config.pickle_file, delimiter=',', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
        self.data_pkl = self.data_pkl.sort_values('Date')
        high_prices = self.data_pkl.loc[:, 'High'].as_matrix()
        low_prices = self.data_pkl.loc[:, 'Low'].as_matrix()
        mid_prices = (high_prices + low_prices) / 2.0

        separate_point = int(0.9 * self.data_pkl.shape[0])
        print("Separate point: {}".format(separate_point))

        self.train_data = mid_prices[:separate_point]
        self.test_data = mid_prices[separate_point:]

        print('Training set', self.train_data.shape)
        print('Test set', self.test_data.shape)

        self.train_len = self.train_data.shape[0]
        self.test_len = self.test_data.shape[0]

        self.train_iterations = (self.train_len + self.config.batch_size - 1) // self.config.batch_size
        self.test_iterations = (self.test_len + self.config.batch_size - 1) // self.config.batch_size

        print("Data loaded successfully..")

        self.scaler = MinMaxScaler()
        self.preprocess_data()

        self.segments = (self.train_data.size // self.config.batch_size) - 1

        # Offsets keeps track of the index of where to collect batch data from.
        # Ex. |dataset| = 200, batch_size = 200, offsets = [0, 40, 80, 120, 160]
        # Then we have a initial batch: [train_data[0], train_data[40], train_data[80], train_data[120], train_data[160]
        # Simply increment the index by 1 and module by batch_size each time we generate a new batch.
        self.offsets = [i * self.segments for i in range(self.config.batch_size)]

        # Starting boundary of our offset
        self.fixed_offsets = [i * self.segments for i in range(self.config.batch_size)]

        return

    def next_batch(self):
        batch_data = np.zeros((self.config.batch_size), dtype=np.float32)
        batch_label = np.zeros((self.config.batch_size), dtype=np.float32)

        #print('\n\tOffsets: ', self.offsets)
        for idx in range(self.config.batch_size):
            offset = self.offsets[idx]
            label_idx = np.random.randint(offset, offset + 5)

            batch_data[idx] = self.train_data[offset]
            batch_label[idx] = self.train_data[label_idx]

            self.offsets[idx] = ((self.offsets[idx] + 1) % self.segments) + self.fixed_offsets[idx]

        return batch_data, batch_label

    def get_unrolls(self):
        data, label = [], []

        for i in range(self.config.num_unrollings):
            d, l = self.next_batch()
            data.append(d)
            label.append(l)

        return data, label

    def preprocess_data(self):
        self.train_data = self.train_data.reshape(-1, 1)
        self.test_data = self.test_data.reshape(-1, 1)

        smoothing_window_size = int(self.train_data.size // 4.4)
        for di in range(0, self.train_data.size, smoothing_window_size):
            self.scaler.fit(self.train_data[di:di + smoothing_window_size, :])
            self.train_data[di:di + smoothing_window_size, :] = self.scaler.transform(
                self.train_data[di:di + smoothing_window_size, :])

        # You normalize the last bit of remaining data

        self.scaler.fit(self.train_data[di:, :])
        self.train_data[di:, :] = self.scaler.transform(self.train_data[di:, :])

        smoothing_window_size = int(self.test_data.size // 4.4)
        for di in range(0, self.test_data.size, smoothing_window_size):
            self.scaler.fit(self.test_data[di: di + smoothing_window_size, :])
            self.test_data[di:di + smoothing_window_size, :] = self.scaler.transform(
                self.test_data[di:di + smoothing_window_size, :])

        self.scaler.fit(self.test_data[di:, :])
        self.test_data[di:, :] = self.scaler.transform(self.test_data[di:, :])

        EMA = 0.0
        gamma = 0.1
        for ti in range(self.train_data.size):
            EMA = gamma * self.train_data[ti] + (1 - gamma) * EMA
            self.train_data[ti] = EMA

        self.all_mid_data = np.concatenate([self.test_data, self.test_data], axis=0)

    def plot(self):
        plt.figure(figsize=(18, 9))
        plt.plot(range(self.train_data.shape[0]), self.train_data)
        plt.xticks(range(0, self.data_pkl.shape[0], 500), self.data_pkl['Date'].loc[::500], rotation=45)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Mid Price', fontsize=18)
        plt.show()
        return


def main():
    ticker = "AAL"
    file_name = 'stock_market_data-%s.csv' % ticker

    class config:
        pickle_file = file_name
        batch_size = 5
        num_unrollings = 4

    tf.reset_default_graph()
    sess = tf.Session()

    data_loader = DataLoader(config)

    data, labels = data_loader.get_unrolls()
    # for idx in range(len(data)):
    #     print('\n\nUnrolled index {}'.format(idx))
    #
    #     print('\tInputs: ', data[idx])
    #     print('\n\tOutputs: ', labels[idx])

if __name__ == '__main__':
    main()








