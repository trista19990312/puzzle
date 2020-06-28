import pickle
import gzip

import numpy as np
import winsound

import tensorflow as tf


# Main class used to construct and train networks #
class Network(object):
    def __init__(self, layers=None):
        self.layers = layers

        # 0:
        self.sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, [16], name='x')
        self.y = tf.placeholder(tf.float32, [1])

        # 构建MLP
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x)
        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(prev_layer.output)

        self.y_out = self.layers[-1].output
        self.h_star = self.layers[-1].hstar

        error = self.y - self.y_out
        self.loss = 0.5*tf.reduce_mean(tf.square(error))

        self.learning_rate = 0.75

        # tensorflow training define
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_gds = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # 99:
        tf.global_variables_initializer().run()

    def train(self, data_set_name):
        print('Start training 10', self.learning_rate)

        # 加载数据集
        set_data = load_data_set(data_set_name)

        # compute number of training and testing
        num_total = len(set_data)
        num_testing = num_total // 10
        num_training = num_total - num_testing

        training_data = set_data[0:num_training]
        testing_data = set_data[num_training:]

        saver = tf.train.Saver(max_to_keep=1)

        # 训练MLP
        for epoch in range(500):
            for index in range(num_training):
                iteration = num_training*epoch + index

                xd, yd = training_data[index]
                feed_dict = {self.x: xd,
                             self.y: yd}
                self.train_gds.run(feed_dict)

                # 一个回合结束, 评估模型
                if (iteration+1) % num_training == 0:
                    train_cost = self.evaluate(training_data)       # 训练Loss
                    test_cost = self.evaluate(testing_data)         # 测试Loss
                    print("Epoch {0}: train cost {1:e} test cost {2:e}".format(epoch, train_cost, test_cost))

                    # 保存MLP模型
                    saver.save(self.sess, './model/Hnn.ckpt')

        winsound.Beep(600, 3000)

    def evaluate(self, data_set):
        cost_list = []
        for j in range(len(data_set)):
            xd, yd = data_set[j]
            feed_dict = {self.x: xd,
                         self.y: yd}

            cost = self.sess.run([self.loss], feed_dict)
            cost_list += [cost]

        return np.mean(cost_list)

    def load_model(self):
        # 加载MLP模型
        saver = tf.train.Saver()
        saver.restore(self.sess, './model/Hnn.ckpt')

    def hnn(self, state):
        state_norm = (state-7.5)/7.5    # normalize
        xd = np.array(state_norm, dtype='float32')
        feed_dict = {self.x: xd}
        h_star = self.sess.run([self.h_star], feed_dict)
        return h_star[0][0]*40.0        # 恢复


class FullyConnectedLayer(object):
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

    def set_inpt(self, inpt):
        self.inpt = tf.reshape(inpt, (-1, self.n_in))

        w_init = tf.random_normal(shape=[self.n_in, self.n_out], stddev=np.sqrt(1.0/self.n_in))
        b_init = tf.zeros(shape=[self.n_out, ])

        self.w = tf.Variable(w_init, dtype=tf.float32, name='w', trainable=True)
        self.b = tf.Variable(b_init, dtype=tf.float32, name='b', trainable=True)

        self.output = tf.nn.sigmoid(tf.matmul(self.inpt, self.w) + self.b, name='output')


class LinearLayer(object):
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

    def set_inpt(self, inpt):
        self.inpt = tf.reshape(inpt, (-1, self.n_in))

        w_init = tf.random_normal(shape=[self.n_in, self.n_out], stddev=np.sqrt(1.0/self.n_in))
        b_init = tf.zeros(shape=[self.n_out, ])

        self.w = tf.Variable(w_init, dtype=tf.float32, name='w', trainable=True)
        self.b = tf.Variable(b_init, dtype=tf.float32, name='b', trainable=True)

        self.output = tf.matmul(self.inpt, self.w) + self.b
        self.hstar = tf.nn.relu(self.output)


# Load the data set
def load_data_set(filename):
    f = gzip.open(filename, 'rb')
    file_data = pickle.load(f, encoding="latin1")
    f.close()

    # normalize
    for idx in range(len(file_data)):
        x, y = file_data[idx]
        x = (x-7.5)/7.5
        y = y/40.0
        file_data[idx] = x, y

    random_method = np.random.RandomState(123456)
    random_method.shuffle(file_data)

    return file_data
