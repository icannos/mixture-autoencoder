"""
Maxime Darrin
Implementation of the Mixture Autoencoder from https://arxiv.org/abs/1712.07788 by D.Zhang
"""

import tensorflow as tf
import numpy as np


class mixture_autoencoder():
    sample_entropy = None

    def __init__(self, training_steps=3000, batch_size=256, autoencoders_topology=(128, 64, 8),
                 hyper_param_bentropy=1,
                 hyper_param_sentropy=1,
                 classifier_topology=(8, 8), input_dim=1024,
                 num_clusters=8,
                 autoencoders_activation=(tf.nn.tanh, tf.nn.tanh, tf.nn.tanh),
                 hp_encoded_reg=0.005):
        """
        Basic initializer, all parameters can be changed by updating
        :param autoencoders_topology: defines the topology of the autoencoders (size of each hidden layer)
        :param classifier_topology: defines the topology of the classifier (size of each hidden layer)
        :param input_dim: size of one input vector
        :param num_clusters: number of expected cluster
        :param autoencoders_activation: function (graph) to use as activation function
        """

        # Parameters
        self.training_steps = training_steps
        self.batch_size = batch_size

        self.autoencoders_topology = autoencoders_topology
        self.classifier_topology = classifier_topology
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.autoencoder_activation = autoencoders_activation
        self.encoded_reg = hp_encoded_reg

        self.hyper_param_bentropy = hyper_param_bentropy
        self.hyper_param_sentropy = hyper_param_sentropy

        # Graph variables
        self.encoders = []
        self.decoders = []
        self.autoencoders_weights = []
        self.Y_preds = []
        self.classifier_weights = {}

        self.losses = []

        self.sess = None

        self.pre_train_ops = []

    def compile(self):
        """
        Build the graph from the parameters of the class
        :return: None, defines the graph variable that we could want to compute
        """
        self.sample_entropy = tf.placeholder("float")
        self.batch_entropy = tf.placeholder("float")
        self.learning_rate = tf.placeholder("float")
        self.X = tf.placeholder("float", [None, self.input_dim])
        self.Y_true = self.X

        # Weights initialization & autoencoder building
        for k in range(self.num_clusters):
            self.autoencoders_weights.append(
                {"encoder-input": tf.Variable(tf.random_normal([self.input_dim, self.autoencoders_topology[0]]))})

            for i in range(len(self.autoencoders_topology) - 1):
                self.autoencoders_weights[k]["encoder-w" + str(i)] = tf.Variable(
                    tf.random_normal([self.autoencoders_topology[i], self.autoencoders_topology[i + 1]]))

            for i in range(len(self.autoencoders_topology) - 1, 0, -1):
                self.autoencoders_weights[k]["decoder-w" + str(i)] = tf.Variable(
                    tf.random_normal([self.autoencoders_topology[i], self.autoencoders_topology[i - 1]]))

            self.autoencoders_weights[k]["decoder-output"] = tf.Variable(
                tf.random_normal([self.autoencoders_topology[0], self.input_dim]))

            self.encoders.append(self.__build_encoder(self.X, k))
            self.decoders.append(self.__build_decoder(self.encoders[k], k))

            self.Y_preds.append(self.decoders[k])

        # Encoded vectors
        self.z = tf.concat(self.encoders, 1)

        # Classifier weigths
        self.classifier_weights["classifier-input"] = tf.Variable(
            tf.random_normal([self.num_clusters * self.autoencoders_topology[-1],
                              self.classifier_topology[0]]))

        for i in range(len(self.classifier_topology) - 1):
            self.classifier_weights["classifier-w" + str(i)] = tf.Variable(
                tf.random_normal([self.classifier_topology[i], self.classifier_topology[i + 1]]))

        self.classifier_weights["classifier-output"] = tf.Variable(
            tf.random_normal([self.classifier_topology[-1], self.num_clusters]))

        self.classifier = self.__build_classifier()

        # Losses of each autoencoder
        for k in range(self.num_clusters):
            self.losses.append(self.__build_cluster_loss(k))

        self.loss = self.__build_loss()

        self.train_network = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        for k in range(self.num_clusters):
            self.pre_train_ops.append(tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses[k]))

        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def init_session(self):
        """
        Initializes a new compute session and randoms weights
        :return:
        """
        self.sess = tf.Session()
        self.sess.run(self.init)

    def pretrain(self, X, k):
        shuffle = np.arange(X.shape[0])
        np.random.shuffle(shuffle)

        for j in range(X.shape[0] // self.batch_size - 2):
            _, loss = self.sess.run(
                [self.pre_train_ops[k], self.losses[k]], feed_dict=
                {self.X: X[shuffle][j * self.batch_size:(j + 1) * self.batch_size],
                })

        return loss


    def train(self, X, entropy_strategy="balanced", sample_entropy=0, batch_entropy=0, learning_rate=0.001):
        """
        Performs one training step
        :param X: input vector
        :param sample_entropy: weight of the sample_entropy term
        :param batch_entropy: weight of the batch_entropy term
        :return: TBD
        """
        shuffle = np.arange(X.shape[0])
        np.random.shuffle(shuffle)

        for j in range(X.shape[0] // self.batch_size - 2):
            if entropy_strategy == "balanced":

                sample_entropy, batch_entropy = \
                    self.sess.run([self.strat_sample_entropy, self.strat_batch_entropy],
                                  feed_dict=
                                  {self.X: X[shuffle][
                                           j * self.batch_size:(j + 1) * self.batch_size],
                                   self.learning_rate: learning_rate
                                   })

            print("Sample entropy:", sample_entropy,"Batch Entropy", batch_entropy)

            _, loss, batch_wise, p_mean = self.sess.run(
                [self.train_network, self.loss, self.batch_wise_entropy, self.p_mean], feed_dict=
                {self.X: X[shuffle][j * self.batch_size:(j + 1) * self.batch_size],
                 self.sample_entropy: self.hyper_param_sentropy*sample_entropy,
                 self.batch_entropy: self.hyper_param_bentropy*batch_entropy,
                 self.learning_rate:learning_rate
                 })

        loss, batch_wise_entropy, p_mean = self.sess.run(
            [self.loss, self.batch_wise_entropy, self.p_mean], feed_dict=
            {self.X: X[shuffle][j * self.batch_size:(j + 1) * self.batch_size],
             self.sample_entropy: sample_entropy,
             self.batch_entropy: batch_entropy
             })
        return loss, batch_wise_entropy, p_mean

    def predict(self, X):
        """
        Return clustering prediction for that input
        :param X: input
        :return: cluster prediction
        """
        return self.sess.run([self.classifier], feed_dict={self.X: X})[0]

    # Private methods

    def __build_cluster_loss(self, k):
        return (tf.reduce_sum(tf.pow(self.Y_true - self.Y_preds[k], 2, name="cluster_loss" + str(k)),1) +
                self.encoded_reg) * tf.reduce_sum(tf.pow(self.encoders[k], 2))  * self.classifier[:, k]

    def __build_loss(self):
        losses = tf.stack(self.losses)

        self.element_wise_entropy = - tf.reduce_sum(self.classifier * tf.log(self.classifier), 1)

        self.p_mean = tf.reduce_mean(self.classifier, 0)

        self.batch_wise_entropy = - tf.reduce_sum(self.p_mean * tf.log(self.p_mean), 0)

        self.reconstr_loss = tf.reduce_mean(tf.reduce_sum(losses, 0))

        self.strat_sample_entropy, self.strat_batch_entropy = self.__build_strats_entropies()

        self.sample_loss = tf.reduce_mean(tf.reduce_sum(self.losses, 0) +
                                          self.sample_entropy * self.element_wise_entropy)

        loss = tf.reduce_mean(tf.reduce_sum(self.losses, 0) +
                              self.sample_entropy * self.element_wise_entropy) - self.batch_entropy * self.batch_wise_entropy

        return loss

    def __build_classifier(self):
        layer = tf.nn.sigmoid((tf.matmul(self.z, self.classifier_weights["classifier-input"])))

        for i in range(len(self.classifier_topology) - 1):
            layer = tf.nn.sigmoid((tf.matmul(layer, self.classifier_weights["classifier-w" + str(i)])))

        layer = tf.nn.softmax((tf.matmul(layer, self.classifier_weights["classifier-output"])))

        return layer

    def __build_encoder(self, X, k):
        '''

        :param X:
        :return:
        '''
        layer = self.autoencoder_activation[0]((tf.matmul(X, self.autoencoders_weights[k]["encoder-input"])))

        for i in range(len(self.autoencoders_topology) - 1):
            layer = self.autoencoder_activation[i](
                (tf.matmul(layer, self.autoencoders_weights[k]["encoder-w" + str(i)])))

        return layer

    def __build_decoder(self, X, k):
        '''

        :param X:
        :return:
        '''
        layer = X

        for i in range(len(self.autoencoders_topology) - 1, 0, -1):
            layer = self.autoencoder_activation[i](
                (tf.matmul(layer, self.autoencoders_weights[k]["decoder-w" + str(i)])))

        layer = self.autoencoder_activation[0]((tf.matmul(layer, self.autoencoders_weights[k]["decoder-output"])))

        return layer

    def __build_strats_entropies(self, strat="balanced"):
        if strat == "balanced":
            return self.__build_entropy_strategy_balanced()

    def __build_entropy_strategy_balanced(self):
        """
        Strategy for choosing alpha & beta (sample entropy and batch entropy), we try to maintain all the terms of
        the error at the same magnitude
        :return:
        """
        cluster_mean_loss = tf.reduce_mean(self.losses)

        sample_mean_entropy = tf.reduce_mean(- tf.reduce_sum(self.classifier * tf.log(self.classifier), 1))

        sample_entropy = cluster_mean_loss / sample_mean_entropy

        preloss = tf.reduce_mean(tf.reduce_sum(self.losses, 0) + sample_entropy * self.element_wise_entropy)
        batch_mean_entropy = tf.reduce_mean(- tf.reduce_sum(self.p_mean * tf.log(self.p_mean), 0))

        batch_entropy =  (preloss/ batch_mean_entropy)

        return (sample_entropy, batch_entropy)
