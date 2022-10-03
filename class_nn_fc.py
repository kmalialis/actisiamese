# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers import Input, Dense, LeakyReLU
from keras.initializers import he_uniform, he_normal


class nn_fc:

    ###########################################################################################
    #                                        Constructor                                      #
    ###########################################################################################

    def __init__(
            self,
            layer_dims,  # [n_x, n_h1, .., n_hL, n_y], at least one hidden layer
            learning_rate,
            num_epochs,
            weight_init='he_normal',
            output_activation='softmax',
            loss_function='categorical_crossentropy',
            minibatch_size=64,
            l2=0.0,
            seed=0
    ):

        # seed
        self.seed = seed

        # NN parameters
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.output_activation = output_activation
        self.loss_function = loss_function
        self.minibatch_size = minibatch_size
        self.l2 = l2

        if weight_init == 'he_uniform':
            self.weight_init = he_uniform(seed=self.seed)
        elif weight_init == 'he_normal':
            self.weight_init = he_normal(seed=self.seed)

        # model
        self.model = self.create_fc_model()
        # self.model.summary()

        # configure model for training
        self.model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=self.loss_function,
            metrics=['accuracy']
        )

    ###########################################################################################
    #                                      Auxiliary                                          #
    ###########################################################################################

    ############
    # FC Model #
    ############

    def create_fc_model(self):
        # Input and output dims
        n_x = self.layer_dims[0]
        n_y = self.layer_dims[-1]

        # Input layer
        X_input = Input(shape=(n_x,), name='input')

        #  First hidden layer
        X = Dense(
            units=self.layer_dims[1],
            activation=None,
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=l2(self.l2),
            bias_regularizer=None,
            activity_regularizer=None
        )(X_input)
        X = LeakyReLU(alpha=0.01)(X)

        #  Other hidden layers (if any)
        for l in self.layer_dims[2:-1]:
            X = Dense(
                units=l,
                activation=None,
                use_bias=True,
                kernel_initializer=self.weight_init,
                bias_initializer='zeros',
                kernel_regularizer=l2(self.l2),
                bias_regularizer=None,
                activity_regularizer=None
            )(X)
            X = LeakyReLU(alpha=0.01)(X)

        # Output layer
        y_out = Dense(
            units=n_y,
            activation=self.output_activation,
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=l2(self.l2),
            bias_regularizer=None,
            activity_regularizer=None,
            name='output'
        )(X)

        # Model
        return Model(inputs=X_input, outputs=y_out)

    ########
    # Plot #
    ########

    def plot_learning_curves(self, history, flag_val):
        # Plot accuracy values
        plt.plot(history.history['accuracy'])
        if flag_val:
            plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

        # Plot loss values
        plt.plot(history.history['loss'])
        if flag_val:
            plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.show()

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ##############
    # Prediction #
    ##############

    def predict(self, x):
        # probability prediction
        y_hat = self.model.predict(x=x, verbose=0)

        # class prediction
        y_hat_max = np.max(y_hat)
        y_hat_argmax = np.argmax(y_hat)

        return y_hat, y_hat_max, y_hat_argmax

    ############
    # Training #
    ############

    def train(self, x, y, validation_data=None):
        history = self.model.fit(
            x=x,
            y=y,
            validation_data=validation_data,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            shuffle=False,
            verbose=0  # 0: off, 1: full, 2: brief
        )

        flag_val = False if validation_data is None else True

        acc = history.history['accuracy'][-1]
        loss = history.history['loss'][-1]

        if flag_val:
            val_acc = history.history['val_accuracy'][-1]
            val_loss = history.history['val_loss'][-1]
            return loss, acc, val_loss, val_acc
        else:
            return loss, acc
