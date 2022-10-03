# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import he_uniform, he_normal
from keras.layers import Input, Dense, LeakyReLU, Lambda


class nn_siamese:

    ###########################################################################################
    #                                        Constructor                                      #
    ###########################################################################################

    def __init__(
            self,
            layer_dims,  # [n_x, n_h1, .., n_hL] - at least one hidden layer
            learning_rate,
            num_epochs,
            weight_init='he_normal',
            output_activation='sigmoid',
            loss_function='binary_crossentropy',
            minibatch_size=64,
            l2=0.0,
            distance_metric='abs',
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
        self.distance_metric = distance_metric

        if weight_init == 'he_uniform':
            self.weight_init = he_uniform(seed=self.seed)
        elif weight_init == 'he_normal':
            self.weight_init = he_normal(seed=self.seed)

        # model
        self.model = self.create_siamese_model()
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

    #################
    # Siamese model #
    #################

    def create_siamese_model(self):
        siamese_base = self.create_siamese_base()

        input_left = Input(shape=(self.layer_dims[0],))
        input_right = Input(shape=(self.layer_dims[0],))

        encoded_left = siamese_base(input_left)
        encoded_right = siamese_base(input_right)

        # Add a customized layer to compute the distance between the encodings
        if self.distance_metric == 'abs':
            lambda_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]), name='abs')
        elif self.distance_metric == 'chi2':
            lambda_layer = Lambda(lambda tensors: (tensors[0] - tensors[1])**2 / (tensors[0] + tensors[1]), name='chi2')

        distance = lambda_layer([encoded_left, encoded_right])

        # Add a dense layer with a sigmoid unit to generate the similarity score
        X_output = Dense(
            units=1,
            activation='sigmoid',
            use_bias=True,
            kernel_initializer=self.weight_init,
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None
        )(distance)

        # Connect the inputs with the outputs
        return Model(inputs=[input_left, input_right], outputs=X_output)

    ################
    # Siamese base #
    ################

    def create_siamese_base(self):
        # Input dims
        n_x = self.layer_dims[0]

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
        for l in self.layer_dims[2:]:
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

        return Model(inputs=X_input, outputs=X)

    ########
    # Plot #
    ########

    def plot_learning_curves(self, history, flag_val):
        # Plot accuracy values
        plt.plot(history.history['accuracy'])
        if flag_val:
            plt.plot(history.history['val_accuracy'])
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

        return y_hat

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
