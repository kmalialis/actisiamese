# -*- coding: utf-8 -*-

#######################################################################################################################
# PAPER                                                                                                               #
#                                                                                                                     #
# You can get a *free* copy of the pre-print version from arXiv or Zenodo. Alternatively, you can get the published   #
# version from the publisher’s website (behind a paywall). Please check the README file for the links.                #
#                                                                                                                     #
# CITATION REQUEST                                                                                                    #
#                                                                                                                     #
# If you have found our paper and / or part of our code useful, please cite our work as follows:                      #
#                                                                                                                     #
# K. Malialis, C. G. Panayiotou, M. M. Polycarpou, Nonstationary data stream classification with online active        #
# learning and siamese neural networks, Neurocomputing, Volume 512, Pages 235-252, 2022,                              #
# doi: 10.1016/j.neucom.2022.09.065.                                                                                  #
#                                                                                                                     #
# INSTRUCTIONS                                                                                                        #
#                                                                                                                     #
# In main_synthetic.py, you must provide the parameters under "Settings: required". For example, if you run           #
# main_synthetic.py as it is, it will generate ActiSiamese's results in the “exps” folder (to be created),            #
# as in Fig. 7b. You can then use the function provided in “plot.py” to plot the results.                             #
#                                                                                                                     #
# REQUIREMENTS                                                                                                        #
#                                                                                                                     #
# Please check the “requirements.txt” file for the necessary libraries and packages.                                  #
#######################################################################################################################

import os

# ## GPU-related code
#
# hide warnings (before importing Keras)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # choose GPU (before importing Keras)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#
# # dynamically grow memory
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = False  # to log device placement (on which device the operation ran)
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

import numpy as np
from copy import deepcopy
from main_synthetic_inner import run
from class_nn_fc import nn_fc
from class_nn_siamese import nn_siamese

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

#######
# I/O #
#######


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()


# Write array to a row in the given file
def write_to_file(filename, arr):
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

############
# Datasets #
############


# sea10
def add_sea10(d):
    d['num_features'] = 2  # two features, x1 and x2
    d['x_min'] = (0, 0)  # (min x1, min x2)
    d['x_max'] = (15, 15)  # (max x1, max x2)
    d['num_classes'] = 10

    d['sea_bins'] = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 30.0])
    d['sea_bins_drifted'] = np.array([0.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0])

    # for recurrent drift
    d['sea_bins_init'] = deepcopy(d['sea_bins'])


# circles10
def add_circles10(d):
    d['num_features'] = 2  # two features, x1 and x2
    d['x_min'] = (0, 0)  # (min x1, min x2)
    d['x_max'] = (15, 15)  # (max x1, max x2)
    d['num_classes'] = 10

    d['circles'] = {0: (4.0, 4.0, 3.0),
                    1: (4.0, 9.0, 2.0),
                    2: (8.0, 12.0, 1.0),
                    3: (12.0, 8.5, 1.0),
                    4: (12.0, 12.0, 2.0),
                    5: (9.0, 4.5, 2.0),
                    6: (4.0, 12.0, 1.0),
                    7: (8.0, 8.5, 2.0),
                    8: (12.5, 5.5, 1.5),
                    9: (12.0, 3.0, 1.0)}

    d['circles_drifted'] = {
                    0: (4.0, 4.0, 1.0),
                    1: (4.0, 9.0, 1.0),
                    2: (9.0, 13.0, 1.0),
                    3: (13.0, 9.5, 1.0),
                    4: (12.0, 13.0, 2.0),
                    5: (10.0, 3.5, 2.0),
                    6: (4.0, 12.0, 2.0),
                    7: (8.0, 8.5, 3.0),
                    8: (13.5, 4.5, 1.5),
                    9: (13.0, 2.0, 1.0)}

    # for recurrent drift
    d['circles_init'] = deepcopy(d['circles'])


# blobs12
def add_blobs12(d):
    d['num_features'] = 3  # two features, x1 and x2
    d['x_min'] = (0, 0, 0)  # (min x1, min x2)
    d['x_max'] = (15, 15, 15)  # (max x1, max x2)
    d['num_classes'] = 12

    d['blob_std'] = 1.5
    d['blob_std_drifted'] = 1.5

    d['blobs'] = {0: (3.0, 3.0, 3.0),  # pink
                  1: (3.0, 3.0, 8.0),  # brown
                  2: (3.0, 3.0, 13.0),  # orange

                  3: (10.0, 3.0, 3.0),  # navy
                  4: (10.0, 3.0, 8.0),  # olive
                  5: (10.0, 3.0, 13.0),  # green

                  6: (3.0, 13.0, 3.0),  # cyan
                  7: (3.0, 13.0, 8.0),  # purple
                  8: (3.0, 13.0, 13.0),  # grey

                  9: (13.0, 13.0, 3.0),  # red
                  10: (13.0, 13.0, 8.0),  # black
                  11: (13.0, 13.0, 13.0)}  # magenta

    d['blobs_drifted'] = {0: (10.0, 3.0, 3.0),  # navy
                          1: (3.0, 3.0, 8.0),  # brown
                          2: (13.0, 13.0, 13.0),  # magenta

                          3: (3.0, 13.0, 3.0),  # cyan
                          4: (10.0, 3.0, 8.0),  # olive
                          5: (3.0, 3.0, 13.0),  # orange

                          6: (13.0, 13.0, 3.0),  # red
                          7: (3.0, 13.0, 8.0),  # purple
                          8: (10.0, 3.0, 13.0),  # green

                          9: (3.0, 3.0, 3.0),  # pink
                          10: (13.0, 13.0, 8.0),  # black
                          11: (3.0, 13.0, 13.0)}  # grey

    # for recurrent drift
    d['blobs_init'] = deepcopy(d['blobs'])
    d['blob_std_init'] = deepcopy(d['blob_std'])

######
# NN #
######


def create_nn_fc(params_env, params_nn, layer_dims, seed):
    return nn_fc(
        layer_dims=layer_dims + [params_env['num_classes']],
        learning_rate=params_nn['learning_rate'],
        num_epochs=params_nn['num_epochs'],
        minibatch_size=params_nn['minibatch_size'],
        l2=params_nn['l2'],
        seed=seed)


def create_nn_siamese(params_nn, layer_dims, seed):
    return nn_siamese(
            layer_dims=layer_dims,
            learning_rate=params_nn['learning_rate'],
            num_epochs=params_nn['num_epochs'],
            minibatch_size=params_nn['minibatch_size'],
            l2=params_nn['l2'],
            seed=seed)


def create_nn_single(params_env, params_nn):
    nn = None
    if params_env['method'] in ['rvus', 'actiq']:
        nn = create_nn_fc(params_env, params_nn, layer_dims=params_nn['layer_dims'], seed=params_env['seed'])
    elif params_env['method'] == 'actisiamese':
        nn = create_nn_siamese(params_nn, layer_dims=params_nn['layer_dims'], seed=params_env['seed'])

    return nn


def create_nn_ensemble(params_env, params_nn):
    nn = None
    ensemble = []
    for i in range(len(params_nn['lst_layer_dims'])):
        layer_dims = params_nn['lst_layer_dims'][i]

        if params_env['method'] in ['rvus', 'actiq']:
            nn = create_nn_fc(params_env, params_nn, layer_dims=layer_dims, seed=params_env['seed'] + i)
        elif params_env['method'] == 'actisiamese':
            nn = create_nn_siamese(params_nn, layer_dims=layer_dims, seed=params_env['seed'] + i)

        ensemble.append(nn)

    return ensemble

#################
# safety checks #
#################


def run_safety_checks(params_env):
    if params_env['flag_learning'] not in ['supervised', 'active']:
        raise Exception('Incorrect learning paradigm entered.')

    if params_env['method'] not in ['rvus', 'actiq', 'actisiamese']:
        raise Exception('Incorrect learning method entered.')

    if params_env['data_source'] not in ['sea10', 'circles10', 'blobs12']:
        raise Exception('Incorrect dataset entered.')

    if params_env['imbalance'] not in ['balanced', 'single_minority', 'multi_minority']:
        raise Exception('Incorrect imbalance type entered.')

    if params_env['method'] == 'actisiamese' and params_env['memory_size'] < 2:
        raise Exception('Siamese network requires memory size >= 2')

    if params_env['method'] == 'actiq' and params_env['memory_size'] < 1:
        raise Exception('Neural network requires memory size >= 1')

    if params_env['active_budget_total'] < 0.0 or params_env['active_budget_total'] > 1.0:
        raise Exception('Budget must be in [0,1].')

    if params_env['flag_drift']:
        if isinstance(params_env['drift_start_time'], int):
            if params_env['drift_start_time'] >= params_env['time_steps']:
                raise Exception('Concept drift is set to start after the end of the simulation.')
        elif isinstance(params_env['drift_start_time'], str):
            if params_env['drift_start_time'] != 'default':
                raise Exception('Concept drift is set incorrectly.')

    if params_env['flag_ensemble'] and params_env['flag_learning'] == 'supervised':
        raise Exception('Ensembling for online supervised learning not implemented.')

###########################################################################################
#                                         Main                                            #
###########################################################################################


def main():

    ######################
    # Settings: required #
    ######################

    # exp parameters
    params_env = {'repeats': 20,
                  'time_steps': 20000,
                  'data_source': 'sea10',  # 'sea10', 'circles10', 'blobs12'
                  'flag_drift': False,
                  'flag_drift_recurrent': False,
                  'drift_start_time': 0,  # 'default' or <int>
                  'imbalance': 'multi_minority',  # 'balanced', 'single_minority' or 'multi_minority'
                  'minority_ratio': 0.001,  # 0.01 (severe), 0.001 (extreme)

                  'method': 'actisiamese',  # 'rvus', 'actiq', 'actisiamese'
                  'memory_size': 10,  # memory size L per queue, it does not apply if method = rvus
                  'flag_learning': 'active',  # 'supervised' or 'active'
                  'active_budget_total': 0.01,  # in [0.0, 1.0] - NOTE: applies only if flag_learning = active

                  'flag_ensemble': False,
                  'ensemble_size': 10
                  }

    # data source
    if params_env['data_source'] == 'sea10':
        add_sea10(params_env)
    elif params_env['data_source'] == 'circles10':
        add_circles10(params_env)
    if params_env['data_source'] == 'blobs12':
        add_blobs12(params_env)

    # nn parameters
    params_nn = {'learning_rate': 0.01,
                 'num_epochs': 1,
                 'layer_dims': [params_env['num_features'], 32, 32],  # [n_x, n_h1, .., n_hL] ie it does not contain n_y
                 'minibatch_size': 64,
                 'l2': 0.0,

                 'lst_layer_dims': [  # NOTE: applies only if flag_ensemble
                     [params_env['num_features'], 32, 32]
                     ] * params_env['ensemble_size']
                 }

    # safety checks for the inserted settings
    run_safety_checks(params_env)

    ###################
    # Settings: fixed #
    ###################
    # NOTE: Keep these parameters fixed to replicate the paper's results

    # fixed - suggested by their authors
    params_env['active_threshold_update'] = 0.01
    params_env['active_budget_window'] = 300
    params_env['active_budget_lambda'] = 1.0 - (1.0 / params_env['active_budget_window'])
    params_env['active_delta'] = 1.0  # N(1, delta) - no randomisation if set to 0

    # fixed
    params_env['seed'] = 0
    params_env['preq_fading_factor'] = 0.99
    params_env['flag_store'] = 1

    #####################
    # Settings: derived #
    #####################

    params_env['random_state'] = np.random.RandomState(params_env['seed'])

    params_env['num_init_per_class'] = params_env['memory_size']  # initial availability of L examples per class

    if params_env['drift_start_time'] == 'default':
        params_env['drift_start_time'] = int(params_env['time_steps'] / 2)

    ################
    # Output files #
    ################

    # file directory and names
    out_method = params_env['method']
    if params_env['flag_ensemble']:
        out_method = out_method + '_wm'

    out_dir = 'exps/'
    out_name = '{}_{}_{}_{}'.format(params_env['data_source'], out_method, params_env['memory_size'],
                                    params_env['active_budget_total'])

    # files to store g-mean
    filename_acc = os.path.join(os.getcwd(), out_dir, out_name + '_preq_acc.txt')
    filename_gmean = os.path.join(os.getcwd(), out_dir, out_name + '_preq_gmean.txt')
    filename_counter = os.path.join(os.getcwd(), out_dir, out_name + '_counter.txt')

    if params_env['flag_store']:
        create_file(filename_acc)
        create_file(filename_gmean)
        create_file(filename_counter)

    #########
    # Start #
    #########

    for r in range(params_env['repeats']):
        print('Repetition: ', r)

        # class probabilities
        if params_env['imbalance'] == 'balanced':
            params_env['probs'] = [1.0 / params_env['num_classes'], ] * params_env['num_classes']
        else:
            # set probs
            idx_minority = 0  # for single minority
            idx_majority = 0  # for multi minority

            if params_env['imbalance'] == 'single_minority':
                majority_ratio = (1.0 - params_env['minority_ratio']) / (params_env['num_classes'] - 1)
                params_env['probs'] = [majority_ratio] * params_env['num_classes']
                params_env['probs'][idx_minority] = params_env['minority_ratio']
            elif params_env['imbalance'] == 'multi_minority':
                majority_ratio = 1.0 - (params_env['minority_ratio'] * (params_env['num_classes'] - 1))
                params_env['probs'] = [params_env['minority_ratio']] * params_env['num_classes']
                params_env['probs'][idx_majority] = majority_ratio

        # create nn
        if params_env['flag_ensemble']:
            params_env['ensemble'] = create_nn_ensemble(params_env, params_nn)
        else:
            params_env['nn'] = create_nn_single(params_env, params_nn)

        # start
        preq_general_accs, _, preq_gmeans, num_labels = run(params_env)

        # store
        if params_env['flag_store']:
            write_to_file(filename_acc, preq_general_accs)
            write_to_file(filename_gmean, preq_gmeans)
            write_to_file(filename_counter, num_labels)

###########################################################################################
#                                        Start                                            #
###########################################################################################


if __name__ == "__main__":
    main()
