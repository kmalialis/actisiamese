# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from itertools import combinations
from keras.utils import to_categorical
from sklearn.datasets import make_blobs

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

########
# Data #
########


# Get synthetic data samples
def sample_data(pair, params_env, flag_rescale=True):
    # pair: tuple (class index, num_samples)
    # xy: np array of shape (num_samples, 3)

    def sample_data_sea10():
        num = 5000
        xs = params_env['random_state'].rand(num * num_samples, 2) * params_env['x_max'][0]
        sums = xs[:, 0] + xs[:, 1]
        idx = np.digitize(sums, params_env['sea_bins']) - 1  # minus 1 to start from class 0
        selected_idx = np.where(idx == class_idx)[0]
        out_x = xs[selected_idx, :]
        out_x1 = out_x[:num_samples, 0]
        out_x2 = out_x[:num_samples, 1]

        return [out_x1, out_x2]

    def sample_data_circles10():
        circle = params_env['circles'][class_idx]
        c_x1, c_x2, c_r = circle

        r = c_r * np.sqrt(params_env['random_state'].rand(num_samples))
        theta = params_env['random_state'].rand(num_samples) * 2.0 * np.pi
        out_x1 = c_x1 + r * np.cos(theta)
        out_x2 = c_x2 + r * np.sin(theta)

        return [out_x1, out_x2]

    def sample_data_blobs12():
        xs, ys = make_blobs(n_samples=num_samples * params_env['num_classes'],  # number of samples *per class*
                            n_features=params_env['num_features'],
                            centers=np.array([i for i in params_env['blobs'].values()]),
                            cluster_std=params_env['blob_std'],
                            shuffle=False,
                            random_state=params_env['random_state'])

        idx = np.where(ys == class_idx)[0]
        selected_xs = xs[idx, :]

        xs = []
        for i in range(selected_xs.shape[1]):
            xs.append(np.clip(selected_xs[:, i], a_max=params_env['x_max'][i], a_min=params_env['x_min'][i]))

        return xs

    # extract info
    class_idx, num_samples = pair

    # sample elements
    if params_env['data_source'] == 'sea10':
        xs = sample_data_sea10()
    elif params_env['data_source'] == 'circles10':
        xs = sample_data_circles10()
    elif params_env['data_source'] == 'blobs12':
        xs = sample_data_blobs12()

    # rescale
    if flag_rescale:
        for i in range(len(xs)):
            xs[i] = (xs[i] - params_env['x_min'][i]) / (params_env['x_max'][i] - params_env['x_min'][i])

    # features
    x = np.stack(xs, axis=1)

    # labels
    y = np.repeat(np.array([class_idx]), num_samples)
    y = np.reshape(y, (y.shape[0], 1))

    # merge
    xy = np.hstack((x, y))

    return xy


# Get samples in dict of queues
def get_sample(params_env, flag_init):
    # get pairs
    if flag_init:
        unique = range(params_env['num_classes'])
        num_samples = params_env['num_init_per_class']
        counts = [int(num_samples), ] * params_env['num_classes']
        pairs = list(zip(unique, counts))
        mem_size = params_env['memory_size']
    else:
        unique = np.arange(params_env['num_classes'])
        cls = params_env['random_state'].choice(a=unique, size=1, replace=True, p=params_env['probs'])[0]
        pairs = [(cls, 1)]
        mem_size = 1

    # get sample from pairs
    d_xs = {}
    for pair in pairs:
        xs = sample_data(pair, params_env)
        d_xs[pair[0]] = deque(xs, maxlen=mem_size)

    return d_xs


##########################
# Prequential evaluation #
##########################

def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric


##################
# Model training #
##################


# data prep for ActiSiamese training
def siamese_prep_training(d, params_env):
    # get all pairs
    pairs = [x for k, q in d.items() for x in q]

    # identical pairs
    input_left_identical = np.asarray(pairs)[:, :-1]
    input_right_identical = np.asarray(pairs)[:, :-1]

    # pairs with same & different class
    input_left_same, input_right_same = [], []
    input_left_diff, input_right_diff = [], []

    for outer_pair in combinations(pairs, 2):
        left_pair = outer_pair[0]
        right_pair = outer_pair[1]

        if left_pair[-1] == right_pair[-1]:  # same class
            input_left_same.append(left_pair[:-1])
            input_right_same.append(right_pair[:-1])
        else:  # different class
            input_left_diff.append(left_pair[:-1])
            input_right_diff.append(right_pair[:-1])

    input_left_same = np.asarray(input_left_same)
    input_right_same = np.asarray(input_right_same)

    input_left_diff = np.asarray(input_left_diff)
    input_right_diff = np.asarray(input_right_diff)

    # positive pairs
    input_left_id_same = np.vstack((input_left_identical, input_left_same))
    input_right_id_same = np.vstack((input_right_identical, input_right_same))

    # balance pairs
    size_id_same = input_left_id_same.shape[0]
    size_diff = input_left_diff.shape[0]

    if size_id_same < size_diff:  # shrink different pairs
        idx = params_env['random_state'].choice(a=range(size_diff), size=size_id_same, replace=False)
        input_left_diff = input_left_diff[idx, :]
        input_right_diff = input_right_diff[idx, :]

    elif size_id_same > size_diff:  # shrink identical + same pairs
        idx = params_env['random_state'].choice(a=range(size_id_same), size=size_diff, replace=False)
        input_left_id_same = input_left_id_same[idx, :]
        input_right_id_same = input_right_id_same[idx, :]

    # merge pairs
    input_left = np.vstack((input_left_id_same, input_left_diff))
    input_right = np.vstack((input_right_id_same, input_right_diff))

    # labels
    y_id_same = np.ones((input_left_id_same.shape[0], 1))
    y_diff = np.zeros((input_left_diff.shape[0], 1))
    y = np.vstack((y_id_same, y_diff))

    # return
    return [input_left, input_right], y


# data prep for ActiQ training
def fc_prep_training(d, params_env):
    # unfold dict
    xy = [a for _, q in d.items() for a in q]
    xy = np.vstack(xy)

    # features
    x = xy[:, :-1]

    # target
    y = xy[:, -1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')
    y = np.reshape(y, (y.shape[0], 1))

    return x, y, y_encoded


# data prep for Incremental training
def incr_fc_prep_training(xy, params_env):
    # features
    x = xy[:-1]

    # target
    y = xy[-1]
    y_encoded = to_categorical(y, num_classes=params_env['num_classes'], dtype='float32')

    # reshape
    x = np.reshape(x, (1, x.shape[0]))
    y_encoded = np.reshape(y_encoded, (1, y_encoded.shape[0]))
    y = np.reshape(y, (1, 1))

    return x, y, y_encoded


# train model (single classifier)
def prep_and_train(d, xy, params_env):
    x = None
    y = None

    # get x and y
    if params_env['method'] == 'rvus':  # y is y_encoded here
        x, _, y = incr_fc_prep_training(xy, params_env)
    elif params_env['method'] == 'actiq':  # y is y_encoded here
        x, _, y = fc_prep_training(d, params_env)
    elif params_env['method'] == 'actisiamese':
        x, y = siamese_prep_training(d, params_env)

    # train
    params_env['nn'].train(x, y)


# train ensemble (of ActiQ's OR ActiSiamese's but not hybrid)
def prep_and_train_ensemble(d_xy, xy, params_env):
    x = None
    y = None

    if params_env['method'] == 'rvus':
        x, _, y = incr_fc_prep_training(xy, params_env)
        for nn in params_env['ensemble']:
            nn.train(x, y)
    else:
        for nn in params_env['ensemble']:
            if params_env['method'] == 'actiq':
                x, _, y = fc_prep_training(d_xy, params_env)
            elif params_env['method'] == 'actisiamese':
                x, y = siamese_prep_training(d_xy, params_env)

            nn.train(x, y)


####################
# Model prediction #
####################


# data prep for ActiSiamese prediction
def data_prep_for_predict(d, x):
    nn_input_xy = np.array([a for _, v in d.items() for a in v])
    nn_input_y = nn_input_xy[:, -1]
    nn_input_1_x = nn_input_xy[:, :-1]
    nn_input_2_x = np.tile(x, (nn_input_1_x.shape[0], 1))

    return nn_input_y, nn_input_1_x, nn_input_2_x


###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(params_env):

    ######################
    # Init preq. metrics #
    ######################

    # general accuracy
    preq_general_accs = []
    preq_general_acc_n = 0.0
    preq_general_acc_s = 0.0

    # class accuracies
    keys = range(params_env['num_classes'])
    preq_class_accs = {k: [] for k in keys}
    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

    # gmean
    preq_gmeans = []

    ####################
    # Init AL strategy #
    ####################

    active_threshold = 1.0
    budget_current = 0.0
    budget_u = 0.0
    d_counter = {k: 0 for k in keys}  # counter for num of labels requested per class

    #############
    # Init data #
    #############

    d_xy = get_sample(params_env, flag_init=True)

    #################
    # Init ensemble #
    #################

    if params_env['flag_ensemble']:
        arr_preds = np.zeros((params_env['num_classes'], params_env['ensemble_size']))
        weights = [1.0] * params_env['ensemble_size']

    #########
    # Start #
    #########

    for t in range(0, params_env['time_steps']):

        if t % 1000 == 0:
            print('Time step: ', t)

        #################
        # Concept drift #
        #################

        if params_env['flag_drift']:
            # abrupt
            if not params_env['flag_drift_recurrent']:
                if t == params_env['drift_start_time']:
                    # drifted dataset
                    if params_env['data_source'] == 'circles10':
                        params_env['circles'] = params_env['circles_drifted']
                    elif params_env['data_source'] == 'sea10':
                        params_env['sea_bins'] = params_env['sea_bins_drifted']
                    if params_env['data_source'] == 'blobs12':
                        params_env['blobs'] = params_env['blobs_drifted']
                        params_env['blob_std'] = params_env['blob_std_drifted']

                    # reset preq. metrics
                    preq_general_acc_n = 0.0
                    preq_general_acc_s = 0.0

                    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

            # recurrent
            else:
                if t in [5000, 15000]:
                    # drifted dataset
                    if params_env['data_source'] == 'circles10':
                        params_env['circles'] = params_env['circles_drifted']
                    elif params_env['data_source'] == 'sea10':
                        params_env['sea_bins'] = params_env['sea_bins_drifted']
                    if params_env['data_source'] == 'blobs12':
                        params_env['blobs'] = params_env['blobs_drifted']
                        params_env['blob_std'] = params_env['blob_std_drifted']

                    # reset preq. metrics
                    preq_general_acc_n = 0.0
                    preq_general_acc_s = 0.0

                    preq_class_acc = dict(
                        zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

                elif t == 10000:
                    # init dataset
                    if params_env['data_source'] == 'circles10':
                        params_env['circles'] = params_env['circles_init']
                    elif params_env['data_source'] == 'sea10':
                        params_env['sea_bins'] = params_env['sea_bins_init']
                    if params_env['data_source'] == 'blobs12':
                        params_env['blobs'] = params_env['blobs_init']
                        params_env['blob_std'] = params_env['blob_std_init']

                    # reset preq. metrics
                    preq_general_acc_n = 0.0
                    preq_general_acc_s = 0.0

                    preq_class_acc = dict(
                        zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

        ###############
        # Get example #
        ###############

        # get example
        d_temp = get_sample(params_env, flag_init=False)
        xy = [list(i) for i in d_temp.values()][0][0]
        xy = np.reshape(xy, (1, len(xy)))

        x = xy[0, :-1]
        y = xy[0, -1]

        # reshape here once to avoid reshaping multiple times later on
        x = np.reshape(x, (1, x.shape[0]))
        xy = np.reshape(xy, (xy.shape[1],))

        ###################
        # Predict example #
        ###################
        # Output:
        # y_pred_max: will be used by the AL strategy
        # pred_class: will be used to determine correctness (evaluation)

        if params_env['method'] in ['rvus', 'actiq']:
            if params_env['flag_ensemble']:
                ind_01_loss = [-1.0] * params_env["ensemble_size"]  # individual 0/1 loss

                for i in range(params_env["ensemble_size"]):
                    y_pred_i, _, pred_class_i = params_env['ensemble'][i].predict(x)
                    arr_preds[:, i] = y_pred_i
                    ind_01_loss[i] = 0 if y == pred_class_i else 1  # individual 0/1 loss

                avg_preds = np.average(arr_preds, axis=1, weights=weights)
                pred_class = np.argmax(avg_preds)
                y_pred_max = np.max(avg_preds)
            else:
                _, y_pred_max, pred_class = params_env['nn'].predict(x)

        elif params_env['method'] == 'actisiamese':
            if params_env['flag_ensemble']:
                ind_01_loss = [-1.0] * params_env["ensemble_size"]  # individual 0/1 loss

                for i in range(params_env["ensemble_size"]):
                    nn_input_y, nn_input_1_x, nn_input_2_x = data_prep_for_predict(d_xy, x)

                    y_pred_i = params_env['ensemble'][i].predict([nn_input_1_x, nn_input_2_x])
                    y_pred_i = np.hstack((y_pred_i, nn_input_y.reshape(nn_input_y.shape[0], 1)))

                    gba_i = np.array([np.mean(y_pred_i[y_pred_i[:, 1] == c][:, 0]) for c in np.unique(nn_input_y)])
                    gba_i = np.reshape(gba_i, (1, gba_i.shape[0]))
                    arr_preds[:, i] = gba_i

                    pred_class_i = np.argmax(gba_i)
                    ind_01_loss[i] = 0 if y == pred_class_i else 1  # individual 0/1 loss

                avg_preds = np.average(arr_preds, axis=1, weights=weights)
                pred_class = np.argmax(avg_preds)
                y_pred_max = np.max(avg_preds)
            else:
                nn_input_y, nn_input_1_x, nn_input_2_x = data_prep_for_predict(d_xy, x)
                y_pred = params_env['nn'].predict([nn_input_1_x, nn_input_2_x])
                y_pred = np.hstack((y_pred, nn_input_y.reshape(nn_input_y.shape[0], 1)))

                gba = np.array([[c, np.mean(y_pred[y_pred[:, 1] == c][:, 0])] for c in np.unique(nn_input_y)])
                gba_max = np.max(gba[:, 1])

                pred_class = gba[gba[:, 1] == gba_max][0][0]  # select class with highest average prediction
                arr = y_pred[y_pred[:, 1] == pred_class][:, 0]  # all predictions in selected class
                y_pred_max = np.max(arr)  # highest prediction in predicted class

        ###############
        # Correctness #
        ###############

        correct = 1 if y == pred_class else 0  # check if prediction was correct

        # update ensemble weights
        if params_env['flag_ensemble']:
            weights = [weights[i] * np.exp(- 0.5 * ind_01_loss[i]) for i in range(params_env['ensemble_size'])]
            weights = [w / sum(weights) for w in weights]  # normalise weights

        ########################
        # Update preq. metrics #
        ########################

        # update general accuracy
        preq_general_acc_s, preq_general_acc_n, preq_general_acc = \
            update_preq_metric(preq_general_acc_s, preq_general_acc_n, correct, params_env['preq_fading_factor'])
        preq_general_accs.append(preq_general_acc)

        # update class accuracies & gmean
        preq_class_acc_s[y], preq_class_acc_n[y], preq_class_acc[y] = update_preq_metric(
            preq_class_acc_s[y], preq_class_acc_n[y], correct, params_env['preq_fading_factor'])

        lst = []
        for k, v in preq_class_acc.items():
            preq_class_accs[k].append(v)
            lst.append(v)

        gmean = np.power(np.prod(lst), 1.0 / len(lst))
        preq_gmeans.append(gmean)

        ###################
        # Online learning #
        ###################
        # NOTE: This is different from setting the budget = 1.0 in active learning below

        if params_env['flag_learning'] == 'supervised':
            d_counter[y] += 1  # increase counter
            d_xy[y].append(xy)  # append new example
            prep_and_train(d_xy, xy, params_env)  # data prep and training

        ####################
        #  Active learning #
        ####################

        elif params_env['flag_learning'] == 'active':
            labelling = 0

            if budget_current < params_env['active_budget_total']:
                rnd = params_env['random_state'].normal(1.0, params_env['active_delta'])
                threshold = active_threshold * rnd

                if y_pred_max <= threshold:
                    labelling = 1  # set flag
                    d_counter[y] += 1  # increase counter
                    d_xy[y].append(xy)  # append to queues

                    # data prep and training
                    if params_env['flag_ensemble']:
                        prep_and_train_ensemble(d_xy, xy, params_env)
                    else:
                        prep_and_train(d_xy, xy, params_env)

                    # reduce AL threshold
                    active_threshold *= (1.0 - params_env['active_threshold_update'])
                else:
                    # increase AL threshold
                    active_threshold *= (1.0 + params_env['active_threshold_update'])

            # update budget
            budget_u = labelling + budget_u * params_env['active_budget_lambda']
            budget_current = budget_u / params_env['active_budget_window']

    # number of labels per class (this is to ensure order)
    num_labels = np.zeros(len(keys))
    for k in keys:
        num_labels[k] = d_counter[k]

    return preq_general_accs, preq_class_accs, preq_gmeans, num_labels
