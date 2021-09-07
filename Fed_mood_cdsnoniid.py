'''
A demo of Fed_mood on BiAffect data.
'''

from __future__ import print_function
import sys
import os.path
import random
import pandas as pd
import numpy as np
import pickle as pk
import re
import tensorflow as tf
import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import *
from keras.optimizers import *
from keras.utils import *
from keras.regularizers import *
from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.normalization import *
from keras.callbacks import LambdaCallback


def load_mode_data(mode, subject_id, params):
    data = {}
    filename = "mood\%s_%d.pickle" % (mode, subject_id)
    df = pd.read_pickle(open(filename, "rb"))
    df['timestamp'] = df['timestamp'].dt.date
    if mode == 'alphanum':
        df['x'] = df['x'].apply(lambda x: x / 2276)
        df['y'] = df['y'].apply(lambda x: x / 959)
        df['dt'] = df['dt'].fillna(0).apply(lambda x: min(x, 5)).apply(lambda x: np.log(x + 1) / np.log(6))
        df['dr'] = df['dr'].apply(lambda x: min(max(50, x), 1000)).apply(lambda x: np.log(x - 49) / np.log(951))
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i][['x', 'y', 'dt', 'dr']].values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i][['x', 'y', 'dt', 'dr']].values
    if mode == 'special':
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i].drop(['timestamp', 'session_number'], 1).values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i].drop(['timestamp', 'session_number'], 1).values
    if mode == 'accel':
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(lambda x: x / 39.2266)
        if params['level'] == 0:
            for i in df['session_number'].unique():
                data[i] = df[lambda x: x.session_number == i][['x', 'y', 'z']].values
        if params['level'] == 1:
            for i in df['timestamp'].unique():
                data[i] = df[lambda x: x.timestamp == i][['x', 'y', 'z']].values
    return data


def load_subject_data(modes, subject_id, params):
    level_name = ['session', 'day']
    filename = "mood\%s_%d.pickle" % (level_name[params['level']], subject_id)
    if os.path.isfile(filename):
        return pd.read_pickle(open(filename, "rb"))
    mdata = {}
    sdata = {}
    for mode in modes:
        mdata[mode] = load_mode_data(mode, subject_id, params)
        sdata[mode] = []
    label = []
    timestamp = []
    ratingfile = "mood\weekly_ratings_%d.pickle" % subject_id
    df_rating = pd.read_pickle(open(ratingfile, "rb"))
    datefile = "mood\%s_%d.pickle" % (modes[0], subject_id)
    df_date = pd.read_pickle(open(datefile, "rb"))
    for i in mdata[modes[0]].keys():
        if all(i in mdata[mode].keys() and len(mdata[mode][i]) >= params['seq_min_len'] for mode in modes):
            for mode in modes:
                lists = np.ndarray.tolist(np.transpose(mdata[mode][i]))
                pad_matrix = pad_sequences(lists, maxlen=params['seq_max_len'], dtype='float32')
                pad_matrix = np.transpose(pad_matrix)
                sdata[mode].append(pad_matrix)
            if params['level'] == 0:
                rating_date = df_date[lambda x: x.session_number == i].iloc[0].timestamp.date()
            if params['level'] == 1:
                rating_date = i
            ratings = df_rating[lambda x: x.date == rating_date.strftime('%Y-%m-%d')].iloc[0]
            label.append([ratings.sighd_17item, ratings.ymrs_total])
            timestamp.append(rating_date)
    pk.dump((sdata, label, timestamp), open(filename, 'wb'), protocol=pk.HIGHEST_PROTOCOL)
    return sdata, label, timestamp


def convert_label(labels, params):
    if params['is_clf']:
        for i in range(len(labels)):
            labels[i] = labels[i][0]
            if labels[i] >= 8:
                labels[i] = 1
            else:
                labels[i] = 0
        return labels
    else:
        for i in range(len(labels)):
            labels[i] = labels[i][1]
        return np.array(labels)

def converts_label(labels, params,subject_ids):
    if params['is_clf']:
        for i in subject_ids:
            for j in range(len(labels[i])):
                labels[i][j] = labels[i][j][0]
                if labels[i][j] >= 8:
                    labels[i][j] = 1
                else:
                    labels[i][j] = 0
        return labels
    else:
        for i in range(len(labels)):
            labels[i] = labels[i][1]
        return np.array(labels)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)


def loads_data(subject_ids, ratio, params):
    data = {}
    label = {}
    timestamp = {}
    n = 0
    for i in subject_ids:
        data[i], label[i], timestamp[i] = load_subject_data(modes, i, params)
        m = len(label[i])
        if ratio > 0:
            idx = range(int(m * ratio))
        else:
            idx = range(int(m * (ratio + 1)), m)
        for mode in modes:
            data[i][mode] = [data[i][mode][j] for j in idx]
        label[i] = [label[i][j] for j in idx]
        timestamp[i] = [timestamp[i][j] for j in idx]
        n += len(idx)
    idx = range(n)
    random.seed(0)
    idx = random.sample(idx, k=len(idx))
    datas = {}
    for mode in modes:
        datas[mode] = np.concatenate([np.asarray(data[i][mode]) for i in subject_ids])
        datas[mode] = np.asarray([datas[mode][i] for i in idx])
        np.asarray(data[i][mode])
    a = {}
    save_obj(label, 'tt')
    for i in subject_ids:
        a[i] = label[i][0]
    for i in subject_ids:
        for j in range(len(label[i])):
            if j == 0:
                continue
            else:
                a[i].append(label[i][j]) 
    newlabel = load_obj('tt')
    labels = np.concatenate([label[i] for i in subject_ids], axis=0)
    ll = 0
    for i in subject_ids:
        labels[ll] = newlabel[i][0]
        ll += len(newlabel[i])
    labels = [labels[i] for i in idx]
    labels = convert_label(labels, params)
    timestamps = np.concatenate([timestamp[i] for i in subject_ids], axis=0)
    timestamps = [timestamps[i] for i in idx]
    whose = np.concatenate([[i] * len(label[i]) for i in subject_ids], axis=0)
    whose = [whose[i] for i in idx]
    return datas, labels, timestamps, whose

def load_data(subject_ids, ratio, params):
    data = {}
    label = {}
    timestamp = {}
    n = 0
    for i in subject_ids:
        data[i], label[i], timestamp[i] = load_subject_data(modes, i, params)
        m = len(label[i])
        if ratio > 0:
            idx = range(int(m * ratio))
        else:
            idx = range(int(m * (ratio + 1)), m)
        for mode in modes:
            data[i][mode] = [data[i][mode][j] for j in idx]
        label[i] = [label[i][j] for j in idx]
        timestamp[i] = [timestamp[i][j] for j in idx]
        n += len(idx)
    random.seed(0)
    datas = copy.deepcopy(data)
    for j in subject_ids:
        idx = random.sample(range(len(data[j]['accel'])), len(data[j]['accel']))
        for mode in modes:
            k = 0
            for i in idx:
                datas[j][mode][k] = np.asarray(data[j][mode][i])
                k = k + 1
            datas[j][mode]=np.concatenate([np.asarray(datas[j][mode])])
    labels = copy.deepcopy(label)
    for i in subject_ids:
        idx = random.sample(range(len(label[i])), len(label[i]))
        k = 0
        for j in idx:
            labels[i][k] = label[i][j]
            k = k + 1
    labels = converts_label(labels, params,subject_ids)
    return datas, labels


def split_data(subject_ids, params):
    if params['test_subject'] == -1:
        train_data, train_label = load_data(subject_ids, 0.8, params)
        test_data, test_label, _, _ = loads_data(subject_ids, -0.2, params)
    else:
        train_subjects = list(subject_ids)
        train_subjects.remove(params['test_subject'])
        train_data, train_label, _, _ = load_data(train_subjects, 1, params)
        test_data, test_label, _, _ = load_data([params['test_subject']], 1, params)
    return train_data, train_label, test_data, test_label


def evaluate(y_test, y_pred, params):
    res = {}
    if params['is_clf']:
        res['accuracy'] = float(sum(y_test == y_pred)) / len(y_test)
        res['precision'] = float(sum(y_test & y_pred) + 1) / (sum(y_pred) + 1)
        res['recall'] = float(sum(y_test & y_pred) + 1) / (sum(y_test) + 1)
        res['f_score'] = 2.0 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    else:
        res['rmse'] = np.sqrt(np.mean(np.square(y_test - y_pred)))
        res['mae'] = np.mean(np.abs(y_test - y_pred))
        res['explained_variance_score'] = 1 - np.square(np.std(y_test - y_pred)) / np.square(np.std(y_test))
        res['r2_score'] = 1 - np.sum(np.square(y_test - y_pred)) / np.sum(np.square(y_test - np.mean(y_test)))
    print(' '.join(["%s: %.4f" % (i, res[i]) for i in res]))
    return res


def mvm_decision_function(arg):
    n_modes = len(arg)
    latentx = arg
    y = K.concatenate([K.sum(
        K.prod(K.stack([latentx[j][:, i * params['n_latent']: (i + 1) * params['n_latent']] for j in range(n_modes)]),
               axis=0), \
        axis=-1, keepdims=True) for i in range(params['n_classes'])])
    return y


def fm_decision_function(arg):
    latentx, bias = arg[0], arg[1]
    pairwise = K.concatenate([K.sum(K.square(latentx[:, i * params['n_latent']: (i + 1) * params['n_latent']]), \
                                    axis=-1, keepdims=True) for i in range(params['n_classes'])])
    y = K.sum(K.tf.stack([pairwise, bias]), axis=0)
    return y


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true > 0.5, y_pred > 0.5))


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


def create_model(train_data, test_data, params):
    input_list = []
    output_list = []
    train_list = []
    test_list = []
    if params['idx'] == 1:
        for mode in train_data[7].keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))
                input_list.append(sub_input)
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))(
                    sub_input)
                sub_output = Bidirectional(
                    GRU(output_dim=params['n_hidden'], return_sequences=False, consume_less='mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                output_list.append(drop_output)
        x = merge(output_list, mode='concat') if len(output_list) > 1 else output_list[0]
        latentx = Dense(params['n_classes'] * params['n_latent'], activation='relu',
                        bias=True if params['bias'] else False)(x)
        y = Dense(params['n_classes'], bias=False)(latentx)
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    if params['idx'] == 2:
        for mode in train_data[7].keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))
                input_list.append(sub_input)
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))(
                    sub_input)
                sub_output = Bidirectional(
                    GRU(output_dim=params['n_hidden'], return_sequences=False, consume_less='mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                output_list.append(drop_output)
        x = merge(output_list, mode='concat') if len(output_list) > 1 else output_list[0]
        latentx = Dense(params['n_latent'] * params['n_classes'], bias=False)(x)
        bias = Dense(params['n_classes'], bias=True if params['bias'] else False)(x)
        y = merge([latentx, bias], mode=fm_decision_function, output_shape=(params['n_classes'],))
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    if params['idx'] == 3:
        for mode in train_data[7].keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))
                input_list.append(sub_input)
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[7][mode][0].shape[1]))(
                    sub_input)
                sub_output = Bidirectional(
                    GRU(output_dim=params['n_hidden'], return_sequences=False, consume_less='mem'))(mask_input)
                drop_output = Dropout(params['dropout'])(sub_output)
                latentx = Dense(params['n_latent'] * params['n_classes'], bias=True if params['bias'] else False)(
                    drop_output)
                output_list.append(latentx)
        y = merge(output_list, mode=mvm_decision_function, output_shape=(params['n_classes'],))
        y_act = Activation('sigmoid')(y) if params['is_clf'] else Activation('linear')(y)
        objective = 'binary_crossentropy' if params['is_clf'] else 'mean_squared_error'
        metric = [acc] if params['is_clf'] else [rmse]
    model = Model(input=input_list, output=y_act)
    model.compile(loss=objective, optimizer=RMSprop(lr=params['lr']), metrics=metric)
    model.summary()
    return model, train_list, test_list


def noniid(train_label, num_users):
    num_shards, num_imgs = 8, 1496
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.asarray(train_label)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def FedAvg(w,lengthdict,params):
    w_avg = []
    c={}
    load=copy.deepcopy(w)
    for i in range(len(load)):
        for j in range(len(load[i])):
            load[i][j] = np.zeros(shape=np.shape(load[i][j]))
    sum=0
    for i in range(len(lengthdict)):
        sum+=lengthdict[i]
    if params['idx'] == 3:
        for i in range(6):
            c[i] = load[i]
        for k in range(6):
            for i in range(len(w[k:len(w):6])):
                for j in range(len(w[k:len(w):6][0])):
                    c[k][j] += w[k:len(w):6][i][j]*lengthdict[i]/sum
        for i in range(6):
            w_avg.append(c[i])
    else:
        for i in range(5):
            c[i] = load[i]
        for k in range(5):
            for i in range(len(w[k:len(w):5])):
                for j in range(len(w[k:len(w):5][0])):
                    c[k][j] += w[k:len(w):5][i][j]*lengthdict[i]/sum
        for i in range(5):
            w_avg.append(c[i])
    return w_avg


def run_model(train_data, test_data, y_train, y_test, params):
    num = [6, 7, 8, 12, 13, 14]
    X_train, X_test = {}, []
    X_train_fed, y_train_fed = {}, {}
    model, _, _ = create_model(train_data, test_data, params)
    users={}
    users[0]=[24,25,39,40]
    users[1]=[7,15,19,20,21,27]
    users[2]=[13,16,17,31,32,33,36,37,38]
    np.random.seed(params['seed'])
    for i in range(3):
        if i==2:
            users[i]=np.random.choice(users[i], 8, replace=False)
        else:
            users[i] = np.random.choice(users[i], 4, replace=False)
    for mode in modes:
        X_test.append(test_data[mode])
    for i in range(params['num_users']):
        X_train[i],X_train_fed[i] = {},[]
        y_train_fed[i]=[]
        user = []
        for k in range(3):
            if k==2:
                b=i*2
                y_train_fed[i]+= y_train[users[k][b]]
                user.append(users[k][b])
                b=b+1
                y_train_fed[i] += y_train[users[k][b]]
                user.append(users[k][b])
            else:
                y_train_fed[i] += y_train[users[k][i]]
                user.append(users[k][i])
        for mode in modes:
            X_train[i][mode] = np.concatenate([np.asarray(train_data[j][mode]) for j in user])
            X_train_fed[i].append(X_train[i][mode])
    np.random.seed(0)
    w_glob, res = [], []
    X_train_merge = []
    y_train_merge = []

    for k in range(4):
        if k==0:
            a = X_train_fed[k]
            b= y_train_fed[k]
        else:
            b+=y_train_fed[k]
            for i in range(3):
                a[i] = np.concatenate((a[i], X_train_fed[k][i]))
    hist = model.fit(a, b, batch_size=params['batch_size'], verbose=2, \
                     nb_epoch=params['n_epochs'],
                     validation_data=(X_test, y_test))
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32') if params['is_clf'] else np.ravel(y_score)
    evaluate(y_test, y_pred, params)
    return y_pred, hist.history

modes = ['alphanum', 'special', 'accel']
params = {'seq_min_len': 10,
          'seq_max_len': 100,
          'batch_size': 256,
          'lr': 0.001,
          'dropout': 0.1,
          'n_epochs': 400,
          'n_hidden': 8,
          'n_latent': 8,
          'n_classes': 1,
          'bias': 1,
          'is_clf': 1,
          'idx': 1,  # 1: dnn, 2: dfm, 3: dmvm
          'test_subject': -1,
          'level': 0,
          'num_users': 4,
          'data': 1500,
          'frac': 0.1,
          'seed':3,
          'epochs':1,
          'includes_alphanum': 1,
          'includes_special': 1,
          'includes_accel': 1,
          'flag': 0}
subject_ids = [7, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 30, 31, 32, 33, 36, 37, 38, 39, 40]
train_data, y_train, test_data, y_test = split_data(subject_ids, params)
y_pred, hist = run_model(train_data, test_data, y_train, y_test, params)
