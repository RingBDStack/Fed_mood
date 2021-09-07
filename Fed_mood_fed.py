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


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pk.load(f)


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


def split_data(subject_ids, params):
    if params['test_subject'] == -1:
        train_data, train_label, _, _ = load_data(subject_ids, 0.8, params)
        test_data, test_label, _, _ = load_data(subject_ids, -0.2, params)
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
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[mode][0].shape[1]))(
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
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[mode][0].shape[1]))(
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
        for mode in train_data.keys():
            if params['includes_%s' % mode]:
                sub_input = Input(shape=(params['seq_max_len'], train_data[mode][0].shape[1]))
                input_list.append(sub_input)
                train_list.append(train_data[mode])
                test_list.append(test_data[mode])
                mask_input = Masking(mask_value=0, input_shape=(params['seq_max_len'], train_data[mode][0].shape[1]))(
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

def iid(train_label,test_label, params):
    dict_users, all_idxs = {}, [i for i in range(len(train_label))]
    np.random.seed(0)
    for i in range(params['num_users']):
        dict_users[i] = set(np.random.choice(all_idxs, params['data'], replace=False))
        if params['data']<=(11968/params['num_users']):
            all_idxs = list(set(all_idxs) - dict_users[i])
    for i in range(params['num_users']):
        dict_users[i] = list(dict_users[i])
    num_items_test = int(3003*params['data']/(11968/params['num_users'])/params['num_users'])
    dict_users_test, all_idxs_test = {}, [i for i in range(len(test_label))]
    for i in range(params['num_users']):
        dict_users_test[i] = set(np.random.choice(all_idxs_test, num_items_test, replace=False))
        if params['data'] <= (11968/params['num_users']):
            all_idxs_test = list(set(all_idxs_test) - dict_users_test[i])
    for i in range(params['num_users']):
        dict_users_test[i] = list(dict_users_test[i])
    return dict_users,dict_users_test

def FedAvg(w,params):
    w_avg = []
    c={}
    load=copy.deepcopy(w)
    for i in range(len(load)):
        for j in range(len(load[i])):
            load[i][j] = np.zeros(shape=np.shape(load[i][j]))
    if params['idx'] == 3:
        for i in range(6):
            c[i] = load[i]
        for k in range(6):
            for i in range(len(w[k:len(w):6])):
                for j in range(len(w[k:len(w):6][0])):
                    c[k][j] += w[k:len(w):6][i][j]
            for l in range(len(c[k])):
                c[k][l] = c[k][l] * 6 / len(w)
        for i in range(6):
            w_avg.append(c[i])
    else:
        for i in range(5):
            c[i] = load[i]
        for k in range(5):
            for i in range(len(w[k:len(w):5])):
                for j in range(len(w[k:len(w):5][0])):
                    c[k][j] += w[k:len(w):5][i][j]
            for l in range(len(c[k])):
                c[k][l] = c[k][l] * 5 / len(w)
        for i in range(5):
            w_avg.append(c[i])
    return w_avg

def run_model(train_data, test_data, y_train, y_test, params):
    num=[6,7,8,13,14]
    model, X_train, X_test = create_model(train_data, test_data, params)
    dict_users,dict_users_test = iid(y_train,y_test, params)
    w_glob,res = [],[]
    for iter in range(params['n_epochs']):
        w_locals ,w_temps= [],[]
        idxs_users = np.random.choice(range(params['num_users']), random.randint(1,params['num_users']), replace=False)
        print('第:',iter,'轮,共',idxs_users,'个用户')
        
        if iter == 0:
            for i in num:
                  w_temps.append(copy.deepcopy(model.layers[i].get_weights()))
            for idx in idxs_users:
                print('第:',iter,'轮,用户',idx)
                j = 0
                for i in num:
                    model.layers[i].set_weights(w_temps[j])
                    j += 1
                
                X_train_fed, y_train_fed=[],[]
                for k in range(3):
                    a = X_train[k][dict_users[idx][0]]
                    for i in dict_users[idx]:
                        if i == dict_users[idx][0]:
                            continue
                        else:
                            a = np.concatenate((a, X_train[k][i]))
                    if k==0:
                        a=a.reshape(params['data'],100,4)
                    elif k==1:
                        a=a.reshape(params['data'],100,6)
                    else:
                        a=a.reshape(params['data'],100,3)
                    X_train_fed.append(a)
                for i in dict_users[idx]:
                    y_train_fed.append(y_train[i])
                if params['flag']==0:
                    hist = model.fit(x=X_train_fed, y=y_train_fed, batch_size=params['batch_size'], verbose=2, \
                                 epochs=20, validation_split=0.2)
                else:
                    hist = model.fit(x=X_train_fed, y=y_train_fed, batch_size=params['batch_size'], verbose=2, \
                                 epochs=15,validation_split=0.2)
                for i in num:
                    w_locals.append(copy.deepcopy(model.layers[i].get_weights()))
            w_glob = FedAvg(w_locals,params)  # update global weights
        else:
            j=0
            for i in num:
                model.layers[i].set_weights(w_glob[j]) 
                j+=1
            outcome=model.evaluate(X_test,y_test,verbose=0)
            res.append(outcome[1])
            print('第:',iter,'轮,test acc:',outcome)
            for i in num:
                w_temps.append(copy.deepcopy(model.layers[i].get_weights()))  
            for idx in idxs_users:
                print('第:',iter,'轮,用户',idx)
                j=0
                for i in num:
                    model.layers[i].set_weights(w_temps[j])  
                    j += 1
                X_train_fed, y_train_fed = [], []
                for k in range(3):
                    a = X_train[k][dict_users[idx][0]]
                    for i in dict_users[idx]:
                        if i == dict_users[idx][0]:
                            continue
                        else:
                            a = np.concatenate((a, X_train[k][i]))
                    if k==0:
                        a=a.reshape(params['data'],100,4)
                    elif k==1:
                        a=a.reshape(params['data'],100,6)
                    else:
                        a=a.reshape(params['data'],100,3)
                    X_train_fed.append(a)
                for i in dict_users[idx]:
                    y_train_fed.append(y_train[i])
                if params['flag']==0:
                    hist = model.fit(x=X_train_fed, y=y_train_fed, batch_size=params['batch_size'], verbose=2, \
                                 epochs=20, validation_split=0.2)
                else:
                    hist = model.fit(x=X_train_fed, y=y_train_fed, batch_size=params['batch_size'], verbose=2, \
                                 epochs=15,validation_split=0.2)
                for i in num:
                    w_locals.append(copy.deepcopy(model.layers[i].get_weights()))
            w_glob = FedAvg(w_locals,params)  # update global weights
    j = 0
    for i in num:
        model.layers[i].set_weights(w_glob[j])
        j+= 1
    outcome = model.evaluate(X_test, y_test, verbose=0)
    res.append(outcome[1])
    model.save_weights('DNN_iid_weights_user%d_data%d_round%d.h5' % (params['num_users'],params['data'],params['round']))
    print('test acc:',outcome)
    y_score = model.predict(X_test, batch_size=params['batch_size'], verbose=0)
    y_pred = (np.ravel(y_score) > 0.5).astype('int32') if params['is_clf'] else np.ravel(y_score)  
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
          'num_users': 12,
          'frac': 0.1,
          'data':1500,
          'round':1,
          'includes_alphanum': 1,
          'includes_special': 1,
          'includes_accel': 1,
          'flag':1}
subject_ids = [7, 13, 15, 16, 17, 19, 20, 21, 24, 25, 27, 30, 31, 32, 33, 36, 37, 38, 39, 40]
train_data, y_train, test_data, y_test = split_data(subject_ids, params)
y_pred, hist = run_model(train_data, test_data, y_train, y_test, params)
res = evaluate(y_test, y_pred, params)