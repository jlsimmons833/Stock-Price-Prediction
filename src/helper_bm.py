#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20-10-2023 11:11:11

@author: jlsimmons33
"""

import pprint
import numpy as np
import matplotlib.pyplot as plt


def normalize_windows(win_data):
    """ Normalize a window
    Input: Window Data
    Output: Normalized Window

    Note: Run from load_data()

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)
    """
    norm_data = []
    for w in win_data:
        norm_data.append(float(w) / float(win_data[0]) - 1)
    return norm_data


def load_data(filename, seq_len, norm_win):
    """
    Loads the data from a csv file into arrays

    Input: Filename, sequence Length, normalization window(True, False)
    Output: X_tr, Y_tr, X_te, Y_te

    Note: Normalization data using n_i = (p_i / p_0) - 1,
    denormalization using p_i = p_0(n_i + 1)

    Note: Run from timeSeriesPredict.py
    """
    fid = open(filename, 'r').read()
    data = fid.split('\n')
    data = data[:len(data) - 1]
    #pprint.pprint(data)
    if norm_win:
        data = normalize_windows(data)
    X_te = np.array(data)
    #X_te = np.reshape(X_te, (X_te.shape[0], X_te.shape[1], 1))
    #pprint.pprint(X_te)
    return [X_te]


def predict_seq_mul(model, data, win_size, pred_len):
    """
    Predicts multiple sequences
    Input: keras model, testing data, window size, prediction length
    Output: Predicted sequence

    Note: Run from timeSeriesPredict.py
    """
    
    #prediction = model.predict(data)
    pprint.pprint(data)
    pred_seq = []
    for i in range(len(data)//pred_len):
        current = data[i * pred_len]
        predicted = []
        for j in range(pred_len):
            predicted.append(model.predict(current[None, :, :])[0, 0])
            current = current[1:]
            current = np.insert(current, [win_size - 1], predicted[-1], axis=0)
        pred_seq.append(predicted)
    return pred_seq
    # return prediction

def predict_pt_pt(model, data):
    """
    Predicts only one timestep ahead
    Input: keras model, testing data
    Output: Predicted sequence

    Note: Run from timeSeriesPredict.py
    """
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size, ))
    return predicted


def plot_mul(Y_hat, Y, pred_len):
    """
    PLots the predicted data versus true data

    Input: Predicted data, True Data, Length of prediction
    Output: return plot

    Note: Run from timeSeriesPredict.py
    """
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(Y, label='Y')
    # Print the predictions in its respective series-length
    for i, j in enumerate(Y_hat):
        shift = [None for p in range(i * pred_len)]
        plt.plot(shift + j, label='Y_hat')
        plt.legend()
    plt.show()
