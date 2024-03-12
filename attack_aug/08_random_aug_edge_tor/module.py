import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras import Sequential
from keras import layers
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.optimizers import adam_v2
from tensorflow.keras.utils import to_categorical
import random as rd
import collections as clt

def load_data(path):
    df = pd.read_csv(path, header=None)
    df = df.sort_values([1, 0])
    data = np.array(df)
    print('data shape:', data.shape)

    train = data[0:8000]
    train_data = train[:, 2:502]
    train_index = train[:, 0]
    train_index = to_categorical(train_index)
    print('train data shape:', train_data.shape)

    test = data[8000:10000]
    test_data = test[:, 2:502]
    test_index = test[:, 0]
    test_index = to_categorical(test_index)
    print('test data shape:', test_data.shape)
    
    return train_data, train_index, test_data, test_index

def load_data_ow(path):
    df_ow = pd.read_csv(path, header=None)
    data_ow = np.array(df_ow)
    test_data_ow = data_ow[:, 2:502]
    print('test data ow shape:', test_data_ow.shape)
    
    return test_data_ow

def get_list(train_data):
    data_count = clt.Counter(train_data.flatten())
    data_count = dict(sorted(data_count.items(), key = lambda x:x[1], reverse = True))
    size_list = list(data_count.keys())[1:701]
    size_count = list(data_count.values())[1:701]
    size_list_count = list()
    for i in range(700):
        for j in range(size_count[i]):
            size_list_count.append(size_list[i])
    size_list = np.array(size_list)
    size_count = np.array(size_count)
    size_list_count = np.array(size_list_count)
    return size_list, size_count, size_list_count

def random_defense(input_data, prob, size):
    
    output_data = np.zeros((input_data.shape[0], input_data.shape[1] * 2))
    
    for i in range(input_data.shape[0]):
        index = 0
        for j in range(input_data.shape[1]):
            if input_data[i][j] == 0:
                break
            output_data[i][index] = input_data[i][j]
            index += 1
            if rd.randint(0, 100) < prob:
                insert = 0
                if rd.randint(0, 1) == 0:
                    insert = rd.randint(-size, -54)
                else:
                    insert = rd.randint(54, size)
                output_data[i][index] = insert
                index += 1
                
                
            if index >= input_data.shape[1] * 2:
                break
                
    return output_data

def filter_attack(input_data, train_data):
    size_list, size_count, size_list_count = get_list(train_data)
    
    filter_data = input_data.copy()
    
    for i in range(filter_data.shape[0]):
        for j in range(filter_data.shape[1]):
            if filter_data[i][j] == 0:
                break
            if filter_data[i][j] not in size_list:
                filter_data[i] = np.append(np.delete(filter_data[i], j), 0)
                
    return filter_data

def list_defense_insert(input_data, prob, train_data):
    size_list, size_count, size_list_count = get_list(train_data)
    
    output_data = np.zeros((input_data.shape[0], input_data.shape[1] * 2))
    
    for i in range(input_data.shape[0]):
        index = 0
        for j in range(input_data.shape[1]):
            if input_data[i][j] == 0:
                break
            output_data[i][index] = input_data[i][j]
            index += 1
            if rd.randint(0, 100) < prob:
                insert = size_list_count[rd.randint(0, size_list_count.shape[0] - 1)]
                output_data[i][index] = insert
                index += 1
                
    return output_data

def list_defense_divide(input_data, prob, train_data):
    size_list, size_count, size_list_count = get_list(train_data)
    
    output_data = np.zeros((input_data.shape[0], input_data.shape[1] * 2))
    
    p_real = 0
    count_sum = 0
    count_extra = 0
    
    for i in range(input_data.shape[0]):
        index = 0
        for j in range(input_data.shape[1]):
            if input_data[i][j] == 0:
                break
            first = 0
            second = 0
            if rd.randint(0, 100) < prob:
                if input_data[i][j] > 200:
                    while True:
                        first = size_list_count[rd.randint(0, size_list_count.shape[0] - 1)]
                        second = input_data[i][j] - first
                        if first > 0 and second > 60:
                            break
                elif input_data[i][j] < -200:
                    while True:
                        first = size_list_count[rd.randint(0, size_list_count.shape[0] - 1)]
                        second = input_data[i][j] - first
                        if first < 0 and second < -60:
                            break
            if first != 0 and second != 0:
                output_data[i][index] = first
                index += 1
                output_data[i][index] = second
                index += 1
                count_extra += 1
            else:
                output_data[i][index] = input_data[i][j]
                index += 1

            count_sum += 1
            
    p_real = count_extra / count_sum * 100
    
    return output_data, p_real

def evaluate(model, test_data, test_index):
    test_data_cut = np.delete(test_data, range(500, test_data.shape[1]), axis=1)
    result = model.evaluate(test_data_cut.astype('float32')/1514, test_index)
    return result

def ow_evaluate(model, test_data, test_data_ow):
    
    test_data_cut = np.delete(test_data, range(500, test_data.shape[1]), axis=1)
    test_data_ow_cut = np.delete(test_data_ow, range(500, test_data_ow.shape[1]), axis=1)
    res = model.predict(test_data_cut.astype('float32')/1514)
    res_ow = model.predict(test_data_ow_cut.astype('float32')/1514)
    
    list_prob_threshold = 1.0 - 1 / np.logspace(0.01, 20, 100, endpoint=True)
    
    TPR = np.zeros(100)
    FPR = np.zeros(100)

    for j in range(100):
        prob_threshold = list_prob_threshold[j]

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(test_data.shape[0]):
            if max(res[i]) > prob_threshold:
                TP += 1
            else:
                FN += 1

        for i in range(test_data_ow.shape[0]):
            if max(res_ow[i]) > prob_threshold:
                FP += 1
            else:
                TN += 1

        TPR[j] = TP / (TP + FN)
        FPR[j] = FP / (FP + TN)
        
    return TPR, FPR

def ow_evaluate_tor(model, test_data, test_data_ow):
    
    test_data_cut = np.delete(test_data, range(5000, test_data.shape[1]), axis=1)
    test_data_ow_cut = np.delete(test_data_ow, range(5000, test_data_ow.shape[1]), axis=1)
    res = model.predict(test_data_cut.astype('float32'))
    res_ow = model.predict(test_data_ow_cut.astype('float32'))
    
    list_prob_threshold = 1.0 - 1 / np.logspace(0.01, 20, 100, endpoint=True)
    
    TPR = np.zeros(100)
    FPR = np.zeros(100)

    for j in range(100):
        prob_threshold = list_prob_threshold[j]

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(test_data.shape[0]):
            if max(res[i]) > prob_threshold:
                TP += 1
            else:
                FN += 1

        for i in range(test_data_ow.shape[0]):
            if max(res_ow[i]) > prob_threshold:
                FP += 1
            else:
                TN += 1

        TPR[j] = TP / (TP + FN)
        FPR[j] = FP / (FP + TN)
        
    return TPR, FPR
