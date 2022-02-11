import numpy as np
import csv

def read_data(file_name):
    with open(file_name) as cf:
        r = csv.reader(cf, delimiter=',')
        next(r) # skip header
        data = []
        for row in r:
            data.append([float(i) for i in row[1:]])
        return data

def load_data(limit_num=600):
    X_train = read_data('train.csv')
    X_test = read_data('test.csv')
    return (np.asarray(X_train[:limit_num]), np.asarray(X_test[:limit_num]))

if __name__ == '__main__':
    load_data()
