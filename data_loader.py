import numpy as np
def load_data():
    X_train = []
    X_test = []
    with open('build/supplemental_data.txt') as f:
        # abondon the first three lines
        f.readline()
        f.readline()
        f.readline()
        st = f.readline()
        value_list = st.rstrip().split('\t')[2:]
        value_list = [float(i) for i in value_list]
        X_train.append(value_list[:64])
        X_test.append(value_list[-25:])
    return (np.asarray(X_train), np.asarray(X_test))

if __name__ == '__main__':
    load_data()
