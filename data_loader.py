import numpy as np
def load_data(limit_num=600):
    X_train = []
    X_test = []
    with open('build/supplemental_data.txt') as f:
        # abondon the first three lines
        f.readline()
        f.readline()
        f.readline()
        st = f.readline()
        while st != '':
            value_list = st.rstrip().split('\t')[2:]
            value_list = [float(i) for i in value_list]
            if len(value_list) == 0:
                break
            X_train.append(value_list[:63])
            X_test.append(value_list[-25:])
            st = f.readline()
    return (np.asarray(X_train[:limit_num]), np.asarray(X_test[:limit_num]))

if __name__ == '__main__':
    load_data()
