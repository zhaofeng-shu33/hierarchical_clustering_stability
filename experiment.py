import argparse
from data_loader import load_data
from scipy.cluster.hierarchy import linkage
from info_cluster import InfoCluster
from ete3 import Tree

def process_record(n, current_root_obj, Z, i):
    left_child_id = int(Z[i,0])
    right_child_id = int(Z[i,1])
    left_root = current_root_obj.add_child(name=str(left_child_id))
    if left_child_id >= n:
        left_root_i = left_child_id - n
        process_record(n, left_root, Z, left_root_i)
    right_root = current_root_obj.add_child(name=str(right_child_id))
    if right_child_id >= n:
        right_root_i = right_child_id - n
        process_record(n, right_root, Z, right_root_i)

def scipy_linkage_obj_to_ete3_tree(Z):
    n = Z.shape[0] + 1
    root = Tree(name=str(2 * n - 2))
    i = n - 2
    process_record(n, root, Z, i)
    return root

def ahc(data, method='average'):
    '''
        classical agglomerative hierarichical clustering
        data: num_of_samples x num_of_attributes
        return the ete3 tree representation of the clustering tree
    '''
    Z = linkage(data, method)
    tree = scipy_linkage_obj_to_ete3_tree(Z)
    return tree

def ic(data, metric='rbf'):
    ic = InfoCluster(affinity=metric)
    ic.fit(data)
    return ic.tree

def compute_score(alg, train_data, test_data):
    tree_1 = alg(train_data)
    tree_2 = alg(test_data)
    res = tree_1.compare(tree_2, unrooted=True)
    metric_score = res['norm_rf']
    return metric_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='ic', choices=['ic', 'ahc'])
    args = parser.parse_args()
    X_train, X_test = load_data()
    if args.method == 'ic':
        score = compute_score(ic, X_train, X_test)
    elif args.method == 'ahc':
        score = compute_score(ahc, X_train, X_test)
    print(score)
