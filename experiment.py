import argparse
from data_loader import load_data
from scipy.cluster.hierarchy import linkage
from dendrogram_purity import dendrogram_purity
from info_cluster import InfoCluster

def scipy_linkage_obj_to_ete3_tree(Z):
    return None

def ahc(data, method='average'):
    '''
        classical agglomerative hierarichical clustering
        data: num_of_samples x num_of_attributes
        return the ete3 tree representation of the clustering tree
    '''
    Z = linkage(data, method)
    tree = scipy_linkage_obj_to_ete3_tree(Z)

def ic(data, metric='rbf'):
    ic = InfoCluster(affinity=metric)
    ic.fit(data)
    return ic.tree

def compute_score(alg, train_data, test_data):
    tree_1 = alg(train_data)
    tree_2 = alg(test_data)
    metric_score = dendrogram_purity(tree_1, tree_2)
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
