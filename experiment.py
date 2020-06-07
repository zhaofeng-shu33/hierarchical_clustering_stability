import argparse

from scipy.cluster.hierarchy import linkage
from info_cluster import InfoCluster
from ete3 import Tree

from core.brt import BayesianRoseTrees
from data_loader import load_data
from utility import scipy_linkage_obj_to_ete3_tree
from utility import create_model
from utility import convert_bhc_tree_to_ete_tree


def brt(data):
    # Hyper-parameters (these values must be optimized!)
    g = 10
    scalling_factor = 0.001
    alpha = 0.5

    model = create_model(data, g, scalling_factor)

    brt_obj = BayesianRoseTrees(data,
                            model,
                            alpha,
                            cut_allowed=False)

    brt_obj.build()
    tree = convert_bhc_tree_to_ete_tree(brt_obj.pch)
    return tree


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
    parser.add_argument('--method', default='ic', choices=['ic', 'ahc', 'brt'])
    parser.add_argument('--limit_num', default=600, type=int, help="how many samples to use")
    args = parser.parse_args()
    X_train, X_test = load_data(args.limit_num)
    if args.method == 'ic':
        score = compute_score(ic, X_train, X_test)
    elif args.method == 'ahc':
        score = compute_score(ahc, X_train, X_test)
    elif args.method == 'brt':
        score = compute_score(brt, X_train, X_test)
    print(score)
