import argparse
import re
import pickle
import os
from scipy.cluster.hierarchy import linkage

try:
    from info_cluster import InfoCluster
    from ete3 import Tree
    from ete_robinson_foulds import robinson_foulds
    from core.brt import BayesianRoseTrees
    from utility import scipy_linkage_obj_to_ete3_tree
    from utility import create_model
    from utility import convert_bhc_tree_to_ete_tree
except ImportError:
    pass

from data_loader import load_data

from matplotlib import pyplot as plt

GRID = [100, 150, 200, 250, 300]

def brt(data):
    # Hyper-parameters (these values must be optimized!)
    g = 10
    scalling_factor = 0.01
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
        classical agglomerative hierarchical clustering
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
    if len(tree_1.children) == train_data.shape[0] \
        and len(tree_2.children) == train_data.shape[0]:
            print('Warning: trivial trees for n = %d' % train_data.shape[0])
    res = robinson_foulds(tree_1, tree_2)    
    metric_score = res['rf']
    return metric_score

def run_all(use_cache=False):
    if use_cache and os.path.exists('build/result_dic.pickle'):
        with open('build/result_dic.pickle', 'rb') as f:
            return pickle.load(f)
    X_train, X_test = load_data(600)
    result_dic = {}
    for method in [ic, ahc, brt]:
        method_name = re.search('function ([a-z]+) at', str(method)).group(1)
        result_dic[method_name] = []     
        for i in GRID:
            score = compute_score(method, X_train[:i,:], X_test[:i,:])
            result_dic[method_name].append(score)
    with open('build/result_dic.pickle', 'wb') as f:
        pickle.dump(result_dic, f)
    return result_dic

def plot_results(result_dic, pic_format='pdf'):
    marker_list = ['o', '+', '*']
    color_list = ['r', 'b', 'g']
    cnt = 0
    show_str =  {'ic': 'HPSP', 'ahc': '系统聚类', 'brt': '贝叶斯玫瑰树'}
    for k,v in result_dic.items():
        plt.plot(GRID, v, label=show_str[k], marker=marker_list[cnt],
            c=color_list[cnt], linewidth=3, markersize=12)
        cnt += 1
    plt.xlabel('样本数',fontsize=18, fontname='SimSun')
    plt.ylabel('距\n离',fontsize=18, fontname='SimSun', rotation=0, labelpad=10)
    L = plt.legend(fontsize='x-large', framealpha=0.0)
    plt.setp(L.texts, fontname='SimSun')
    plt.savefig('build/genes.' + pic_format, transparent=True)
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='single', choices=['single', 'all', 'plot'])
    parser.add_argument('--use_cache', default=False, nargs='?', const=True, type=bool)
    parser.add_argument('--format', default='pdf', choices=['pdf', 'svg'])
    parser.add_argument('--method', default='ic', choices=['ic', 'ahc', 'brt'])
    parser.add_argument('--limit_num', default=600, type=int, help="how many samples to use")
    args = parser.parse_args()
    if args.task == 'single':
        X_train, X_test = load_data(args.limit_num)
        if args.method == 'ic':
            score = compute_score(ic, X_train, X_test)
        elif args.method == 'ahc':
            score = compute_score(ahc, X_train, X_test)
        elif args.method == 'brt':
            score = compute_score(brt, X_train, X_test)
        print(score)
    elif args.task == 'all':
        result_dic = run_all(args.use_cache)
        plot_results(result_dic, args.format)
    elif args.task == 'plot':
        with open('build/result_dic.pickle', 'rb') as f:
            result_dic = pickle.load(f)
        plot_results(result_dic, args.format)