'''
   tree format convert
'''
import numpy as np
from ete3 import Tree

from core.prior import NormalInverseWishart

def add_child(pch, tree_obj):
    tree_id = int(tree_obj.name)
    for i in pch[tree_id]:
        tree_child = tree_obj.add_child(name=str(i))
        add_child(pch, tree_child)

def convert_bhc_tree_to_ete_tree(pch):
    # found root
    root_id = max(pch.keys())
    root = Tree(name=str(root_id))
    add_child(pch, root)
    return root

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

def create_model(data, g, scalling_factor):
    degrees_of_freedom = data.shape[1] + 1
    data_mean = np.mean(data, axis=0)
    data_matrix_cov = np.cov(data.T)
    scatter_matrix = (data_matrix_cov / g).T

    return NormalInverseWishart(scatter_matrix,
                                scalling_factor,
                                degrees_of_freedom,
                                data_mean)