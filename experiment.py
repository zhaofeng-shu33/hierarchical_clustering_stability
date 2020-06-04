import argparse
from data_loader import load_data
from scipy.cluster.hierarchy import linkage

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