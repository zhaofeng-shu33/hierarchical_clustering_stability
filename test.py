import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage
from ete3 import Tree

from utility import scipy_linkage_obj_to_ete3_tree
from utility import convert_bhc_tree_to_ete_tree


class TreeConvertTest(unittest.TestCase):
    def test_convert_linkage_obj_to_ete_tree(self):
        data = np.array([[0, 0], [1, 1], [5, 5]])
        Z = linkage(data)
        tree = scipy_linkage_obj_to_ete3_tree(Z)
        self.assertEqual(tree.write(format=9), '(2,(0,1));')
    def test_convert_bhc_tree_to_ete_tree(self):
        pch = {4:{2,3}, 3:{0, 1}, 2:{},1:{},0:{}}
        tree = convert_bhc_tree_to_ete_tree(pch)
        self.assertEqual(tree.write(format=9), '(2,(0,1));')


if __name__ == '__main__':
    unittest.main()