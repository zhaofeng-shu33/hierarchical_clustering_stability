import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage
from ete3 import Tree

from utility import scipy_linkage_obj_to_ete3_tree
from utility import convert_bhc_tree_to_ete_tree
from utility import robinson_foulds

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

class TreeCompare(unittest.TestCase):
    def test_tree_compare(self):
        t_1 = Tree('((a,b),c);')
        t_2 = Tree('((a,c),b);')
        result = robinson_foulds(t_1, t_2)
        self.assertAlmostEqual(result['norm_rf'], 1.0)
    def test_non_binary_tree_compare(self):
        t_1 = Tree('((a,b,d),c);')
        t_2 = Tree('(((a,b),d),c);')
        result = robinson_foulds(t_1, t_2)
        self.assertAlmostEqual(result['norm_rf'], 1/3.0)

if __name__ == '__main__':
    unittest.main()