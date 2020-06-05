import unittest

import numpy as np
from scipy.cluster.hierarchy import linkage

from utility import scipy_linkage_obj_to_ete3_tree

class TreeConvertTest(unittest.TestCase):
    def test_convert_linkage_obj_to_ete_tree(self):
        data = np.array([[0, 0], [1, 1], [5, 5]])
        Z = linkage(data)
        tree = scipy_linkage_obj_to_ete3_tree(Z)
        self.assertEqual(tree.write(format=9), '(2,(0,1));')

if __name__ == '__main__':
    unittest.main()