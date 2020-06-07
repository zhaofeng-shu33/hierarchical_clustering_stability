'''
   tree format convert
'''
import numpy as np
from ete3 import Tree
import six

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

class TreeError(Exception):
    """
    A problem occurred during a TreeNode operation
    """
    def __init__(self, value=''):
        self.value = value
    def __str__(self):
        return repr(self.value)

def robinson_foulds(t1, t2, attr_t1="name", attr_t2="name",
                    unrooted_trees=False, expand_polytomies=False,
                    polytomy_size_limit=5, skip_large_polytomies=False,
                    correct_by_polytomy_size=False, min_support_t1=0.0,
                    min_support_t2=0.0):
    """
    .. versionadded: 2.2

    Returns the Robinson-Foulds symmetric distance between current
    tree and a different tree instance.

    :param t2: reference tree

    :param name attr_t1: Compare trees using a custom node
                            attribute as a node name.

    :param name attr_t2: Compare trees using a custom node
                            attribute as a node name in target tree.

    :param False attr_t2: If True, consider trees as unrooted.

    :param False expand_polytomies: If True, all polytomies in the reference
        and target tree will be expanded into all possible binary
        trees. Robinson-foulds distance will be calculated between all
        tree combinations and the minimum value will be returned.
        See also, :func:`NodeTree.expand_polytomy`.

    :returns: (rf, rf_max, common_attrs, names, edges_t1, edges_t2,  discarded_edges_t1, discarded_edges_t2)

    """
    ref_t = t1
    target_t = t2

    if expand_polytomies and correct_by_polytomy_size:
        raise TreeError("expand_polytomies and correct_by_polytomy_size are mutually exclusive.")

    if expand_polytomies and unrooted_trees:
        raise TreeError("expand_polytomies and unrooted_trees arguments cannot be enabled at the same time")


    attrs_t1 = set([getattr(n, attr_t1) for n in ref_t.iter_leaves() if hasattr(n, attr_t1)])
    attrs_t2 = set([getattr(n, attr_t2) for n in target_t.iter_leaves() if hasattr(n, attr_t2)])
    common_attrs = attrs_t1 & attrs_t2
    # release mem
    attrs_t1, attrs_t2 = None, None

    # Check for duplicated items (is it necessary? can we optimize? what's the impact in performance?')
    size1 = len([True for n in ref_t.iter_leaves() if getattr(n, attr_t1, None) in common_attrs])
    size2 = len([True for n in target_t.iter_leaves() if getattr(n, attr_t2, None) in common_attrs])
    if size1 > len(common_attrs):
        raise TreeError('Duplicated items found in source tree')
    if size2 > len(common_attrs):
        raise TreeError('Duplicated items found in reference tree')

    if expand_polytomies:
        ref_trees = [Tree(nw) for nw in
                        ref_t.expand_polytomies(map_attr=attr_t1,
                                                polytomy_size_limit=polytomy_size_limit,
                                                skip_large_polytomies=skip_large_polytomies)]
        target_trees = [Tree(nw) for nw in
                        target_t.expand_polytomies(map_attr=attr_t2,
                                                    polytomy_size_limit=polytomy_size_limit,
                                                    skip_large_polytomies=skip_large_polytomies)]
        attr_t1, attr_t2 = "name", "name"
    else:
        ref_trees = [ref_t]
        target_trees = [target_t]

    polytomy_correction = 0
    if correct_by_polytomy_size:
        corr1 = sum([0]+[len(n.children) - 2 for n in ref_t.traverse() if len(n.children) > 2])
        corr2 = sum([0]+[len(n.children) - 2 for n in target_t.traverse() if len(n.children) > 2])
        if corr1 and corr2:
            raise TreeError("Both trees contain polytomies! Try expand_polytomies=True instead")
        else:
            polytomy_correction = max([corr1, corr2])

    min_comparison = None
    for t1 in ref_trees:
        t1_content = t1.get_cached_content()
        t1_leaves = t1_content[t1]
        if unrooted_trees:
            edges1 = set([
                    tuple(sorted([tuple(sorted([getattr(n, attr_t1) for n in content if hasattr(n, attr_t1) and getattr(n, attr_t1) in common_attrs])),
                                    tuple(sorted([getattr(n, attr_t1) for n in t1_leaves-content if hasattr(n, attr_t1) and getattr(n, attr_t1) in common_attrs]))]))
                    for content in six.itervalues(t1_content)])
            edges1.discard(((),()))
        else:
            edges1 = set([
                    tuple(sorted([getattr(n, attr_t1) for n in content if hasattr(n, attr_t1) and getattr(n, attr_t1) in common_attrs]))
                    for content in six.itervalues(t1_content)])
            edges1.discard(())

        if min_support_t1:
            support_t1 = dict([
                    (tuple(sorted([getattr(n, attr_t1) for n in content if hasattr(n, attr_t1) and getattr(n, attr_t1) in common_attrs])), branch.support)
                    for branch, content in six.iteritems(t1_content)])

        for t2 in target_trees:
            t2_content = t2.get_cached_content()
            t2_leaves = t2_content[t2]
            if unrooted_trees:
                edges2 = set([
                        tuple(sorted([
                                    tuple(sorted([getattr(n, attr_t2) for n in content if hasattr(n, attr_t2) and getattr(n, attr_t2) in common_attrs])),
                                    tuple(sorted([getattr(n, attr_t2) for n in t2_leaves-content if hasattr(n, attr_t2) and getattr(n, attr_t2) in common_attrs]))]))
                        for content in six.itervalues(t2_content)])
                edges2.discard(((),()))
            else:
                edges2 = set([
                        tuple(sorted([getattr(n, attr_t2) for n in content if hasattr(n, attr_t2) and getattr(n, attr_t2) in common_attrs]))
                        for content in six.itervalues(t2_content)])
                edges2.discard(())

            if min_support_t2:
                support_t2 = dict([
                    (tuple(sorted(([getattr(n, attr_t2) for n in content if hasattr(n, attr_t2) and getattr(n, attr_t2) in common_attrs]))), branch.support)
                    for branch, content in six.iteritems(t2_content)])


            # if a support value is passed as a constraint, discard lowly supported branches from the analysis
            discard_t1, discard_t2 = set(), set()
            if min_support_t1 and unrooted_trees:
                discard_t1 = set([p for p in edges1 if support_t1.get(p[0], support_t1.get(p[1], 999999999)) < min_support_t1])
            elif min_support_t1:
                discard_t1 = set([p for p in edges1 if support_t1[p] < min_support_t1])

            if min_support_t2 and unrooted_trees:
                discard_t2 = set([p for p in edges2 if support_t2.get(p[0], support_t2.get(p[1], 999999999)) < min_support_t2])
            elif min_support_t2:
                discard_t2 = set([p for p in edges2 if support_t2[p] < min_support_t2])


            #rf = len(edges1 ^ edges2) - (len(discard_t1) + len(discard_t2)) - polytomy_correction # poly_corr is 0 if the flag is not enabled
            #rf = len((edges1-discard_t1) ^ (edges2-discard_t2)) - polytomy_correction

            # the two root edges are never counted here, as they are always
            # present in both trees because of the common attr filters
            rf = len(((edges1 ^ edges2) - discard_t2) - discard_t1) - polytomy_correction

            if unrooted_trees:
                # thought this may work, but it does not, still I don't see why
                #max_parts = (len(common_attrs)*2) - 6 - len(discard_t1) - len(discard_t2)
                max_parts = (len([p for p in edges1 - discard_t1 if len(p[0])>1 and len(p[1])>1]) +
                                len([p for p in edges2 - discard_t2 if len(p[0])>1 and len(p[1])>1]))
            else:
                # thought this may work, but it does not, still I don't see why
                #max_parts = (len(common_attrs)*2) - 4 - len(discard_t1) - len(discard_t2)

                # Otherwise we need to count the actual number of valid
                # partitions in each tree -2 is to avoid counting the root
                # partition of the two trees (only needed in rooted trees)
                max_parts = (len([p for p in edges1 - discard_t1 if len(p)>1]) +
                                len([p for p in edges2 - discard_t2 if len(p)>1])) - 2

                # print max_parts

            if not min_comparison or min_comparison[0] > rf:
                min_comparison = [rf, max_parts, common_attrs, edges1, edges2, discard_t1, discard_t2]
        result = {}
        result["norm_rf"] = rf / max_parts
        return result