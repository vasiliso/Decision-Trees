import numpy as np
from numpy import log2 as log
import pandas as pd
import pprint
import copy
import pydot
eps = np.finfo(float).eps


def open_file(filename):
    # opens desired file and formats columns
    # it also finds continuous attributes and splits them into bins,
    # randomizes data and split it randomly into training (80%), testing (10%), and validation (10%)

    if filename == 'mushroom.data':
        df = pd.read_csv(filename)
        labels = df['E/P']
        df.drop(labels='E/P', axis=1, inplace=True)
        df.insert(df.columns.size, 'E/P', labels)
    if filename == 'crx.data':
        df = pd.read_csv(filename)
        cont_attr = get_continuous(df)
        continuous(df, cont_attr, 4)
    if filename == 'horse-colic.data':
        df = pd.read_csv(filename, delim_whitespace=True)

        cont_attr = get_continuous(df)
        continuous(df, cont_attr, 4)

        labels = df['SUL']
        df.drop(labels='SUL', axis=1, inplace=True)
        df.insert(df.columns.size, 'SUL', labels)
    if filename == 'hepatitis.data':
        df = pd.read_csv(filename)
        cont_attr = get_continuous(df)
        continuous(df, cont_attr, 4)

        labels = df['CLASS']
        df.drop(labels='CLASS', axis=1, inplace=True)
        df.insert(df.columns.size, 'CLASS', labels)
    if filename == 'haberman.data':
        df = pd.read_csv(filename)
        cont_attr = get_continuous(df)
        continuous(df, cont_attr, 4)
    print(df)
    df = df.sample(frac=1).reset_index(drop=True)
    training = df.sample(frac=0.6)
    rest = df.drop(training.index).reset_index(drop=True)
    training = training.reset_index(drop=True)
    testing = rest.sample(frac=0.5)
    validation = rest.drop(testing.index)
    testing = testing.reset_index(drop=True)
    validation = validation.reset_index(drop=True)

    return df, training, testing, validation


def numbers(x):
    # gets all values that are numbers for binning purposes
    try:
        new_val = float(x)
        return new_val
    except:
        return np.nan


def fix(x, maximum, minimum, bins):
    # splits an attribute into bins if numeric
    try:
        new_val = float(x)
        div = (maximum + minimum)/bins

        for i in range(1, bins):
            if new_val <= minimum + (div * i):
                return int(i)
        return int(i+1)
    except:
        return x


def continuous(df, attributes, bins):
    # takes a dataframe and a list of attributes and converts those attributes into discrete bins
    for attribute in attributes:
        num_arr = df.apply(lambda x: numbers(x[attribute]), axis=1)
        df[attribute] = df.apply(lambda x: fix(
            x[attribute], np.nanmax(num_arr), np.nanmin(num_arr), bins), axis=1)


def get_continuous(df):
    # returns a list of all attributes with more than 4 discrete values, in order to make the data easier to parse
    continuous = []
    columns = list(df.keys())
    del columns[-1]
    for column in columns:
        if len(df[column].unique()) > 4:
            continuous.append(column)
    return continuous


def decision_tree(df, used=None):
    # Recursively builds a decision tree from a given dataframe until entropy(df) is 0 or all attributes used

    truth_col = df.keys()[-1]

    if used is None:
        used = []
    # get best attribute based on information gain
    attr = find_best(df[df.columns.difference(used, sort=False)])
    used.append(attr)
    attributes = list(df)

    values = df[attr].unique()

    tree = {}
    tree[attr] = {}

    # loop through unique values of attr and create a branch for each value. If entropy of sorting on value is 0, the branch is a lead
    for value in values:
        sub = df[df[attr] == value].reset_index(drop=True)
        if entropy(sub) == 0 or (len(attributes) - 1) == len(np.unique(used)):
            unique, counts = np.unique(sub[truth_col], return_counts=True)
            tree[attr][value] = unique[np.argmax(counts)]
        else:
            tree[attr][value] = decision_tree(sub, used=used)
    return tree


def entropy(df):
    # returns the entropy of a certain set

    truth_col = df.keys()[-1]
    entropy = 0
    labels = df[truth_col].unique()
    for label in labels:
        frac = df[truth_col].value_counts()[label] / len(df[truth_col])
        entropy += -frac * log(frac)
    return entropy


def gain(df, attr):
    # returns the information gain of sorting on a certain attribute

    truth_col = df.keys()[-1]
    labels = df[truth_col].unique()
    values = df[attr].unique()
    a_entropy = 0
    for value in values:
        frac = df[attr].value_counts()[value] / (len(df[attr]) + eps)
        entropy2 = entropy(df[df[attr] == value]
                           )
        a_entropy += frac*entropy2
    gain = entropy(df) - a_entropy
    return gain


def find_best(df):
    # returns the attribute with the highest information gain

    ig = []
    for key in df.keys()[:-1]:
        ig.append(gain(df, key))
    return df.keys()[:-1][np.argmax(ig)]


def test(hypothesis, tree):
    # test a hypothesis by traversing the tree and comparing the leaf of the tree to the label of the hypothesis

    if type(tree) is dict:
        try:
            return test(hypothesis, tree[list(tree)[0]][hypothesis[list(tree)[0]]])
        except:
            print("Label found in test set that does not exist in the decision tree.")
            return 0
    else:
        if(hypothesis[-1] == tree):
            return 1
        else:
            return 0


def test_all(df, tree):
    # tests all hypotheses in the set agains the tree

    count = 0
    for i in range(len(df[df.keys()[-1]])):
        count += test(df.loc[i, :], tree)
    return count / len(df[df.keys()[-1]])


def pruning(df, tree):
    # prunes a decision tree by finding all nodes whose children are all leaves, then checking to see if simplifying the node to a leaf gives a better accuracy,
    # and removes the node that gives the greatest gain in accuracy
    # essentially tries to find the shortest tree without loss of accuracy

    nodes = []
    # test on current tree for benchmark
    default_accuracy = test_all(df, tree)
    max_accuracy = 0
    # save current tree as best possible tree (that we know of)
    best_tree = tree
    # get all nodes with all leaf children
    get_nodes(tree, nodes)

    # test if simplifying each node produces a better or equal accuracy than the benchmark
    for i in range(len(nodes)):
        tcopy = copy.deepcopy(tree)
        tcopy = simplify_node(tcopy, nodes[i])
        accuracy = test_all(df, tcopy)
        if accuracy >= max_accuracy:
            best_tree = copy.deepcopy(tcopy)
            max_accuracy = accuracy
    # recursively prune the tree if the accuracy of the pruned tree is greater or equal to the benchmark, otherwise return the current tree
    if (max_accuracy >= default_accuracy):
        return pruning(df, best_tree)
    else:
        return tree


def get_nodes(tree, nodes, node=None):
    # traverse the tree and get all nodes that have all leaf children

    if node is None:
        node = []
    for key in tree.keys():
        if type(tree[key]) is dict:
            if is_not_grandparent(tree[key]):
                node.append(key)
                nodes.append(node)
                return
            else:
                ncopy = copy.deepcopy(node)
                ncopy.append(key)
                get_nodes(tree[key], nodes, ncopy)


def is_not_grandparent(tree):
    # check if node's children are all leaves

    leaves = 0
    for key in tree.keys():
        if type(tree[key]) is not dict:
            leaves += 1
    if leaves == len(tree.keys()):
        return True
    return False


def simplify_node(tree, node):
    # return a tree with the given node simplified to the label with the most occurences

    newtree = tree
    for path in node:
        parent = newtree
        newtree = newtree[path]
    l = list(newtree[key] for key in newtree.keys())

    unique, counts = np.unique(l, return_counts=True)
    newtree.clear()
    parent.update({path: unique[np.argmax(counts)]})
    return tree


def visit(node, parent=None):
    # this function traverses the tree in dict format and returns a graphviz graph of the tree for visualization

    for k, v in node.items():
        if isinstance(v, dict):
            # We start with the root node whose parent is None
            # we don't want to graph the None node
            if parent:
                draw(str(parent), str(parent) + '_'+str(k))
                visit(v, str(parent) + '_'+str(k))
            else:
                visit(v, str(k))
        else:
            draw(str(parent), str(parent)+'_'+str(k))
            # drawing the label using a distinct name
            draw(str(parent)+'_'+str(k), str(parent)+'_'+str(k)+'_'+str(v))


def draw(parent_name, child_name):
    # this function creates a graphviz edge between two nodes

    edge = pydot.Edge(parent_name, child_name)
    graph.add_edge(edge)


# run the decision tree algorithm on a dataset, prune, test, and visualize
df, training, testing, validation = open_file('hepatitis.data')
tree = decision_tree(training)
pruned_tree = pruning(validation, tree)
a = test_all(training, tree)
b = test_all(training, pruned_tree)
c = test_all(testing, tree)
d = test_all(testing, pruned_tree)
e = test_all(validation, tree)
f = test_all(validation, pruned_tree)


print('\n\nOriginal Tree\n\n')
graph = pydot.Dot(graph_type='graph')
visit(tree)
graph.write_png('original_tree.png')
pprint.pprint(tree)

print('\n\nPruned Tree\n\n')
graph = pydot.Dot(graph_type='graph')
visit(pruned_tree)
graph.write_png('pruned_tree.png')
pprint.pprint(pruned_tree)

print("Testing on training data:")
print('Original Accuracy:\t', a)
print('Pruned Accuracy:\t', b)
print('')
print("Testing on test data:")
print('Original Accuracy:\t', c)
print('Pruned Accuracy:\t', d)
print('')
print("Testing on validation data:")
print('Original Accuracy:\t', e)
print('Pruned Accuracy:\t', f)
