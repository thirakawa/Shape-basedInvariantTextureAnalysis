# -*- coding:utf-8 -*-

import numpy as np
import networkx as nx
from TreeOfShapes import *

# Graph ================================================
# construct hierarchy using NetworkX
# adding attribute
def createAttributeGraph(tree1, im):
    num_x = im.shape[1]
    num_y = im.shape[0]
    g = nx.Graph()
    nodePixMap = np.zeros((im.shape[0]-2, im.shape[1]-2), dtype=int)
    g.graph['leaves'] = []

    # creating tree
    for i in tree1.iterateFromRootToLeaves(False):
        # print i, "->", tree1[i]
        # not root node
        if tree1.data[i] != -1:
            g.add_edge(i, tree1.data[i])
            g.node[i]['grayLevel'] = tree1.levels[i]
            g.node[i]['area'] = tree1.area[i]
            if 'children' in g.node[tree1.data[i]]:    # exist
                g.node[tree1.data[i]]['children'].append(i)
            else:                                 # not exist
                g.node[tree1.data[i]]['children'] = [i]

            if 'parent' in g.node[i]:
                g.node[i]['parent'].append(tree1.data[i])
            else:
                g.node[i]['parent'] = [tree1.data[i]]
        # root node
        else:
            g.add_node(i)
            g.node[i]['parent'] = -1
            g.node[i]['grayLevel'] = tree1.levels[i]
            g.node[i]['area'] = tree1.area[i]

    # adding node attribution
    for i in tree1.iterateFromRootToLeaves(False):
        # in case of the node has children attribute (i.e. not leaf node)
        if 'children' in g.node[i]:
            g.node[i]['leaf'] = False
            g.node[i]['pixels'] = []
        # no children attribute (i.e. leaf node)
        else:
            g.node[i]['leaf'] = True
            g.graph['leaves'].append(i)
            g.node[i]['pixels'] = []

    # adding leaf =============
    for y in range(3,num_y*2-2, 2):
        for x in range(3, num_x*2-2, 2):
            # print (2*num_x+1)*y + x, "->", tree1.data[(2*num_x+1)*y + x]
            g.node[ tree1.data[(2*num_x+1)*y + x] ]['pixels'].append([(x-1)/2-1, (y-1)/2-1])
            nodePixMap[(y-1)/2-1, (x-1)/2-1] = int(tree1.data[(2*num_x+1)*y + x])

    # adding number of nodes ==
    g.graph['n_nodes'] = ( max(g.nodes()) - min(g.nodes()) + 1 )
    g.graph['orderNodes'] = tree1.iterateFromRootToLeaves(False)

    return g, nodePixMap
# ======================================================
