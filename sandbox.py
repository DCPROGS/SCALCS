"""Playground for testing graphs.
On the long run, graphs might come in handy for several reasons
(e.g. microscopic reversibility and plotting). Could be moved to mechanism.py
if useful. But probably we'll have to implement a graph class
ourselves that is tailored for gating schemes..
TODO: Read Colquhoun04 and understand how to implement it.
TODO: Find out how to represent a gating scheme using a directed graph
      (if possible at all)
TODO: Create pretty plots with arrows and such.
"""

import sys
import numpy as np
from numpy import linalg as nplin
from dcpyps import samples
from dcpyps import mechanism

try:
    import networkx as nx
except:
    raise ImportError("networkx module is missing")

try:
    import matplotlib.pyplot as plt
except:
    raise ImportError("matplotlib is missing")

def create_graph(mec):
    G = nx.MultiDiGraph()

    # add vertices
    G.add_nodes_from(range(1, len(mec.States)+1))

    # add edges
    for Rate in mec.Rates:
        if Rate.mr:
            w = len(mec.States)
        else:
            w = 1
        G.add_edge(Rate.State1, Rate.State2, weight=w)

    state_labels = {}
    for State in mec.States:
        state_labels[State.no] = State.name

    rate_labels = {}
    for Rate in mec.Rates:
        rate_labels[(Rate.State1, Rate.State2)] = Rate.name

    return G, state_labels, rate_labels


if __name__ == "__main__":

#    demomec = samples.CH82()
#    G, state_labels, rate_labels = create_graph(demomec)
#    pos = nx.spring_layout(G)
#
#    T = nx.minimum_spanning_tree(G.to_undirected())
#
#    nx.draw_networkx_nodes(G,pos,
#                           nodelist=G.nodes(),
#                           node_color='r',
#                           node_size=500,
#                           alpha=0.8)
#    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
#    nx.draw_networkx_edges(G, pos,
#                           edgelist=T.edges(),
#                           width=8,alpha=0.5,edge_color='b')
#    nx.draw_networkx_labels(G, pos, state_labels,font_size=16)
#    nx.draw_networkx_edge_labels(G, pos, rate_labels,font_size=16)
#    plt.axis('off')
#
#    plt.show()


    mec1 = samples.CH82()
    #mec1 = samples.six_cycles_mec()
#    sys.stdout.write('%s' % mec1)
#    mec1.update_mr()
#    sys.stdout.write('\n\nmr imposed:')
#    sys.stdout.write('%s' % mec1)
#
    gr1 = mechanism.Graph(mec1.Rates)
    print '\n\n\n    CH82:\n', gr1.graph
    print 'mr=', gr1.mr
    print 'fixed=', gr1.fixed
    print 'constrained=', gr1.constrained
    nodes, edges = gr1.nodes_edges(gr1.graph)
    print 'nodes:\n', nodes
    print 'edges:\n', edges
    cycles1 = gr1.find_cycles(gr1.graph)
    print 'cycles:\n', cycles1

    DEG = gr1.degree(gr1.graph)
    print '\nDegree matrix = '
    print DEG
    DEGlist = gr1.degree_list(gr1.graph)
    print '\nDegree list = '
    print DEGlist
    ADJ =gr1.adjacency(gr1.graph)
    print '\nAdjacency matrix = '
    print ADJ
    print 'eigenvalues of Adjacency matrix ='
    eigvals, eigvecs = nplin.eig(ADJ)
    print eigvals
    print '\nAdjacency matrix square = '
    print np.dot(ADJ, ADJ)
    print '\nAdjacency matrix cube = '
    print np.dot(np.dot(ADJ, ADJ), ADJ)
    print '\nAdjacency matrix forth power = '
    A4 = np.dot(np.dot(np.dot(ADJ, ADJ), ADJ), ADJ)
    print A4
    tr = np.trace(A4)
    print 'trace of A4 = ', tr
    INC = gr1.incidence(gr1.graph)
    print '\nIncidence matrix = '
    print INC
    LAP = gr1.laplacian(gr1.graph)
    print '\nLaplacian matrix = '
    print LAP

#    mec2 = samples.six_cycles_mec()
#    gr2 = mechanism.Graph(mec2.Rates)
#    print '\n\n    Six cycle graph:\n', gr2.graph
#    cycles2 = gr2.find_cycles(gr2.graph)
#    print 'cycles :\n', cycles2

