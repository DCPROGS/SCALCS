"""Remis' sandbox"""

import sys
import numpy as np
from dcpyps import samples
from dcpyps import mechanism

if __name__ == "__main__":

    mec1 = samples.CH82()
    sys.stdout.write('%s' % mec1)
    mec1.update_mr()
    sys.stdout.write('\n\nmr imposed:')
    sys.stdout.write('%s' % mec1)

    gr1 = mechanism.Graph(mec1.Rates)
    print '\n\n\n    CH82:\n', gr1.graph
    nodes, edges = gr1.nodes_edges(gr1.graph)
    print 'nodes:\n', nodes
    print 'edges:\n', edges
    cycles1 = gr1.find_cycles(gr1.graph)
    print 'cycles:\n', cycles1

#    DEG = gr1.degree(gr1.graph)
#    print '\nDegree matrix = '
#    print DEG
#    DEGlist = gr1.degree_list(gr1.graph)
#    print '\nDegree list = '
#    print DEGlist
#    ADJ =gr1.adjacency(gr1.graph)
#    print '\nAdjacency matrix = '
#    print ADJ
#    print 'eigenvalues of Adjacency matrix ='
#    eigvals, eigvecs = nlin.eig(ADJ)
#    print eigvals
#    print '\nAdjacency matrix square = '
#    print np.dot(ADJ, ADJ)
#    print '\nAdjacency matrix cube = '
#    print np.dot(np.dot(ADJ, ADJ), ADJ)
#    print '\nAdjacency matrix forth power = '
#    A4 = np.dot(np.dot(np.dot(ADJ, ADJ), ADJ), ADJ)
#    print A4
#    tr = np.trace(A4)
#    print 'trace of A4 = ', tr
#    INC = gr1.incidence(gr1.graph)
#    print '\nIncidence matrix = '
#    print INC
#    LAP = gr1.laplacian(gr1.graph)
#    print '\nLaplacian matrix = '
#    print LAP

    mec2 = samples.six_cycles_mec()
    gr2 = mechanism.Graph(mec2.Rates)
    print '\n\n    Six cycle graph:\n', gr2.graph
    cycles2 = gr2.find_cycles(gr2.graph)
    print 'cycles :\n', cycles2

