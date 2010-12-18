#! /usr/bin/python
"""
Some utilities to represent gating schemes as graphs.
"""

# As of yet this is a playground for testing graphs. On the long run,
# graphs might come in handy for several reasons (e.g. microscopic 
# reversibility and plotting). Could be moved to mechanism.py
# if useful. But probably we'll have to implement a graph class 
# ourselves that is tailored for gating schemes..
# TODO: Read Colquhoun04 and understand how to implement it.
# TODO: Find out how to represent a gating scheme using a directed graph
#       (if possible at all)
# TODO: Create pretty plots with arrows and such.

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
        G.add_edge(Rate.state1, Rate.state2, weight=w)
            
    state_labels = {}
    for State in mec.States:
        state_labels[State.no] = State.name

    rate_labels = {}
    for Rate in mec.Rates:
        rate_labels[(Rate.state1, Rate.state2)] = Rate.name
        
    return G, state_labels, rate_labels

if __name__=="__main__":
    import calculate_demo as demo

    demomec = demo.demoQ()
    G, state_labels, rate_labels = create_graph(demomec)
    pos = nx.spring_layout(G)

    T = nx.minimum_spanning_tree(G.to_undirected())
    
    nx.draw_networkx_nodes(G,pos,
                           nodelist=G.nodes(),
                           node_color='r',
                           node_size=500,
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1, alpha=0.5)
    nx.draw_networkx_edges(G, pos,
                           edgelist=T.edges(),
                           width=8,alpha=0.5,edge_color='b')
    nx.draw_networkx_labels(G, pos, state_labels,font_size=16)
    nx.draw_networkx_edge_labels(G, pos, rate_labels,font_size=16)
    plt.axis('off')

    plt.show()
