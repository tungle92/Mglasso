from igraph import Graph, plot
import numpy as np

def adjacency_matrix(Beta):
    Beta = np.array(Beta)
    boolean = (Beta != 0)
    return boolean.astype(int)

def igraph(Beta):
    Beta = np.array(Beta)
    p = Beta.shape[0]
    vertices = ["X"+str(i) for i in range(p)]
    edges = []
    
    for i in range(p-1):
        for j in range(i, p):
            if (Beta[i,j] != 0):
                edges.append((i,j))
                
    g = Graph(vertex_attrs={"label": vertices}, edges=edges)
    
    return plot(g)

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def dendrogram(Z):
    hierarchy.dendrogram(Z)          
    plt.show()
    