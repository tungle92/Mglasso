from igraph import Graph, plot
import numpy as np

def adjacency_metrix(Beta):
    Beta = np.array(Beta)
    return np.int(Beta != 0)

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

def dendrogram(Z, option = 'lambda2'):
    if (option == 'lambda2'):
        hierarchy.dendrogram(Z[:, 0:4])
    elif (option == 'gains'):
        hierarchy.dendrogram(Z[:, [0,1,4,3]])            
    plt.show()