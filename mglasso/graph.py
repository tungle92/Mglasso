from igraph import Graph, plot
import numpy as np

def adjacency_matrix(Beta, option = "AND"):
    Beta = np.array(Beta)
    if (option == "AND"):
        boolean = (Beta != 0) * (Beta.T != 0)
    elif (option == "OR"):
        boolean = (Beta != 0) + (Beta.T != 0)
    return boolean.astype(int)

def igraph(Beta, option = "AND"):
    Beta = np.array(Beta)
    p = Beta.shape[0]
    vertices = ["X"+str(i) for i in range(p)]
    edges = []
    
    for i in range(p-1):
        for j in range(i, p):
            if (option == "AND"):
                if ((Beta[i,j] * Beta[j,i]) != 0):
                    edges.append((i,j))
            elif (option == "OR"):
                if ((Beta[i,j] != 0) + (Beta[j,i] != 0)):
                    edges.append((i,j))
                    
    g = Graph(vertex_attrs={"label": vertices}, edges=edges)
    
    return plot(g)

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def dendrogram(Z):
    hierarchy.dendrogram(Z)          
    plt.show()
    