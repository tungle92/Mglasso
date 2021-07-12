import numpy as np
from mglasso.functions.merge import merge_proc
from mglasso.functions.conesta import conesta_rwrapper
from mglasso.functions.cost import cost
from mglasso.functions.utils import dist_beta
import mglasso.graph as gr

class hggm:
    def __init__(self,
                 maxit = 1000, 
                 silent = True, 
                 distance = "euclidean", 
                 solver = "conesta"):
        self.maxit = maxit 
        self.silent = silent 
        self.distance = distance 
        self.solver = solver 
        
        
    def estimator(self,
                  X,
                  fuse_thresh = 1e-3,
                  lambda1 = 0,
                  lambda2_start = 1e-4,
                  lambda2_factor = 1.5):
        self.X = X
        self.fuse_thresh = fuse_thresh
        self.lambda1 = lambda1 
        self.lambda2_start = lambda2_start 
        self.lambda2_factor = lambda2_factor
        
        ## Initialisations
        X = X.copy()
        X = np.array(X)
        p = X.shape[1]
        gains = np.zeros(p-1)
        merge = np.zeros((p-1, 2)) # matrix of merging clusters at each level
        old_level = level = 0
        labels = np.arange(p)   # vector of clusters labels
        clusters = np.arange(p)
        counts = np.ones(p)
        height = np.zeros(p-1)
        size = np.zeros(p-1)
        
        lambda2 = lambda2_start
        old_lambda2 = 0
        #old_lambda1 = lambda1
        Beta = conesta_rwrapper(X, lambda1, old_lambda2)
      
        old_costf = costf =  cost(Beta, X, lambda1, old_lambda2)
      
        t = 0 # index for the out list.
        #iteration = 0
        out = []
        out = np.append(out, {"Beta" : Beta, "clusters" : clusters.copy()})
        #names(out)[[1]] = "level0"
        prev = len(np.unique(clusters))
        ## End Initialisations
      
      
        ## Loop until all the variables merged
        while (len(np.unique(clusters)) > 1):
            #oldp = X.shape[1]
            #iteration = iteration + 1
        
            Beta = conesta_rwrapper(X, lambda1, lambda2)
        
            ## Update distance matrix
            diffs = dist_beta(Beta, distance = self.distance)
        
            ## Clustering starts here
            pairs_to_merge = np.concatenate(np.where(diffs<=fuse_thresh)).reshape((2,-1)).T
        
            if (pairs_to_merge.shape[0] != 0):
                
                gain_level = costf - old_costf
                out_mergeproc = merge_proc(pairs_to_merge, clusters, X, Beta, level, gain_level, gains, labels, merge, counts, size)
    
                X        = out_mergeproc['X']
                Beta     = out_mergeproc['Beta']
                clusters = out_mergeproc['clusters']
    
                level    = out_mergeproc['level']
                gains    = out_mergeproc['gains']
                merge    = out_mergeproc['merge']
                labels   = out_mergeproc['labels'] 
                counts   = out_mergeproc['counts']
                size   = out_mergeproc['size']
                height[old_level:level] = lambda2
                
                old_level = level
            ## Clustering ends here
        
            costf = cost(Beta, X, lambda1, lambda2)
            print("nclusters: ", len(np.unique(clusters)), "lambda2 :", lambda2, "cost: ", costf)
            
            gains[np.isnan(gains)] = old_costf - costf
            old_costf = costf
        
            if(len(np.unique(clusters)) != prev):
                out = np.append(out, {"Beta" : Beta, "clusters" : clusters.copy()})
                prev = len(np.unique(clusters))
                t = t + 1
        
            old_lambda2 = lambda2
        
            lambda2 = lambda2*lambda2_factor
            
        tree = np.c_[merge, size, height, np.cumsum(gains)]
      
        self.result = out
        self.tree = tree
      
        return self
    
    def dendrogram(self, option = 'lambda2'):
        if (option == 'lambda2'):
            gr.dendrogram(self.tree[:, [0,1,3,2]])
        else:
            gr.dendrogram(self.tree[:, [0,1,4,2]])

class igraph:
    def __init__(self, X, lambda1 = 0, lambda2 = 1e-3):
        self.X = X
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        self.Beta = conesta_rwrapper(self.X, self.lambda1, self.lambda2)
        
    def graph(self, option = "AND"):
        return gr.igraph(self.Beta, option)
        
    def adjacency_matrix(self, option = "AND"):
        return gr.adjacency_matrix(self.Beta, option)