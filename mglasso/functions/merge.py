import numpy as np
from mglasso.functions.utils import plus_lines

# Merge X
def mergeX(X, pair_to_merge, clusters):
    X = X.copy()
    i = np.min(pair_to_merge)
    j = np.max(pair_to_merge)
  
    ni = np.sum(clusters == i)
    nj = np.sum(clusters == j)
  
    X[:,i] = (ni*X[:,i] + nj*X[:,j])/(ni + nj)
    X = np.delete(X, j, 1)
    
    return X

#' Merge Beta
#' Different types of merging and their effect
def merge_beta(Beta, pair_to_merge, clusters):
    Beta = Beta.copy()
    i = min(pair_to_merge)
    j = max(pair_to_merge)
  
    ni = np.sum(clusters == i)
    nj = np.sum(clusters == j)
  
    Beta[i,:] = plus_lines(i, j, Beta, ni, nj) / (ni + nj)
    Beta = np.delete(np.delete(Beta, j, 1), j, 0)
  
    return Beta

#' Merge labels
def merge_labels(merged_pair, labels):
    i = min(merged_pair)
    j = max(merged_pair)
    labels[i] = max(labels)+1
    labels = np.delete(labels, j , None)
    return labels

#' Merge counts
def merge_counts(merged_pair, counts):
    i = min(merged_pair)
    j = max(merged_pair)
    counts[i] = counts[i]+counts[j]
    counts = np.delete(counts, j , None)
    return counts

#' Merge procedure
def merge_proc(pairs_to_merge, 
               clusters, 
               X,
               Beta,
               level,
               gain_level,
               gains,
               labels,
               merge,
               counts,
               size):
    X = X.copy()
    Beta = Beta.copy()
    for l in range(pairs_to_merge.shape[0]):
        pair_to_merge = pairs_to_merge[l,:]
    
        # can also take the 1st element cause it's always the min for a upper-triangular matrix
        i = np.min(pair_to_merge)
        j = np.max(pair_to_merge)
    
        if(i != j):
            
            # update size of clusters
            size[level] = counts[i] + counts[j]
            counts = merge_counts(pair_to_merge, counts)
      
            # merge lines/cols in Beta and X
            Beta = merge_beta(Beta, pair_to_merge, clusters)
            X = mergeX(X, pair_to_merge, clusters)
      
            # update dendrogram
            merge[level, :] = [labels[i], labels[j]]
            labels = merge_labels(pair_to_merge, labels)
            gains[level] = 0 if l>0 else np.nan
      
            # merge clusters
            clusters[clusters == j] = i 
            clusters[clusters > j] = clusters[clusters > j] - 1
      
            # update the rest of the table with the new clusters
            pairs_to_merge[pairs_to_merge == j] = i
            pairs_to_merge[pairs_to_merge > j] = pairs_to_merge[pairs_to_merge > j] - 1
            
            level = level + 1
            
    out_mergeproc = {"clusters" : clusters, 
                     "Beta" : Beta, 
                     "X" : X, 
                     "level" : level, 
                     "gains" : gains, 
                     "merge" : merge, 
                     "labels" : labels,
                     "counts" : counts,
                     "size" : size}
    return(out_mergeproc)
