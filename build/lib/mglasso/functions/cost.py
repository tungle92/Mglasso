import numpy as np

from mglasso.functions.utils import minus_lines

# Cost function
def cost(Beta, X, lambda1=0, lambda2=0):
    p = X.shape[1]
    P2 = 0
    L = 0
    # loss term
    if (p > 1):
        L = np.sum(np.vectorize(lambda i: np.linalg.norm(X[:,i] - X @ Beta[i,:])**2)(range(p)))

        # fuse-group lasso penalty
        for i in range(p-1):
            for j in range(i+1,p):
                P2 = P2 + np.linalg.norm(minus_lines(i, j, Beta))

    # lasso penalty
    P1 = np.sum(np.abs(Beta))
    
    return L + lambda1*P1 + lambda2*P2