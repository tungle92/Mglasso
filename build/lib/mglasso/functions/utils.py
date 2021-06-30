import numpy as np

# minus and plus lines with reodering the coefficients
def minus_lines(i, j, Beta, ni = 1, nj = 1):
    Y_coeffs = Beta[i, :].copy()
    X_coeffs = Beta[j, :].copy()
    X_i = X_coeffs[i]
    X_coeffs[i] = X_coeffs[j]
    X_coeffs[j] = X_i
    return ni*Y_coeffs - nj*X_coeffs

def plus_lines(i, j, Beta, ni = 1, nj = 1):
    Y_coeffs = Beta[i, :].copy()
    X_coeffs = Beta[j, :].copy()
    X_i = X_coeffs[i]
    X_coeffs[i] = X_coeffs[j]
    X_coeffs[j] = X_i
    return ni*Y_coeffs + nj*X_coeffs

# distances Beta
def dist_beta(Beta, distance = "euclidean"):
    K = Beta.shape[1]
    
    if (K != 1):
        diffs = np.ones((K, K))*np.inf
        for i in range(K-1):
            for j in range(i+1,K):
                diffs[i,j] = np.linalg.norm(minus_lines(i, j, Beta))
    
        if (distance == "relative"):
            Dsum = np.ones((K, K))
            for i in range(K-1):
                for j in range(i+1,K):
                    Dsum[i,j] = np.linalg.norm(Beta[i,:]) + np.linalg.norm(Beta[j,:])

            diffs = diffs/Dsum

    else:
        diffs = np.zeros((1,1))
  
    return diffs