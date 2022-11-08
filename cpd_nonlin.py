import numpy as np


def calc_scatters(K):
    n = K.shape[0]
    K1 = np.cumsum([0] + list(np.diag(K)))
    K2 = np.zeros((n+1, n+1))
    K2[1:, 1:] = np.cumsum(np.cumsum(K, 0), 1) 

    scatters = np.zeros((n, n))

    diagK2 = np.diag(K2)

    i = np.arange(n).reshape((-1,1))
    j = np.arange(n).reshape((1,-1))
    scatters = (K1[1:].reshape((1,-1))-K1[:-1].reshape((-1,1))
                - (diagK2[1:].reshape((1,-1)) + diagK2[:-1].reshape((-1,1)) - K2[1:,:-1].T - K2[:-1,1:]) / ((j-i+1).astype(float) + (j==i-1).astype(float)))
    #print(scatters[0])
    scatters[j<i]=0
    
    return scatters

def dis_func(backtrack,J,n,m,lmin,lmax,I,p):
    count=0
    for k in range(1,m+1):
        #print(count)
        for l in range((k+1)*lmin, n+1):
            tmin = max(k*lmin, l-lmax)
            tmax = l-lmin+1
            c = J[tmin:tmax,l-1].reshape(-1) + I[k-1, tmin:tmax].reshape(-1)
            I[k,l] = np.min(c)
            if backtrack:
                p[k,l] = np.argmin(c)+tmin
                #print(p[2])
        count=count+1
    return backtrack,n,m,I,p

def cpd_nonlin(K, ncp, lmin=1, lmax=100000, backtrack=True, verbose=True,
    out_scatters=None):
    
    
    m = int(ncp) 
    #print(m)
    (n, n1) = K.shape  
    #print(K.shape)
    assert(n == n1), "Kernel matrix awaited."
    assert(n >= (m+1)*lmin)
    assert(n <= (m+1)*lmax)
    assert(lmax >= lmin >= 1)

    #if verbose:
        #print ("Precomputing scatters...")
    J = calc_scatters(K)

    #if verbose:
        #print ("Inferring best change points...")
    
    I = 1e101*np.ones((m+1, n+1))
    #print(I[1])
    I[0, lmin:lmax] = J[0, lmin-1:lmax-1]
    #print(I[0, lmin:lmax])
    #print(J[0, lmin-1:lmax-1])
    if backtrack:
        p = np.zeros((m+1, n+1), dtype=int)
    else:
        p = np.zeros((1,1), dtype=int)
    
    backtrack,n,m,I,p = dis_func(backtrack,J,n,m,lmin,lmax,I,p)
    
    cps = np.zeros(m, dtype=int)

    if backtrack:
        cur = n
        for k in range(m, 0, -1):
            cps[k-1] = p[k, cur]
            cur = cps[k-1]

    scores = I[:, n].copy()
    scores[scores>1e99] = np.inf
    return cps, scores