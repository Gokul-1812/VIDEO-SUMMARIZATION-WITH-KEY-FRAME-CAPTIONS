import numpy as np
from cpd_nonlin import cpd_nonlin

def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
 
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    #print("scores:",scores)
    N = K.shape[0]
    N2 = N*desc_rate  

    penalties = np.zeros(m+1)
    
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)

    costs = scores/float(N) + penalties
    #print("costs:",costs)
    m_best = np.argmin(costs)
    #print("m_best:",m_best)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)
    
    return (cps,scores2)


def estimate_vmax(K_stable):
    n = K_stable.shape[0]
    vmax = np.trace(centering(K_stable)/n)
    return vmax


def centering(K):
    mean_rows = np.mean(K, 1)[:, np.newaxis]
    return K - mean_rows - mean_rows.T + np.mean(mean_rows)


def eval_score(K, cps):
    N = K.shape[0]
    cps = [0] + list(cps) + [N]
    V1 = 0
    V2 = 0
    for i in range(len(cps)-1):
        K_sub = K[cps[i]:cps[i+1], :][:, cps[i]:cps[i+1]]
        V1 += np.sum(np.diag(K_sub))
        V2 += np.sum(K_sub) / float(cps[i+1] - cps[i])
    return (V1 - V2)


def eval_cost(K, cps, score, vmax):
    N = K.shape[0]
    penalty = (vmax*len(cps)/(2.0*N))*(np.log(float(N)/len(cps))+1)
    return score/float(N) + penalty