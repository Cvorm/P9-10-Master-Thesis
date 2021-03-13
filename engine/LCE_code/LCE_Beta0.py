import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
import random

def LCE_Beta0(xs, xu, k, alpha, lambdaaa, epsilon, maxiter, verbose):
    np.random.seed(354)
    objhistory = []

    n = np.shape(xs[0])
    v1 = np.shape(xs[1])
    v2 = np.shape(xu[1])

    w = abs(np.random.rand(n,k))
    hs = abs(np.random.rand(k, v1))
    hu = abs(np.random.rand(k, v2))

    beta = 1.0 - alpha
    trxstxs = tr(xs, xs)
    trxutxu = tr(xu, xu)

    wtw = w.conj().T @ w
    wtxs = w.conj().T @ xs
    wtxu = w.conj().T @ xu
    wtwhs = wtw @ hs
    wtwhu = wtw @ hu

    itnum = 1
    delta = 2 * epsilon

    while((delta > epsilon) and (itnum <= maxiter)):
        hs = hs * ((alpha @ wtxs) / max((alpha @ wtwhs + lambdaaa @ hs), 1e-10))
        hu = hu * ((beta @ wtxu) / max((beta @ wtwhu + lambdaaa @ hu), 1e-10))

        w = w * ((alpha @ xs @ hs.conj().T + beta @ xu @ hu.conj().T) / max((alpha @ w @ hs @ hs.conj().T + beta @ w @ hu @ hu.conj().T + lambdaaa @ w), 1e-10))

        wtw = w.conj().T @ w
        wtxs = w.conj().T @ xs
        wtxu = w.conj().T @ xu
        wtwhs = wtw @ hs
        wtwhu = wtw @ hu


        tr1 = alpha @ (trxstxs - 2 * tr(hs,wtxs) + tr(hs, wtwhs))
        tr2 = beta @ (trxutxu - 2 * tr(hu, wtxu) + tr(hu, wtwhu))
        tr3 = lambdaaa @ (np.trace(wtw) + tr(hs, hs) + tr(hu, hu))
        obj = tr1 + tr2 + tr3

        objhistory[itnum] = obj

        if itnum != 1:
            delta = abs(objhistory[itnum] - objhistory[itnum - 1])
    #     if verbose, fprintf('Iteration: %d \t Objective: %f \t Delta: %f \n', itNum, Obj, delta); end
    # else
    #     if verbose, fprintf('Iteration: %d \t Objective: %f \n', itNum, Obj); end
    # end
        itnum = itnum + 1

    return w, hs, hu, objhistory

def tr(a,b):
    trab = np.sum(np.sum(a * b))
    return trab