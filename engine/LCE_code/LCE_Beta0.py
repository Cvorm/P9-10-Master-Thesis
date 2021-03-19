import numpy as np
from numpy import array, r_
import math
import pandas as pd
import scipy as sp
import scipy.sparse
import random
def LCE(Xs, Xu, A, k, alpha, beta, lamb, epsilon, maxiter, verbose=True):

    n = Xs.shape[0]
    v1 = Xs.shape[1]
    v2 = Xu.shape[1]

    W = abs(np.random.rand(n, k))
    Hs = abs(np.random.rand(k, v1))
    Hu = abs(np.random.rand(k, v2))

    D = sp.sparse.dia_matrix((A.sum(axis=0), 0), A.shape)

    gamma = 1. - alpha
    trXstXs = tr(Xs, Xs)
    trXutXu = tr(Xu, Xu)

    WtW = W.T.dot(W)
    WtXs = W.T.dot(Xs)
    WtXu = W.T.dot(Xu)
    WtWHs = WtW.dot(Hs)
    WtWHu = WtW.dot(Hu)
    DW = D.dot(W)
    AW = A.dot(W)

    itNum = 1
    delta = 2.0 * epsilon

    ObjHist = []

    while True:

        # update H
        Hs_1 = np.divide(
            (alpha * WtXs), np.maximum(alpha * WtWHs + lamb * Hs, 1e-10))
        Hs = np.multiply(Hs, Hs_1)

        Hu_1 = np.divide(
            (gamma * WtXu), np.maximum(gamma * WtWHu + lamb * Hu, 1e-10))
        Hu = np.multiply(Hu, Hu_1)

        # update W
        W_t1 = alpha * Xs.dot(Hs.T) + gamma * Xu.dot(Hu.T) + beta * AW
        W_t2 = alpha * W.dot(Hs.dot(Hs.T)) + gamma * \
            W.dot(Hu.dot(Hu.T)) + beta * DW + lamb * W
        W_t3 = np.divide(W_t1, np.maximum(W_t2, 1e-10))
        W = np.multiply(W, W_t3)

        # calculate objective function
        WtW = W.T.dot(W)
        WtXs = W.T.dot(Xs)
        WtXu = W.T.dot(Xu)
        WtWHs = WtW.dot(Hs)
        WtWHu = WtW.dot(Hu)
        DW = D.dot(W)
        AW = A.dot(W)

        tr1 = alpha * (trXstXs - 2. * tr(Hs, WtXs) + tr(Hs, WtWHs))
        tr2 = gamma * (trXutXu - 2. * tr(Hu, WtXu) + tr(Hu, WtWHu))
        tr3 = beta * (tr(W, DW) - tr(W, AW))
        tr4 = lamb * (np.trace(WtW) + tr(Hs, Hs) + tr(Hu, Hu))

        Obj = tr1 + tr2 + tr3 + tr4
        ObjHist.append(Obj)

        if itNum > 1:
            delta = abs(ObjHist[-1] - ObjHist[-2])
            if verbose:
                print ("Iteration: ", itNum, "Objective: ", Obj, "Delta: ", delta)
            if itNum > maxiter or delta < epsilon:
                break

        itNum += 1

    return W, Hu, Hs, ObjHist

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