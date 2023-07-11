import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import zscore
from numpy.linalg import matrix_rank
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import sklearn

def test():
    print('yes')


def estimators(V_obs, thres, TR):
    
    T, N = np.shape(V_obs)
    Cov = np.cov(V_obs, rowvar=False)
    precision = inv(Cov)
    Fx = V_obs - thres
    Fx[Fx < 0] = 0
    tmp = np.hstack((Fx, V_obs))
    B_tmp = np.cov(tmp, rowvar=False)
    B = B_tmp[0:N, N:]
    dV = np.array(((-1 / 2 * V_obs[0:-2, :] + 1 / 2 * V_obs[2:, :])) / TR)
    rowmean = np.mean(dV, axis=0)
    dV = np.vstack([rowmean, dV, rowmean])
    tmp_2 = np.hstack((dV, V_obs))
    dCov = tmp_2[0:N, N + 1 :]

    return Cov, precision, B, dCov


def derivative_123(f, dm, dt):
    
    t = np.arange(dm, len(f) - dm)
    D1, D2, D3 = 0, 0, 0
    d1, d2, d3 = 0, 0, 0
    for n1 in range(1, dm + 1):
        # n1i = n1 - 1
        for n2 in range(n1 + 1, dm + 1):
            # n2i = n2 - 1
            d1 += 1
            D1 += -(
                (
                    f[t - n2] * n1 ** 3
                    - f[t + n2] * n1 ** 3
                    - f[t - n1] * n2 ** 3
                    + f[t + n1] * n2 ** 3
                )
                / (2 * dt * n1 ** 3 * n2 - 2 * dt * n1 * n2 ** 3)
            )
            """
            for n3 in range(n2 + 1, dm + 1):
                #n3i = n3-1
                d3 += 1
                D3 += (3 * (f[t - n3i] * n1 * n2 * (n1 ** 4 - n2 ** 4) + 
                            f[t + n3i] * (-(n1 ** 5 * n2) + n1 * n2 ** 5) + 
                            n3 * ((f[t - n1i] - f[t + n1i]) * n2 * (n2 ** 4 - n3 ** 4) + 
                                  f[t + n2i] * (n1 ** 5 - n1 * n3 ** 4) + f[t - n2i] * (-n1 ** 5 + n1 * n3 ** 4)))) / \
                      (dt ** 3 * n1 * (n1 ** 2 - n2 ** 2) * n3 * (n1 ** 2 - n3 ** 2) * (n2 ** 3 - n2 * n3 ** 2))

            d2 += 1
            D2 += (f[t - n2i] * n1 ** 4 + f[t + n2i] * n1 ** 4 - f[t - n1i] * n2 ** 4 - f[t + n1i] * n2 ** 4 - 
                   2 * f[t] * (n1 ** 4 - n2 ** 4)) / \
                  (dt ** 2 * n2 ** 2 * (n1 ** 4 - n1 ** 2 * n2 ** 2))
            """
    D1 = D1 / d1
    # D2 = D2 / d2
    # D3 = D3 / d3

    return D1, D2, D3


def dCov_numerical(cx, h, dm=4):
    
    T, N = np.shape(cx)
    diff_cx = np.array((cx[1:, :] - cx[0:-1, :]) / h)
    rowmean = np.mean(diff_cx, axis=0)
    diff_cx = np.vstack([diff_cx, rowmean])
    Csample = np.cov(np.hstack((diff_cx, cx)).T)
    dCov1 = Csample[0:N, N : N + N]

    diff_cx = np.array((1 / 2 * cx[2:, :] - (1 / 2 * cx[0:-2, :])) / h)
    rowmean = np.mean(diff_cx, axis=0)
    diff_cx = np.vstack([rowmean, diff_cx, rowmean])
    Csample = np.cov(np.hstack((diff_cx, cx)).T)
    dCov2 = Csample[0:N, N : N + N]

    diff_cx = np.array(
        (-cx[4:, :] + 8 * cx[3:-1, :] - 8 * cx[1:-3, :] + cx[:-4, :]) / (12 * h)
    )
    rowmean = np.mean(diff_cx, axis=0)
    diff_cx = np.vstack([rowmean, rowmean, diff_cx, rowmean, rowmean])
    Csample = np.cov(np.hstack((diff_cx, cx)).T)
    dCov5 = Csample[0:N, N : N + N]

    diff_cx = None
    for i in range(N):
        dx, _, _ = derivative_123(cx[:, i], dm, h)
        if diff_cx is None:
            diff_cx = dx
        else:
            diff_cx = np.c_[diff_cx, dx]
    cx_trunc = cx[dm : T - dm, :]
    Csample = np.cov(np.hstack((diff_cx, cx_trunc)).T)
    dCov_center = Csample[:N, N : N + N]

    return dCov1, dCov2, dCov5, dCov_center


def prctile(x, p):
    
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p - 50) * n / (n - 1) + 50
    p = np.clip(p, 0, 100)
    
    return np.percentile(x, p)


def compute_ddc(ts, TR, d):

    # Input:
    # ts = time series (number of timepoints x number of nodes)
    # TR = time resolution
    # d = derivative to be used, either 2 for dCov2 or c for dCov_center
    # Output:
    # Cov = covariance matrix
    # DDC = dynamical differential covariance
    # Reg_DDC = regularized DDC
    # nl_DDC = non linear DDC estimator
    
    if np.max(ts[0,:])==ts[0,-1] and np.min(ts[0,:])==ts[0,0]:
        ts=ts[1:,:]

    T, N = ts.shape
    V_obs = zscore(ts, ddof=1)
    dCov1, dCov2, _, dCov_center = dCov_numerical(V_obs, TR)
    Cov, Precision, B, _ = estimators(V_obs, 0, TR)

    if matrix_rank(Cov) < len(Cov):
        print("Cov is rank deficient!")

    if d == 2:  # using dCov2

        DDC = dCov2 @ Precision  # Delta L
        nl_DDC = dCov2 @ B  # Delta ReLu

        C = dCov2
        B = Cov
        Reg_DDC = np.zeros(np.shape(C))

        l = 1e-2
        Bb = sklearn.preprocessing.scale(B)

        for n in range(len(C)):
            ci = C[n, :]
            ridgereg = Ridge(alpha=l)
            ridgereg.fit(Bb, ci.T)
            coef = ridgereg.coef_
            Reg_DDC[n, :] = coef.T

    elif d == 'c':  # using dCov_center

        DDC = dCov_center @ Precision  # Delta L
        nl_DDC = dCov_center @ B  # Delta ReLu

        C = dCov_center
        B = Cov
        Reg_DDC = np.zeros(np.shape(C))

        l = 1e-2
        Bb = sklearn.preprocessing.scale(B)
        for n in range(len(C)):
            ci = C[n, :]
            ridgereg = Ridge(alpha=l)
            ridgereg.fit(Bb, ci.T)
            coef = ridgereg.coef_
            Reg_DDC[n, :] = coef.T

    return Cov, DDC, Reg_DDC, nl_DDC
