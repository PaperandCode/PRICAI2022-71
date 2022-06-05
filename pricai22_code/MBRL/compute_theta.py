import numpy as np
import pandas as pd
import math
from joblib import Parallel, delayed
from functools import partial, wraps
def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)
def compute_bq(blist, noise_term_vector, r, k, q):
    br_bar = blist[r-1]
    value = -br_bar*nCr(r, q) * noise_term_vector[r-q-1]
    for u in range(k-1-q, 0, -1):
        value = value - blist[q+u-1]*nCr(q+u, q)*noise_term_vector[u-1]
    return value
def Algorithm_coeff(noise_term_vector, r, k):
    '''noise_term_vector is a list: [E_nu, E_nu^2, ..., E_nu^r]'''
    blist = np.zeros(r).tolist()
    blist[r-1] = 1/noise_term_vector[r-1] # compute b_r
    for q in range(k-1, 0, -1):
        bq = compute_bq(blist, noise_term_vector, r, k, q)
        blist[q-1] = bq
    return blist

def mult_moments_computation(blist, nu_residual, E_res_m_list, k, r):
    br_bar = blist[r-1]
    value = 1/br_bar * nu_residual ** r
    for q in range(1, k):
        value = value + blist[q-1] * (nu_residual ** q - E_res_m_list[q-1])
    return value

def compute_dml_theta(g_pre, yf_i, g_pre_i, Prob_i):
    theta = np.mean(g_pre) + 1/g_pre.shape[0]*np.sum((yf_i - g_pre_i) / Prob_i)
    return theta
def compute_rcl_theta(g_pre, res_m, yi_minus_gi_given_z_list, r, k, E_res_m_list):
    '''E_res_m_list is estimate E_nu, E_nur_r_list is true E_nu'''
    if k == 1:
        mult_m = 1 / E_res_m_list[r-1] * (res_m ** r)
        theta_0_order = np.mean(g_pre)
        theta_reg_list = np.dot(yi_minus_gi_given_z_list.T, mult_m)/mult_m.shape[0]
        theta = theta_0_order + np.mean(theta_reg_list)
        return theta
    else:
        blist_unknown = Algorithm_coeff(noise_term_vector=E_res_m_list, r=r, k=k)
        mult_m = mult_moments_computation(blist=blist_unknown, nu_residual=res_m, E_res_m_list=E_res_m_list, k=k, r=r)
        theta_0_order = np.mean(g_pre)
        theta_reg_list = np.dot(yi_minus_gi_given_z_list.T, mult_m)/mult_m.shape[0]
        theta = theta_0_order + np.mean(theta_reg_list)
        return theta

def prepare_set_A(idx_i, yf, g_i_pre_given_z, N, sample_times):
    yi_minus_gi_given_z = yf - g_i_pre_given_z
    yi_minus_gi_given_zi = yi_minus_gi_given_z[idx_i]
    idx_not_i = list(set(list(range(0, len(yi_minus_gi_given_z)))) - set(idx_i.tolist()))
    newfun = partial(make_yi_minus_gi_given_z, yi_minus_gi_given_z, yi_minus_gi_given_zi, idx_not_i)
    sampletimes = range(0, sample_times)
    yi_minus_gi_given_z_list = np.array(list(map(newfun, sampletimes))).T
    return yi_minus_gi_given_z_list

def make_yi_minus_gi_given_z(yi_minus_gi_given_z, yi_minus_gi_given_zi, idx_not_i, sampletime):
    cf_xi = np.random.choice(np.squeeze(yi_minus_gi_given_zi.copy()), len(idx_not_i))
    yi_minus_gi_given_z[idx_not_i] = cf_xi
    return yi_minus_gi_given_z
def compute_E_max_r(x, max_r):
    '''moment_1 to moment_max_r'''
    value = []
    for r in range(0, max_r):
        value.append(np.mean(x**(r+1)))
    return value
