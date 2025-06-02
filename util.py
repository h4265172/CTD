import numpy as np
import pandas as pd
import torch
import os
#from pathlib import Path

import datetime
from scipy.stats import multivariate_normal
from sympy.utilities.iterables import multiset_permutations
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GM
if torch.cuda.is_available():
    cuda = True
#%%
def compute_est_performance(pred, 
                             truth, 
                             history,
                             method_name,
                             idx,
                             save_to=None,
                             file_name=None,
                             types = ["SMAPE", "NRMSE"]):
    
    #Expect history input containing nan
    assert pred.shape == truth.shape, "pred and truth must have the same dim"
    
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(truth):
        truth = truth.detach().cpu().numpy()
    if torch.is_tensor(history):
        history = history.detach().cpu().numpy()
    
    N = pred.shape[0]
    tau = pred.shape[-1]
    T = history.shape[-1]
    
    result = pd.DataFrame(columns=["method", "datetime", "idx", 
                                   "SMAPE", "NRMSE"])
    result.loc[0, "method"] = method_name
    result.loc[0, "datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result.loc[0, "idx"] = idx
    RMSE = None
    
    valid_loc = ~np.isnan(pred-truth) * (np.abs(truth) > 0)
    
    if "SMAPE" in types:
        result.loc[0, "SMAPE"] = \
            np.mean(2 * np.abs(pred - truth)[valid_loc] / (np.abs(truth) + np.abs(pred))[valid_loc])
        
    if "NRMSE" in types:
        if RMSE is None:
            RMSE = np.sqrt(((pred-truth) ** 2)[valid_loc].mean())
        result.loc[0, "NRMSE"] = RMSE/np.sqrt((truth ** 2)[valid_loc].mean())
    
    if save_to is None:
        return result
    else:
        path = save_to + "/" + file_name
        if os.path.exists(path):
            existing = pd.read_csv(path)
            new_row_idx = len(existing.index)
            for column in result.columns:
                existing.loc[new_row_idx,column] = result.loc[0,column]
            existing.to_csv(path, index=False)
        else:
            #Path(save_to).mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)

#%%
def compute_cluster_performance(est_labels, true_labels):
    if torch.is_tensor(est_labels):
        est_labels = est_labels.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    
    K = np.unique(true_labels).size
    N = true_labels.size
    
    unscaled_accuracy = 0.
    for perm in multiset_permutations(range(K)):
        temp = 0.
        for k in perm:
            if est_labels.ndim == 1:
                temp += np.sum(np.in1d(np.where(est_labels == perm[k]), 
                                                  np.where(true_labels == k)))
            elif est_labels.ndim == 2:
                temp += np.sum(est_labels[true_labels == k, perm[k]])
        if temp > unscaled_accuracy:
            unscaled_accuracy = temp
    
    return 1 - (unscaled_accuracy / N)


#%%
def compute_cluster_baselines(data, pca_n, true_labels,
                              method = "tied",
                              CTD_result=None, seed=None, save_to=None, file_name=None, message=None):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    N, M = data.shape
    
    assert N == true_labels.size, "length of true labels must be the same as nrows of data"
    K = np.unique(true_labels).size
    
    if message:
        print("computing result for PCA K means")
    if pca_n == M:
        if message:
            print("PCA_n = M; no transformation is performed")
        pca_matrix = data
    else:
        pca = PCA(pca_n)
        pca_matrix = pca.fit_transform(data)
    
    
    ###GM tied########
    if method == "tied":
        if message:
            print("computing result for PCA tied MG")

        pca_tiedGM = GM(n_components=K, covariance_type=method, random_state=seed).fit(pca_matrix)
        
        tiedGM_likelihoods = np.empty(shape=(N, K))
        for k in range(K):
            tiedGM_likelihoods[:, k] = multivariate_normal(mean=pca_tiedGM.means_[k, :],
                                                        cov=pca_tiedGM.covariances_).pdf(pca_matrix)
            
        tiedGM_membership_num = np.multiply(tiedGM_likelihoods, pca_tiedGM.weights_)
        tiedGM_membership_denom = (1/(tiedGM_membership_num.sum(axis=1))).reshape(-1,1)
        pca_tiedGM_membership = np.multiply(tiedGM_membership_num, tiedGM_membership_denom)
        error = compute_cluster_performance(pca_tiedGM_membership, true_labels).item()

    else:
        if message:
            print("computing result for PCA full MG")
        pca_fullGM = GM(n_components=K, covariance_type="full", random_state=seed).fit(pca_matrix)
        
        fullGM_likelihoods = np.empty(shape=(N, K))
        for k in range(K):
            fullGM_likelihoods[:, k] = multivariate_normal(mean=pca_fullGM.means_[k, :],
                                                        cov=pca_fullGM.covariances_[k, :]).pdf(pca_matrix)
            
        fullGM_membership_num = np.multiply(fullGM_likelihoods, pca_fullGM.weights_)
        fullGM_membership_denom = (1/(fullGM_membership_num.sum(axis=1))).reshape(-1,1)
        pca_fullGM_membership = np.multiply(fullGM_membership_num, fullGM_membership_denom)
        error = compute_cluster_performance(pca_fullGM_membership, true_labels).item()
    
    #########
    result = pd.DataFrame(columns=["datetime", "idx", "CTD", "GM"])
    result.loc[0, "datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result.loc[0, "idx"] = seed
    result.loc[0, "CTD"] = CTD_result
    result.loc[0, "GM"] = error

    if save_to is None:
        return result
    else:
        path = save_to + "/" + file_name
        if os.path.exists(path):
            existing = pd.read_csv(path)
            new_row_idx = len(existing.index)
            for column in result.columns:
                existing.loc[new_row_idx,column] = result.loc[0,column]
            existing.to_csv(path, index=False)
        else:
            #Path(save_to).mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)
    
# %%
