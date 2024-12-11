import numpy as np
import pandas as pd
import torch
if torch.cuda.is_available():
    cuda = True

import datetime
from scipy.stats import multivariate_normal, norm
from sympy.utilities.iterables import multiset_permutations
from sklearn.decomposition import PCA
from sklearn import cluster as skcluster
from sklearn_extra import cluster
from sklearn.mixture import GaussianMixture as GM

def compute_est_performance(pred, 
                             truth, 
                             history,
                             method_name,
                             dataset_idx,
                             save_to=None,
                             types = ["SMAPE","NRMSE"]):
    
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
    
    result = pd.DataFrame(columns=["method", "datetime", "dataset_idx"]+types)
    result.loc[0, "method"] = method_name
    result.loc[0, "datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result.loc[0, "dataset_idx"] = dataset_idx

    valid_loc = ~np.isnan(pred-truth) * (np.abs(truth) > 0)
    
    if "SMAPE" in types:
        result.loc[0, "SMAPE"] = \
            np.mean(2 * np.abs(pred - truth)[valid_loc] / (np.abs(truth) + np.abs(pred))[valid_loc])
        
    if "NRMSE" in types:
        RMSE = np.sqrt(((pred-truth) ** 2)[valid_loc].mean())
        result.loc[0, "NRMSE"] = RMSE/np.sqrt((truth ** 2)[valid_loc].mean())
    
    if save_to is not None:
        if os.path.exists(save_to):
            existing = pd.read_csv(save_to)
            existing.loc[len(existing.index)] = result.loc[0]
            existing.to_csv(save_to, index=False)
        else:
            result.to_csv(save_to, index=False)
        
    return result


def compute_cluster_performance(est_labels, true_labels):
    if torch.is_tensor(est_labels):
        est_labels = est_labels.detach().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().numpy()
    
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


def compute_cluster_baseline(data, pca_n, true_labels, seed=None, save_to=None, message=True):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    N, M = data.shape
    
    assert N == true_labels.size, "length of true labels must be the same as nrows of data"
    K = np.unique(true_labels).size
    
    if pca_n == M:
        if message:
            print("PCA_n = M; no transformation is performed")
        pca_matrix = data
    else:
        pca = PCA(pca_n)
        pca_matrix = pca.fit_transform(data)
        
    ###GM tied########
    if message:
        print("computing result for PCA MG")
    pca_tiedGM = GM(n_components=K, covariance_type="tied", random_state=seed).fit(pca_matrix)
    
    tiedGM_likelihoods = np.empty(shape=(N, K))
    for k in range(K):
        tiedGM_likelihoods[:, k] = multivariate_normal(mean=pca_tiedGM.means_[k, :],
                                                       cov=pca_tiedGM.covariances_).pdf(pca_matrix)
        
    tiedGM_membership_num = np.multiply(tiedGM_likelihoods, pca_tiedGM.weights_)
    tiedGM_membership_denom = (1/(tiedGM_membership_num.sum(axis=1))).reshape(-1,1)
    pca_tiedGM_membership = np.multiply(tiedGM_membership_num, tiedGM_membership_denom)
    pca_tiedGM_error = compute_cluster_performance(pca_tiedGM_membership, true_labels)
    
 
    #########
    result = pd.DataFrame(columns=["datetime", "dataset_idx", "pca_GM"])
    result.loc[0, "datetime"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result.loc[0, "dataset_idx"] = seed
    result.iloc[0, 2:] = [pca_tiedGM_error]
    
    if save_to is not None:
        if os.path.exists(save_to):
            existing = pd.read_csv(save_to)
            existing.loc[len(existing.index)] = result.loc[0]
            existing.to_csv(save_to, index=False)
        else:
            result.to_csv(save_to, index=False)
    
    return result
