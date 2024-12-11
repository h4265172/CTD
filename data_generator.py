import torch
import random
import numpy.random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.manifold import TSNE
#from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.vector_ar.var_model import VARProcess
from statsmodels.tsa.arima_process import ArmaProcess

import util


class Simulate():
    def __init__(self, N, T, M, K, W,
                 cluster_props, centers, Sigmas, tmp_params,
                 p_missing=0., D=1,
                 AR_sd=5., Z_sd=5., seed=1234, torch_device="cuda"):

        self.device = torch_device
        self.seed = seed
        
        self.N = N
        self.T = T
        self.M = M
        self.K = K
        self.W = W
        self.D = D
        self.p_missing = p_missing

        np.random.seed(seed)
        
        self.cluster_props = cluster_props
        self.labels = rnd.choice(np.arange(K), N, p=self.cluster_props)  #if p is None then gives uniform over 1-K
        
        self.centers = centers
        self.Sigmas = Sigmas
        
        self.AR_sd = AR_sd
        self.Z_sd = Z_sd
        
        self.mask = None
        self.Z_obs = None
        
        
        if self.W > 1:
            self.VAR_consts, self.VAR_lag_mats = tmp_params
            assert centers.shape == (K, M * W), "centers do not have dim (K x M*W)"
            assert Sigmas.shape == (K, M * W), "Sigmas do not have dim (K x M*W); genereated clusters have diag cov by default"
            assert self.VAR_consts.shape == (W, M), "VAR params do not have dim (W x M)"
            assert self.VAR_lag_mats.shape[-2:] == (W, W), "lag matrices must have dim (M x max_lag x W x W)"
            
            self.X = np.empty((N, W, M))
            self.Y = np.empty((T, W, M))
            self.Z = np.empty((N, T, W))
            
            
        else:
            self.tmp_params = tmp_params
            assert self.centers.shape[1] == self.tmp_params.shape[1], "inconsistant M between X and Y; check dim of hyperparameters"

            self.X = np.empty((N, M))
            self.Y = np.empty((T, M))
            #self.Y_AR = np.empty((T, M))
            self.Z = np.empty((N, T))
    
    def generate(self):
        np.random.seed(self.seed)
        self._generate_X()
        self._generate_Y()
        self._generate_mask()
        
        if self.W > 1:
            self.Z = np.einsum('njm,tjm->ntj', self.X, self.Y) + \
                              rnd.normal(0, self.Z_sd, 
                                         size=(self.N, self.T, self.W))
        else:
            self.Z = np.dot(self.X, self.Y.T) + rnd.normal(0, self.Z_sd, size=(self.N, self.T))

        if self.mask is not None:
            self.Z_obs = np.where(self.mask == 0, np.nan, self.Z)
            print("X, Y, mask, Z, Z_obs generated")
        else:
            print("X, Y, Z generated")
        
        if self.device is not None:
            self.centers = torch.from_numpy(self.centers).to(self.device)
            self.Sigmas = torch.from_numpy(self.Sigmas).to(self.device)
            if self.cluster_props is not None:
                self.cluster_props = torch.from_numpy(self.cluster_props).to(self.device)
            
            if self.W > 1:
                self.VAR_consts = torch.from_numpy(self.VAR_consts).to(self.device)
                self.VAR_lag_mats = torch.from_numpy(self.VAR_lag_mats).to(self.device)
            else:
                self.tmp_params = torch.from_numpy(self.tmp_params).to(self.device)
            
            self.X = torch.from_numpy(self.X).to(self.device)
            self.Y = torch.from_numpy(self.Y).to(self.device)
            self.Z = torch.from_numpy(self.Z).to(self.device)
            if self.mask is not None:
                self.mask = torch.from_numpy(self.mask).to(self.device)
            if self.Z_obs is not None:
                self.Z_obs = torch.from_numpy(self.Z_obs).to(self.device)
            print("All objects converted to torch tensor")
    
    def _generate_X(self):
        if self.W > 1:
            for k in range(self.K):
                self.X[self.labels==k, :, :] = rnd.multivariate_normal(mean=self.centers[k, :],
                                                                  cov=np.diag(self.Sigmas[k,:] ** 2),
                                                                  size=np.sum(self.labels == k)).reshape(-1, self.W, self.M)
        else:    
            for k in range(self.K):
                self.X[self.labels == k, :] = rnd.multivariate_normal(mean=self.centers[k, :],
                                                                  cov=np.diag(self.Sigmas[k,:]**2),
                                                                  size=np.sum(self.labels == k))
    def _generate_Y(self):
        if self.W > 1:
            for m in range(self.M):
                VAR_generator = VARProcess(self.VAR_lag_mats[m,:,:,:], None, np.diag(self.AR_sd * np.ones(self.W)))
                #assert VAR_generator.isstationary, "{}th AR model is not stationary".format(m + 1)
                self.Y_VAR = self.VAR_consts[:, m] + VAR_generator.simulate_var(steps=self.T)
                self.Y[:, :, m] = self.Y_VAR[:,:]
        else:
            for m in range(self.M):
                AR_generator = ArmaProcess(np.concatenate(([1], -self.tmp_params[1:, m])), np.array([self.AR_sd]))
                #assert AR_generator.isstationary, "{}th AR model is not stationary".format(m + 1)
                self.Y[:, m] = AR_generator.generate_sample(nsample=self.T, burnin=100)
                #self.Y = self.Y_AR[:,:]
        
        for _ in range(self.D):
            self.Y = self.Y.cumsum(axis=0)
        #self.Y = self.Y/self.Y.std(axis=0)
    
    def _generate_mask(self):
        self.mask = rnd.choice(np.array([0, 1]), 
                                   self.N * self.T * self.W, 
                                   p=(self.p_missing, 1-self.p_missing))
        if self.W > 1:
            self.mask = self.mask.reshape(self.N, self.T, self.W)
        else:
            self.mask = self.mask.reshape(self.N, self.T)
